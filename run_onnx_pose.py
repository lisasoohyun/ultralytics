"""Run raw three-scale YOLO26 pose ONNX output with host-side decoding and NMS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


NAMES = ("standing", "hands_up", "sitting", "fall_down", "ood", "skeleton")
STRIDES = (8, 16, 32)
SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image into centered square canvas matching Ultralytics inference preprocessing."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_size = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, resized_size, interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - resized_size[0], size - resized_size[1]
    left, top = round(pad_w / 2 - 0.1), round(pad_h / 2 - 0.1)
    right, bottom = round(pad_w / 2 + 0.1), round(pad_h / 2 + 0.1)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)), ratio, (
        left,
        top,
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply numerically stable sigmoid."""
    return np.where(values >= 0, 1 / (1 + np.exp(-values)), np.exp(values) / (1 + np.exp(values)))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Return indices retained by greedy NMS for one class."""
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        index = order[0]
        kept.append(index)
        if order.size == 1:
            break
        remaining = order[1:]
        x1 = np.maximum(boxes[index, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[index, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[index, 2], boxes[remaining, 2])
        y2 = np.minimum(boxes[index, 3], boxes[remaining, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = intersection / (area[index] + area[remaining] - intersection + 1e-7)
        order = remaining[iou <= iou_threshold]
    return np.array(kept, dtype=np.int64)


def decode(outputs: list[np.ndarray], conf_threshold: float, iou_threshold: float, max_det: int) -> np.ndarray:
    """Restore decode layers removed from ONNX: anchors, strides, sigmoid, and class-aware NMS."""
    boxes, scores, class_ids, keypoints = [], [], [], []
    for output, stride in zip(outputs, STRIDES):
        _, channels, height, width = output.shape
        if channels != 61:
            raise ValueError(f"Expected 61 channels [ltrb, 6 classes, 17x3 keypoints], got {channels}")
        grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        anchors = np.stack((grid_x.reshape(-1) + 0.5, grid_y.reshape(-1) + 0.5), axis=1)
        prediction = output[0].transpose(1, 2, 0).reshape(-1, channels)
        class_scores = sigmoid(prediction[:, 4:10])
        ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(ids)), ids]
        selected = confidences >= conf_threshold
        if not selected.any():
            continue

        distances = prediction[selected, :4]
        selected_anchors = anchors[selected] * stride
        boxes.append(
            np.column_stack(
                (
                    selected_anchors[:, 0] - distances[:, 0] * stride,
                    selected_anchors[:, 1] - distances[:, 1] * stride,
                    selected_anchors[:, 0] + distances[:, 2] * stride,
                    selected_anchors[:, 1] + distances[:, 3] * stride,
                )
            )
        )
        raw_keypoints = prediction[selected, 10:].reshape(-1, 17, 3)
        decoded_keypoints = raw_keypoints.copy()
        decoded_keypoints[..., :2] = (decoded_keypoints[..., :2] + anchors[selected, None, :]) * stride
        decoded_keypoints[..., 2] = sigmoid(decoded_keypoints[..., 2])
        keypoints.append(decoded_keypoints)
        scores.append(confidences[selected])
        class_ids.append(ids[selected])

    if not boxes:
        return np.empty((0, 57), dtype=np.float32)
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    class_ids = np.concatenate(class_ids)
    keypoints = np.concatenate(keypoints)
    # Each class-local NMS index must map back to its original prediction.
    kept = np.concatenate(
        [np.flatnonzero(class_ids == cls)[nms(boxes[class_ids == cls], scores[class_ids == cls], iou_threshold)] for cls in np.unique(class_ids)]
    )
    kept = kept[np.argsort(scores[kept])[::-1][:max_det]]
    return np.concatenate((boxes[kept], scores[kept, None], class_ids[kept, None], keypoints[kept].reshape(-1, 51)), axis=1)


def scale_predictions(predictions: np.ndarray, ratio: float, padding: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """Map xyxy boxes and xy keypoints from letterboxed model space to source image space."""
    scaled = predictions.copy()
    left, top = padding
    height, width = shape
    scaled[:, [0, 2]] = np.clip((scaled[:, [0, 2]] - left) / ratio, 0, width)
    scaled[:, [1, 3]] = np.clip((scaled[:, [1, 3]] - top) / ratio, 0, height)
    points = scaled[:, 6:].reshape(-1, 17, 3)
    points[..., 0] = np.clip((points[..., 0] - left) / ratio, 0, width)
    points[..., 1] = np.clip((points[..., 1] - top) / ratio, 0, height)
    return scaled


def draw_predictions(image: np.ndarray, predictions: np.ndarray, kpt_threshold: float) -> np.ndarray:
    """Draw detections and COCO-17 skeleton."""
    for prediction in predictions:
        x1, y1, x2, y2, confidence, class_id = prediction[:6]
        color = (0, 255, 0)
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        cv2.putText(
            image,
            f"{NAMES[int(class_id)]} {confidence:.2f}",
            (round(x1), max(round(y1) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        points = prediction[6:].reshape(17, 3)
        for start, end in SKELETON:
            if min(points[start, 2], points[end, 2]) >= kpt_threshold:
                cv2.line(image, tuple(points[start, :2].round().astype(int)), tuple(points[end, :2].round().astype(int)), color, 2)
        for x, y, visibility in points:
            if visibility >= kpt_threshold:
                cv2.circle(image, (round(x), round(y)), 3, (0, 0, 255), -1)
    return image


def predict(session: ort.InferenceSession, image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Run ONNX Runtime and host-side postprocessing for one BGR frame."""
    input_height, input_width = session.get_inputs()[0].shape[2:]
    if input_height != input_width:
        raise ValueError(f"Only square model inputs supported, got {input_height}x{input_width}")
    letterboxed, ratio, padding = letterbox(image, input_height)
    tensor = np.ascontiguousarray(letterboxed.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    return scale_predictions(decode(outputs, args.conf, args.iou, args.max_det), ratio, padding, image.shape[:2])


def main() -> None:
    """Run image or video source and save annotated output plus per-frame JSON detections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("runs/pose/train-7/export/best.onnx"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/pose/train-7/onnx_predict"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--kpt-conf", type=float, default=0.5)
    args = parser.parse_args()
    if not args.model.is_file() or not args.source.is_file():
        raise FileNotFoundError(f"Model/source not found: {args.model}, {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width, height = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_image = capture.get(cv2.CAP_PROP_FRAME_COUNT) == 1
    destination = args.output / (f"{args.source.stem}.jpg" if is_image else f"{args.source.stem}.mp4")
    writer = None if is_image else cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    records = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        predictions = predict(session, frame, args)
        records.append(predictions.tolist())
        annotated = draw_predictions(frame, predictions, args.kpt_conf)
        if is_image:
            cv2.imwrite(str(destination), annotated)
        else:
            writer.write(annotated)
    capture.release()
    if writer is not None:
        writer.release()
    (args.output / f"{args.source.stem}.json").write_text(json.dumps(records), encoding="utf-8")
    print(f"Saved {destination} and {args.output / f'{args.source.stem}.json'}")


if __name__ == "__main__":
    main()
