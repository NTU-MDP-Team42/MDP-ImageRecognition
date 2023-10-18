from PIL import Image
from typing import Any
import numpy as np
import supervision as sv

def _open_image(img_path: str) -> Image.Image:
    return Image.open(img_path)

def show_image(img_path: str) -> None:
    _open_image(img_path).show()

def _get_biggest_bbox(results: Any):
    detections: sv.Detections = sv.Detections.from_yolov5(results)
    w = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    h = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    max_bbox_area = max(w * h)
    return detections[detections.area >= max_bbox_area][0]

def _get_labels_from_detection(model, detections):
    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "One": 11,
        "Two": 12,
        "Three": 13,
        "Four": 14,
        "Five": 15,
        "Six": 16,
        "Seven": 17,
        "Eight": 18,
        "Nine": 19,
        "A": 20,
        "B": 21,
        "C": 22,
        "D": 23,
        "E": 24,
        "F": 25,
        "G": 26,
        "H": 27,
        "S": 28,
        "T": 29,
        "U": 30,
        "V": 31,
        "W": 32,
        "X": 33,
        "Y": 34,
        "Z": 35,
        "Up": 36,
        "Down": 37,
        "Right": 38,
        "Left": 39,
        "Up Arrow": 36,
        "Down Arrow": 37,
        "Right Arrow": 38,
        "Left Arrow": 39,
        "Stop": 40
    }
    return [f"{name_to_id[model.model.names[class_id]]}.{model.model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]

def _run_model(model: Any, image: Image.Image, biggest_bbox_only=False):
    results = model(image)
    if biggest_bbox_only:
        biggest_bbox = _get_biggest_bbox(results)
        box_annotator = sv.BoxAnnotator(thickness=4, text_scale=2, text_thickness=4)
        labels = _get_labels_from_detection(model, biggest_bbox)
        filtered_img = Image.fromarray(box_annotator.annotate(np.asarray(image), biggest_bbox, labels=labels))
        return filtered_img
    return Image.fromarray(results.render()[0])

# def _biggest_bbox_label_and_annotate(detections):


def _save_img(img: Image.Image, path: str) -> str:
    try:
        img.save(path)
        return path
    except OSError:
        print(f"Cannot saved to {path}")

def run_model_and_save_augmented(model: Any, img_path: str, output_path: str) -> str:
    return _save_img(
        _run_model(
            model, _open_image(img_path)
        ), output_path
    )

def run_model_and_save_augmented_only_biggest_bbox(model, img_path: str, output_path: str) -> str:
    return _save_img(
        _run_model(
            model, _open_image(img_path), biggest_bbox_only=True
        ) , output_path
    )

def is_right(img_path: str) -> bool:
    bbox = _get_biggest_bbox(models_loading.week9(_open_image(img_path)))
    return bbox.class_id.flat[0] == 0 and bbox.confidence.flat[0] >= 0.9

def is_left(img_path: str) -> bool:
    bbox = _get_biggest_bbox(models_loading.week9(_open_image(img_path)))
    return bbox.class_id.flat[0] == 1 and bbox.confidence.flat[0] >= 0.9