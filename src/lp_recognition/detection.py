from typing import List, Tuple

import cv2
import numpy as np


def find_plate_contours(processed: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[np.ndarray] = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        area = w * h
        if 2.0 < aspect < 6.5 and area > 500:
            candidates.append(cnt)
    return candidates


def extract_plate(image: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])
    if width == 0 or height == 0:
        raise ValueError("无效的矩形区域")

    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    if height > width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        width, height = height, width

    x, y, w, h = cv2.boundingRect(box)
    return warped, (x, y, w, h)


def extract_all_plates(image: np.ndarray, contours: List[np.ndarray]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    plates = []
    for contour in contours:
        try:
            plate, bbox = extract_plate(image, contour)
            plates.append((plate, bbox))
        except ValueError:
            continue
    return plates
