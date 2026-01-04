from typing import List, Tuple

import cv2
import numpy as np


def deskew(plate: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = plate.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(plate, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def trim_edges(plate: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    h, w = plate.shape[:2]
    dx, dy = int(w * ratio), int(h * ratio)
    return plate[dy : h - dy, dx : w - dx]


def segment_characters(plate: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_regions: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / plate.shape[0] > 0.4 and 0.1 < w / h < 1.0:
            char_regions.append((x, y, w, h))
    char_regions = sorted(char_regions, key=lambda r: r[0])

    chars: List[np.ndarray] = []
    for (x, y, w, h) in char_regions:
        char_img = thresh[y : y + h, x : x + w]
        chars.append(char_img)
    return chars

