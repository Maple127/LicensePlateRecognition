from typing import Tuple

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )


def morphology(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    blur = denoise(gray)
    binary = adaptive_threshold(blur)
    morphed = morphology(binary, (3, 3))
    edges = cv2.Canny(morphed, 80, 200)
    return edges

