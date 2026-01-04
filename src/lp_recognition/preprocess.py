from typing import Tuple

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    brightened = cv2.convertScaleAbs(image, alpha=1.2, beta=15)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(brightened)


def denoise(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5
    )


def threshold_with_fallback(image: np.ndarray) -> np.ndarray:
    adaptive = adaptive_threshold(image)
    if adaptive.mean() < 50:
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return otsu
    return adaptive


def morphology(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    enhanced = enhance_contrast(gray)
    blur = denoise(enhanced)
    binary = threshold_with_fallback(blur)
    morphed = morphology(binary, (5, 5))
    edges = cv2.Canny(morphed, 50, 150)
    return edges
