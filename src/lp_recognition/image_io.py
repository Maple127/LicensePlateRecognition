from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """读取图片为 BGR 格式。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def save_image(image: np.ndarray, path: Path) -> Path:
    """保存 BGR 或灰度图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buf = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"保存图片失败: {path}")
    path.write_bytes(buf.tobytes())
    return path


def resize_with_aspect_ratio(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """等比缩放到指定宽高的短边，再填充。"""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (height - new_h) // 2
    bottom = height - new_h - top
    left = (width - new_w) // 2
    right = width - new_w - left
    color = (255, 255, 255)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_dir(path: Path) -> Tuple[Path, bool]:
    """确保目录存在并返回 (路径, 是否新建)。"""
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    return path, not existed

