from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:  # pragma: no cover - 提示用户安装
    pytesseract = None


def recognize_text(char_images: List[np.ndarray]) -> str:
    if pytesseract is None:
        return "需要安装 pytesseract 及系统 tesseract 可执行文件"

    texts = []
    for img in char_images:
        resized = cv2.resize(img, (32, 48), interpolation=cv2.INTER_CUBIC)
        config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789广州深圳北京上海苏浙粤湘赣闽鲁川京津渝辽吉黑冀皖鄂桂云贵琼甘陕晋蒙宁青新藏港澳"
        text = pytesseract.image_to_string(resized, lang="chi_sim", config=config)
        texts.append(text.strip())
    return "".join(texts)


def save_segments(char_images: List[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(char_images):
        path = output_dir / f"char_{idx}.png"
        cv2.imwrite(str(path), img)

