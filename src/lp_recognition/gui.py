from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

from lp_recognition import config
from lp_recognition import detection, image_io, ocr, preprocess, segmentation


class LicensePlateApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("电动自行车车牌识别")
        self.root.configure(bg="#f0f4ff")

        self.original_image = None
        self.preprocessed_image = None
        self.detected_plate: Optional[cv2.typing.MatLike] = None
        self.detected_plates: List[cv2.typing.MatLike] = []
        self.optimized_plate: Optional[cv2.typing.MatLike] = None
        self.characters: List[cv2.typing.MatLike] = []

        self._build_layout()

    def _build_layout(self) -> None:
        title = tk.Label(
            self.root,
            text="电动自行车车牌识别演示",
            font=("Microsoft YaHei", 18, "bold"),
            bg="#5061f0",
            fg="white",
            padx=16,
            pady=10,
        )
        title.pack(fill=tk.X)

        btn_frame = tk.Frame(self.root, bg="#f0f4ff")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="选择原图", command=self.open_image, width=12, bg="#5b8def", fg="white").grid(
            row=0, column=0, padx=5
        )
        tk.Button(btn_frame, text="处理图像", command=self.handle_preprocess, width=12, bg="#5b8def", fg="white").grid(
            row=0, column=1, padx=5
        )
        tk.Button(btn_frame, text="定位车牌", command=self.handle_detection, width=12, bg="#5b8def", fg="white").grid(
            row=0, column=2, padx=5
        )
        tk.Button(btn_frame, text="车牌优化", command=self.handle_optimize, width=12, bg="#5b8def", fg="white").grid(
            row=0, column=3, padx=5
        )
        tk.Button(btn_frame, text="字符分割", command=self.handle_segment, width=12, bg="#5b8def", fg="white").grid(
            row=0, column=4, padx=5
        )
        tk.Button(btn_frame, text="识别结果", command=self.handle_recognize, width=12, bg="#5b8def", fg="white").grid(
            row=0, column=5, padx=5
        )

        grid = tk.Frame(self.root, bg="#f0f4ff")
        grid.pack(padx=10, pady=10)

        self.original_panel = self._create_panel(grid, "原图", 0, 0)
        self.preprocess_panel = self._create_panel(grid, "预处理", 0, 1)
        self.locate_panel = self._create_panel(grid, "车牌定位", 1, 0)
        self.optimize_panel = self._create_panel(grid, "车牌优化", 1, 1)
        self.segment_panel = self._create_panel(grid, "字符分割", 2, 0)

        self.result_label = tk.Label(grid, text="识别结果：", bg="#f0f4ff", font=("Microsoft YaHei", 12, "bold"))
        self.result_label.grid(row=2, column=1, padx=10, pady=10, sticky="w")

    def _create_panel(self, parent: tk.Frame, title: str, row: int, col: int) -> tk.Label:
        frame = tk.Frame(parent, bg="white", bd=2, relief=tk.GROOVE)
        frame.grid(row=row, column=col, padx=10, pady=10)
        tk.Label(frame, text=title, bg="white", font=("Microsoft YaHei", 12, "bold")).pack()
        label = tk.Label(frame, bg="white")
        label.pack(padx=5, pady=5)
        return label

    def open_image(self) -> None:
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp")]
        initialdir = config.IMAGE_DIR
        path = filedialog.askopenfilename(initialdir=initialdir, filetypes=filetypes, title="选择车牌图片")
        if not path:
            return
        try:
            self.original_image = image_io.load_image(Path(path))
            self._show_image(self.original_panel, self.original_image)
        except FileNotFoundError as e:
            messagebox.showerror("错误", str(e))

    def handle_preprocess(self) -> None:
        if self.original_image is None:
            messagebox.showwarning("提示", "请先选择原图")
            return
        self.preprocessed_image = preprocess.preprocess_pipeline(self.original_image)
        self._show_image(self.preprocess_panel, self.preprocessed_image, is_gray=True)

    def handle_detection(self) -> None:
        if self.preprocessed_image is None:
            messagebox.showwarning("提示", "请先完成预处理")
            return
        contours = detection.find_plate_contours(self.preprocessed_image)
        if not contours:
            messagebox.showwarning("提示", "未检测到车牌候选区域")
            return
        plates = detection.extract_all_plates(self.original_image, contours)
        if not plates:
            messagebox.showwarning("提示", "未成功提取车牌图像")
            return
        self.detected_plates = [p for p, _ in plates]
        # 默认选用面积最大的车牌进入后续流程
        plate, (x, y, w, h) = max(plates, key=lambda item: item[0].shape[0] * item[0].shape[1])
        self.detected_plate = plate
        annotated = self.original_image.copy()
        for _, (bx, by, bw, bh) in plates:
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        self._show_image(self.locate_panel, annotated)


    def handle_optimize(self) -> None:
        if self.detected_plate is None:
            messagebox.showwarning("提示", "请先定位车牌")
            return
        deskewed = segmentation.deskew(self.detected_plate)
        trimmed = segmentation.trim_edges(deskewed)
        self.optimized_plate = trimmed
        self._show_image(self.optimize_panel, trimmed)

    def handle_segment(self) -> None:
        if self.optimized_plate is None:
            messagebox.showwarning("提示", "请先完成车牌优化")
            return
        self.characters = segmentation.segment_characters(self.optimized_plate)
        if not self.characters:
            messagebox.showwarning("提示", "未能分割出字符")
            return
        canvas = self._assemble_characters(self.characters)
        self._show_image(self.segment_panel, canvas, is_gray=True)

    def handle_recognize(self) -> None:
        if not self.characters:
            messagebox.showwarning("提示", "请先完成字符分割")
            return
        text = ocr.recognize_text(self.characters)
        self.result_label.config(text=f"识别结果：{text}")
        ocr.save_segments(self.characters, config.OUTPUT_DIR / "segments")

    def _assemble_characters(self, chars: List[cv2.typing.MatLike]) -> cv2.typing.MatLike:
        resized = [cv2.resize(c, (32, 48)) for c in chars]
        spacing = 4
        canvas = 255 * np.ones((48, len(resized) * (32 + spacing), 1), dtype=np.uint8)
        x = 0
        for r in resized:
            canvas[0:48, x : x + 32] = r
            x += 32 + spacing
        return canvas

    def _show_image(self, panel: tk.Label, image, is_gray: bool = False) -> None:
        if is_gray:
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = image_io.resize_with_aspect_ratio(display_img, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        im = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(im)
        panel.configure(image=photo)
        panel.image = photo

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = LicensePlateApp()
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
