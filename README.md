# LicensePlateRecognition

## 项目简介
这是一个面向课程作业的 Python 车牌识别小项目，包含图像预处理、车牌定位、倾斜校正、字符分割与 OCR 的完整流程。UI 使用 Tkinter，默认从 `src/main/resources/image` 选择图片，处理结果保存在 `src/main/resources/output`。

## 环境准备
1. Python 3.10+
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 安装系统 Tesseract（用于 OCR）并确保命令 `tesseract` 可用，常见安装方式：
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Windows: 下载并添加到 PATH（https://github.com/tesseract-ocr/tesseract）

## 运行
```bash
python -m src.main
```

## 功能步骤
1. **选择原图**：默认打开 `src/main/resources/image`，支持 JPG/PNG。
2. **处理图像**：完成灰度、高斯模糊、自适应阈值、形态学和 Canny 边缘检测。
3. **定位车牌**：寻找车牌候选轮廓，框选位置并裁剪车牌区域。
4. **车牌优化**：倾斜校正并裁去边缘噪声。
5. **字符分割**：按连通域分割字符并展示。
6. **识别结果**：使用 Tesseract OCR 输出中文省份+字母数字结果。

## 目录说明
- `src/lp_recognition`：核心代码（预处理、检测、分割、OCR、GUI）。
- `src/main/resources/image`：待识别图片（请放置 4 类共 20+ 张示例图）。
- `src/main/resources/output`：处理输出与字符切片。

## 作业要求提示
- 需覆盖四类测试场景，每类至少 5 张图片：
  1. 正常车牌
  2. 明显倾斜/畸变车牌
  3. 有中文广告/其他文字干扰的车牌
  4. 多台电动车同框的车牌（需全部识别）

将对应图片放入 `src/main/resources/image`，运行后按按钮流程完成演示并截图记录识别结果。
