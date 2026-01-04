import pathlib

# 根目录
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# 图片存放目录
IMAGE_DIR = PROJECT_ROOT / "src" / "main" / "resources" / "image"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "src" / "main" / "resources" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GUI 展示尺寸
DISPLAY_WIDTH = 360
DISPLAY_HEIGHT = 240

