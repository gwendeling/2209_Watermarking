from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import WATERMARK_WIDTH, WATERMARK_HEIGHT, WATERMARK_IMAGE_PATH, RANDOM_SEED


class WatermarkGenerator:
    def __init__(self, width: int = WATERMARK_WIDTH, height: int = WATERMARK_HEIGHT):
        self.width = width
        self.height = height

    def generate_metadata_watermark(self, save_path: Path = WATERMARK_IMAGE_PATH) -> Path:
        rng = np.random.default_rng(RANDOM_SEED)
        # Create a simple synthetic metadata-like image: blocks with letters
        img = Image.new("L", (self.width, self.height), color=0)
        draw = ImageDraw.Draw(img)
        # Draw simple text labels (User/Model/Time) in different zones
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw.rectangle([0, 0, self.width // 3, self.height - 1], outline=255, fill=32)
        draw.rectangle([self.width // 3, 0, 2 * self.width // 3, self.height - 1], outline=255, fill=64)
        draw.rectangle([2 * self.width // 3, 0, self.width - 1, self.height - 1], outline=255, fill=96)

        draw.text((4, self.height // 3), "USR", fill=255, font=font)
        draw.text((self.width // 3 + 4, self.height // 3), "MDL", fill=255, font=font)
        draw.text((2 * self.width // 3 + 4, self.height // 3), "TS", fill=255, font=font)

        # Add a few random dots
        for _ in range(200):
            x = int(rng.integers(0, self.width))
            y = int(rng.integers(0, self.height))
            img.putpixel((x, y), 255)

        # Binarize watermark
        img = img.convert("L")
        arr = np.array(img)
        arr = (arr > 127).astype(np.uint8) * 255
        img = Image.fromarray(arr, mode="L")

        save_path = Path(save_path)
        img.save(save_path)
        return save_path


if __name__ == "__main__":
    WatermarkGenerator().generate_metadata_watermark()


