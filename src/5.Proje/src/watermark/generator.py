from __future__ import annotations
from typing import Tuple
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class WatermarkGenerator:
	def __init__(self, width: int = 64, height: int = 64, seed: int | None = 42) -> None:
		self.width = width
		self.height = height
		self.seed = seed

	def generate_logo(self, text: str = "WM", invert: bool = False) -> np.ndarray:
		"""
		Generate a simple high-contrast square watermark with centered text.
		Returns a binary (0/255) numpy array of shape (H, W).
		"""
		img = Image.new("L", (self.width, self.height), color=0 if not invert else 255)
		draw = ImageDraw.Draw(img)

		# Try to load a default font; fallback to basic if not available
		try:
			font = ImageFont.load_default()
		except Exception:
			font = None

		# Compute text bounding box for sizing
		bbox = draw.textbbox((0, 0), text, font=font)
		text_w = bbox[2] - bbox[0]
		text_h = bbox[3] - bbox[1]
		x = (self.width - text_w) // 2
		y = (self.height - text_h) // 2

		fg = 255 if not invert else 0
		draw.text((x, y), text, fill=fg, font=font)

		arr = np.array(img, dtype=np.uint8)
		# Binarize to ensure crisp edges
		arr = (arr > 128).astype(np.uint8) * 255
		return arr

	def save(self, watermark: np.ndarray, path: str) -> None:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		Image.fromarray(watermark).save(path)
