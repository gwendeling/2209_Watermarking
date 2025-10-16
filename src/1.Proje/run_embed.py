from pathlib import Path
import numpy as np
from PIL import Image

from config import COVER_IMAGE_PATH, WATERMARK_IMAGE_PATH, STEGO_IMAGE_PATH
from embedder import DwtHybridEmbedder
from watermark_generator import WatermarkGenerator
from utils import psnr


def ensure_cover_exists(path: Path):
    if not Path(path).exists():
        # Create a simple grayscale gradient cover if none provided
        w, h = 512, 512
        x = np.linspace(0, 255, w, dtype=np.float64)
        img = np.tile(x, (h, 1))
        Image.fromarray(img.astype(np.uint8)).save(path)


if __name__ == "__main__":
    # Generate example watermark if not exists
    if not Path(WATERMARK_IMAGE_PATH).exists():
        WatermarkGenerator().generate_metadata_watermark(WATERMARK_IMAGE_PATH)

    # Ensure cover exists or create synthetic
    ensure_cover_exists(COVER_IMAGE_PATH)

    # Embed
    embedder = DwtHybridEmbedder()
    out_path = embedder.embed(COVER_IMAGE_PATH, WATERMARK_IMAGE_PATH, STEGO_IMAGE_PATH)

    # PSNR
    cover = np.array(Image.open(COVER_IMAGE_PATH).convert("L"), dtype=np.uint8)
    stego = np.array(Image.open(out_path).convert("L"), dtype=np.uint8)
    value = psnr(cover, stego)
    print(f"PSNR between cover and stego: {value:.2f} dB")


