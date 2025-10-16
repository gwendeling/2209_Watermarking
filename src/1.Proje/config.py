from pathlib import Path

# Paths (edit these as needed)
COVER_IMAGE_PATH: Path = Path("cover_input.png")
WATERMARK_IMAGE_PATH: Path = Path("watermark_input.png")
STEGO_IMAGE_PATH: Path = Path("stego_output.png")
EXTRACTED_WATERMARK_PATH: Path = Path("extracted_watermark.png")

# General parameters
BLOCK_SIZE: int = 8  # DCT block size
EMBED_STRENGTH: float = 6.0  # Threshold T for coefficient inequality embedding

# DWT parameters
DWT_WAVELET: str = "haar"

# Chaotic scrambling parameters
# Two different seeds/keys for cover scrambling and watermark scrambling
CHAOS_MAP: str = "logistic"  # "logistic" or "tent"
CHAOS_SEED_COVER: float = 0.387  # in (0,1)
CHAOS_R_COVER: float = 3.91  # for logistic map r in (3.57, 4)

CHAOS_SEED_WM: float = 0.731  # in (0,1)
CHAOS_R_WM: float = 3.86  # for logistic map

# Watermark size (pixels). Will be embedded as binary (0/1) image.
WATERMARK_WIDTH: int = 64
WATERMARK_HEIGHT: int = 64

# Random seed for any synthetic generation
RANDOM_SEED: int = 42


