from pathlib import Path
from extractor import DwtHybridExtractor
from config import STEGO_IMAGE_PATH, EXTRACTED_WATERMARK_PATH, WATERMARK_WIDTH, WATERMARK_HEIGHT


if __name__ == "__main__":
    extractor = DwtHybridExtractor()
    out = extractor.extract(
        stego_path=Path(STEGO_IMAGE_PATH),
        out_w=WATERMARK_WIDTH,
        out_h=WATERMARK_HEIGHT,
        save_path=Path(EXTRACTED_WATERMARK_PATH),
    )
    print(f"Extracted watermark saved to: {out}")


