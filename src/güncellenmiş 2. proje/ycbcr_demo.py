import cv2
import numpy as np
from datetime import datetime
import os

from watermark_embedder import WatermarkEmbedder
from image_metrics import ImageMetrics


def load_baboon(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{path}'. Make sure 'baboon.png' exists.")
    return image


def jpeg_attack(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Apply JPEG compression attack and decode back to BGR.
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", image, encode_params)
    if not ok:
        raise RuntimeError("JPEG encode failed during attack")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed during attack")
    return decoded


def ncc_on_y_channel(img_a_bgr: np.ndarray, img_b_bgr: np.ndarray) -> float:
    """
    Compute NCC on Y channel of YCrCb color space.
    """
    a_ycrcb = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2YCrCb)
    b_ycrcb = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2YCrCb)
    a_y = a_ycrcb[:, :, 0]
    b_y = b_ycrcb[:, :, 0]
    return ImageMetrics.calculate_ncc(a_y, b_y)


def main():
    baboon_path = "baboon.png"
    out_watermarked = "baboon_watermarked.png"
    out_attacked = "baboon_attacked.jpg"

    # 1) Load image
    original_bgr = load_baboon(baboon_path)

    # 2) Prepare metadata
    metadata = {
        "user": "user_12345",
        "model": "stable-diffusion",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d"),
    }

    # 3) Embed watermark in Y channel (YCbCr/YC r Cb in OpenCV)
    embedder = WatermarkEmbedder()
    watermarked_bgr = embedder.embed_from_array(original_bgr, metadata)
    cv2.imwrite(out_watermarked, watermarked_bgr)

    # 4) Apply attack (JPEG compression)
    attacked_bgr = jpeg_attack(watermarked_bgr, quality=50)
    cv2.imwrite(out_attacked, attacked_bgr)

    # 5) Measure NCC on Y channel between watermarked and attacked
    ncc_value = ncc_on_y_channel(watermarked_bgr, attacked_bgr)
    print(f"NCC (Y channel) between watermarked and attacked: {ncc_value:.4f}")


if __name__ == "__main__":
    main()


