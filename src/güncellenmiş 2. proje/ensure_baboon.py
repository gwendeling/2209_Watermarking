import os
import io
import urllib.request
import cv2
import numpy as np


def download_to_memory(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read()


def save_png_from_buffer(buf: bytes, out_path: str) -> bool:
    data = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return False
    ok = cv2.imwrite(out_path, img)
    return bool(ok)


def create_placeholder(out_path: str) -> None:
    img = np.full((512, 512, 3), 235, dtype=np.uint8)
    for i in range(0, 512, 32):
        cv2.line(img, (i, 0), (i, 512), (210, 210, 210), 1)
        cv2.line(img, (0, i), (512, i), (210, 210, 210), 1)
    cv2.putText(img, "Placeholder Baboon", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def ensure_baboon(png_path: str = "baboon.png") -> None:
    if os.path.exists(png_path):
        print(f"[ensure_baboon] Found existing: {png_path}")
        return

    urls = [
        # OpenCV sample image (jpg) - we will re-encode as PNG
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
        # Backup URL (same file via GitHub CDN)
        "https://github.com/opencv/opencv/raw/master/samples/data/baboon.jpg",
    ]

    for url in urls:
        try:
            print(f"[ensure_baboon] Trying to download from: {url}")
            buf = download_to_memory(url)
            if save_png_from_buffer(buf, png_path):
                print(f"[ensure_baboon] Saved: {png_path}")
                return
        except Exception as e:
            print(f"[ensure_baboon] Download failed from {url}: {e}")

    print("[ensure_baboon] Falling back to placeholder image.")
    create_placeholder(png_path)
    print(f"[ensure_baboon] Placeholder saved at: {png_path}")


if __name__ == "__main__":
    ensure_baboon("baboon.png")


