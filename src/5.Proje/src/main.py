from __future__ import annotations
import os
import numpy as np
from PIL import Image

from config import DATA_DIR, OUTPUT_DIR, COVER_IMAGE_PATH, WATERMARK_IMAGE_PATH
from watermark.generator import WatermarkGenerator
from watermark.embedder import DwtDctEmbedder
from watermark.extractor import DwtDctExtractor
from watermark.utils import read_grayscale, save_grayscale, psnr, ncc, dwt2_levels, block_view


def ensure_cover(path: str, size: int = 256) -> None:
	if os.path.exists(path):
		return
	# Create a simple grayscale gradient as cover if missing
	x = np.linspace(0, 255, size, dtype=np.float64)
	img = np.tile(x, (size, 1))
	Image.fromarray(img.astype(np.uint8), mode="L").save(path)


def main() -> None:
	# Ensure we have a cover image; user can replace file at configured path
	ensure_cover(COVER_IMAGE_PATH)

	# Generate or reuse watermark image
	if not os.path.exists(WATERMARK_IMAGE_PATH):
		wm_gen = WatermarkGenerator(width=64, height=64)
		wm_img = wm_gen.generate_logo(text="WM")
		wm_gen.save(wm_img, WATERMARK_IMAGE_PATH)
	else:
		wm_img = np.array(Image.open(WATERMARK_IMAGE_PATH).convert("L"), dtype=np.uint8)
		wm_img = (wm_img > 128).astype(np.uint8) * 255

	cover = read_grayscale(COVER_IMAGE_PATH)

	# Compute embedding capacity based on LL2 and LH1 block counts
	(LL1_c, LH1_c, HL1_c, HH1_c), (LL2_c, LH2_c, HL2_c, HH2_c) = dwt2_levels(cover, wavelet="haar")
	bs = 8
	bl_ll2, _ = block_view(LL2_c, bs)
	bl_lh1, _ = block_view(LH1_c, bs)
	cap_ll2 = bl_ll2.shape[0] * bl_ll2.shape[1]
	cap_lh1 = bl_lh1.shape[0] * bl_lh1.shape[1]
	capacity = min(cap_ll2, cap_lh1)
	# Resize watermark to fit capacity (square close to sqrt(capacity))
	target_side = int(np.floor(np.sqrt(capacity)))
	if target_side <= 0:
		raise RuntimeError("Insufficient capacity for watermark embedding.")
	if wm_img.shape[0] != target_side or wm_img.shape[1] != target_side:
		wm_img_resized = Image.fromarray(wm_img).resize((target_side, target_side), resample=Image.NEAREST)
		wm_img = (np.array(wm_img_resized) > 128).astype(np.uint8) * 255

	# Prepare watermark bits as 0/1 array
	wm_bits = (wm_img > 0).astype(np.uint8)

	embedder = DwtDctEmbedder(block_size=8, wavelet="haar", alpha=7.5)
	watermarked, meta = embedder.embed(cover, wm_bits)

	# Save watermarked image
	wm_out_path = os.path.join(OUTPUT_DIR, "watermarked.png")
	save_grayscale(watermarked, wm_out_path)

	# Compute PSNR
	val_psnr = psnr(cover, watermarked)
	print(f"PSNR between cover and watermarked: {val_psnr:.2f} dB")

	# Extract
	extractor = DwtDctExtractor(block_size=meta["block_size"], wavelet=meta["wavelet"]) 
	extracted_wm = extractor.extract(watermarked, tuple(meta["wm_shape"]))

	# Compute NCC between original and extracted watermark
	val_ncc = ncc((wm_bits * 255).astype(np.uint8), extracted_wm)
	print(f"NCC between original and extracted watermark: {val_ncc:.4f}")

	# Save extracted watermark
	extr_path = os.path.join(OUTPUT_DIR, "extracted_watermark.png")
	Image.fromarray(extracted_wm, mode="L").save(extr_path)

	print("Done. Outputs saved to:")
	print(f" - {wm_out_path}")
	print(f" - {extr_path}")


if __name__ == "__main__":
	main()
