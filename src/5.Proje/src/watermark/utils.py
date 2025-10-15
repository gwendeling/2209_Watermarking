from __future__ import annotations
from typing import Tuple, Iterable
import numpy as np
from PIL import Image
import pywt
from scipy.fftpack import dct, idct


def read_grayscale(path: str) -> np.ndarray:
	img = Image.open(path).convert("L")
	return np.array(img, dtype=np.float64)


def save_grayscale(arr: np.ndarray, path: str) -> None:
	arr_clip = np.clip(arr, 0, 255).astype(np.uint8)
	Image.fromarray(arr_clip, mode="L").save(path)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
	mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
	if mse == 0:
		return float("inf")
	max_i = 255.0
	return 20.0 * np.log10(max_i / np.sqrt(mse))


def ncc(img1: np.ndarray, img2: np.ndarray) -> float:
	x = img1.astype(np.float64).ravel()
	y = img2.astype(np.float64).ravel()
	xm = x - np.mean(x)
	ym = y - np.mean(y)
	num = np.sum(xm * ym)
	den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
	if den == 0:
		return 0.0
	return float(num / den)


def dwt2_levels(img: np.ndarray, wavelet: str = "haar"):
	# Level 1
	LL1, (LH1, HL1, HH1) = pywt.dwt2(img, wavelet)
	# Level 2 on LL1
	LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, wavelet)
	return (LL1, LH1, HL1, HH1), (LL2, LH2, HL2, HH2)


def idwt2_levels(level1, level2, wavelet: str = "haar") -> np.ndarray:
	LL1, LH1, HL1, HH1 = level1
	LL2, LH2, HL2, HH2 = level2
	# Reconstruct LL1 from level2
	LL1_rec = pywt.idwt2((LL2, (LH2, HL2, HH2)), wavelet)
	# Reconstruct full image
	rec = pywt.idwt2((LL1_rec, (LH1, HL1, HH1)), wavelet)
	return rec


def block_view(arr: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
	H, W = arr.shape
	bh = bw = block_size
	H_pad = (H + bh - 1) // bh * bh
	W_pad = (W + bw - 1) // bw * bw
	padded = np.zeros((H_pad, W_pad), dtype=arr.dtype)
	padded[:H, :W] = arr
	n_rows = H_pad // bh
	n_cols = W_pad // bw
	blocks = padded.reshape(n_rows, bh, n_cols, bw).swapaxes(1, 2)
	return blocks, (H, W)


def unblock_view(blocks: np.ndarray, orig_shape: Tuple[int, int]) -> np.ndarray:
	n_rows, n_cols, bh, bw = blocks.shape
	padded = blocks.swapaxes(1, 2).reshape(n_rows * bh, n_cols * bw)
	H, W = orig_shape
	return padded[:H, :W]


def dct2(block: np.ndarray) -> np.ndarray:
	return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block: np.ndarray) -> np.ndarray:
	return idct(idct(block.T, norm="ortho").T, norm="ortho")
