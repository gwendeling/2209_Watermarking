from __future__ import annotations
from typing import Tuple
import numpy as np
from .utils import dwt2_levels, block_view, dct2


class DwtDctExtractor:
	def __init__(self, block_size: int = 8, wavelet: str = "haar") -> None:
		self.block_size = block_size
		self.wavelet = wavelet

	def _extract_from_band(self, band: np.ndarray, num_bits: int, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> np.ndarray:
		blocks, _ = block_view(band, self.block_size)
		n_rows, n_cols, bh, bw = blocks.shape
		flat_blocks = blocks.reshape(-1, bh, bw)
		bits = []
		for i in range(min(num_bits, len(flat_blocks))):
			B = dct2(flat_blocks[i])
			ax, ay = pos_a
			bx, by = pos_b
			bits.append(1 if B[ax, ay] > B[bx, by] else 0)
		return np.array(bits, dtype=np.uint8)

	def extract(self, watermarked: np.ndarray, wm_shape: Tuple[int, int]) -> np.ndarray:
		(LL1, LH1, HL1, HH1), (LL2, LH2, HL2, HH2) = dwt2_levels(watermarked, wavelet=self.wavelet)
		num_bits = wm_shape[0] * wm_shape[1]
		bits_ll2 = self._extract_from_band(LL2, num_bits, (1, 3), (3, 1))
		bits_lh1 = self._extract_from_band(LH1, num_bits, (2, 4), (4, 2))

		# Majority voting across two bands (tie -> prefer LL2)
		bits = []
		for i in range(num_bits):
			b1 = bits_ll2[i] if i < len(bits_ll2) else 0
			b2 = bits_lh1[i] if i < len(bits_lh1) else 0
			bits.append(1 if (b1 + b2) >= 1 else 0)

		arr = np.array(bits, dtype=np.uint8).reshape(wm_shape)
		return (arr * 255).astype(np.uint8)
