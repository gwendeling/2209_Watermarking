from __future__ import annotations
from typing import Tuple, List
import numpy as np
from .utils import dwt2_levels, idwt2_levels, block_view, unblock_view, dct2, idct2


class DwtDctEmbedder:
	def __init__(self, block_size: int = 8, wavelet: str = "haar", alpha: float = 5.0) -> None:
		self.block_size = block_size
		self.wavelet = wavelet
		self.alpha = alpha  # embedding strength

	def _embed_in_band(self, band: np.ndarray, bits: np.ndarray, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> Tuple[np.ndarray, List[int]]:
		blocks, orig_shape = block_view(band, self.block_size)
		n_rows, n_cols, bh, bw = blocks.shape
		flat_blocks = blocks.reshape(-1, bh, bw)
		out_blocks = np.copy(flat_blocks)
		used_indices: List[int] = []

		for i, bit in enumerate(bits.flatten()):
			if i >= len(flat_blocks):
				break
			B = dct2(flat_blocks[i])
			ax, ay = pos_a
			bx, by = pos_b
			ca = B[ax, ay]
			cb = B[bx, by]
			# Encode bit by enforcing ca > cb for bit=1 else cb > ca
			if bit > 0:
				if ca <= cb:
					# push difference
					B[ax, ay] = cb + self.alpha
				else:
					B[ax, ay] = ca + self.alpha * 0.25
					B[bx, by] = cb - self.alpha * 0.25
			else:
				if cb <= ca:
					B[bx, by] = ca + self.alpha
				else:
					B[bx, by] = cb + self.alpha * 0.25
					B[ax, ay] = ca - self.alpha * 0.25

			out_blocks[i] = idct2(B)
			used_indices.append(i)

		# Put back into band shape
		out = out_blocks.reshape(n_rows, n_cols, bh, bw)
		out = unblock_view(out, orig_shape)
		return out, used_indices

	def embed(self, cover: np.ndarray, watermark_bits: np.ndarray) -> Tuple[np.ndarray, dict]:
		# DWT 2-levels
		(LL1, LH1, HL1, HH1), (LL2, LH2, HL2, HH2) = dwt2_levels(cover, wavelet=self.wavelet)

		# Redundant embedding: LL2 using (1,3) vs (3,1), LH1 using (2,4) vs (4,2)
		LL2_emb, idx_ll2 = self._embed_in_band(LL2, watermark_bits, (1, 3), (3, 1))
		LH1_emb, idx_lh1 = self._embed_in_band(LH1, watermark_bits, (2, 4), (4, 2))

		# Reconstruct
		watermarked = idwt2_levels(
			(LL1, LH1_emb, HL1, HH1),
			(LL2_emb, LH2, HL2, HH2),
			wavelet=self.wavelet,
		)

		meta = {
			"block_size": self.block_size,
			"wavelet": self.wavelet,
			"alpha": self.alpha,
			"idx_ll2": idx_ll2,
			"idx_lh1": idx_lh1,
			"wm_shape": list(watermark_bits.shape),
		}
		return watermarked, meta
