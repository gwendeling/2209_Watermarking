from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from config import (
    COVER_IMAGE_PATH,
    WATERMARK_IMAGE_PATH,
    STEGO_IMAGE_PATH,
    BLOCK_SIZE,
    EMBED_STRENGTH,
    DWT_WAVELET,
    CHAOS_MAP,
    CHAOS_SEED_COVER,
    CHAOS_R_COVER,
    CHAOS_SEED_WM,
    CHAOS_R_WM,
)
from utils import dwt2, idwt2, block_view, dct2, idct2
from chaos import chaotic_permutation, scramble_block


@dataclass
class EmbeddingKeys:
    block_perm_seed: float
    block_perm_r: float
    wm_perm_seed: float
    wm_perm_r: float


class DwtHybridEmbedder:
    def __init__(self,
                 block_size: int = BLOCK_SIZE,
                 embed_strength: float = EMBED_STRENGTH,
                 wavelet: str = DWT_WAVELET,
                 chaos_map: str = CHAOS_MAP,
                 keys: EmbeddingKeys | None = None):
        self.block_size = block_size
        self.embed_strength = embed_strength
        self.wavelet = wavelet
        self.chaos_map = chaos_map
        if keys is None:
            keys = EmbeddingKeys(
                block_perm_seed=CHAOS_SEED_COVER,
                block_perm_r=CHAOS_R_COVER,
                wm_perm_seed=CHAOS_SEED_WM,
                wm_perm_r=CHAOS_R_WM,
            )
        self.keys = keys

    def _prepare_images(self, cover_path: Path, watermark_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        cover = Image.open(cover_path).convert("L")
        cover_arr = np.array(cover, dtype=np.float64)

        wm = Image.open(watermark_path).convert("L")
        wm_arr = np.array(wm, dtype=np.uint8)
        # binarize watermark to 0/1
        wm_bits = (wm_arr > 127).astype(np.uint8)
        return cover_arr, wm_bits

    def _embed_block(self, dct_block: np.ndarray, bit: int) -> np.ndarray:
        # Modify relationship between a mid and a high frequency coefficient
        # Choose positions (2,4) and (4,4) in 0-indexed coordinates
        i1, j1 = 2, 4
        i2, j2 = 4, 4
        c1 = dct_block[i1, j1]
        c2 = dct_block[i2, j2]

        # Enforce inequality depending on bit
        T = self.embed_strength
        if bit == 1:
            # Make c1 >= c2 + T
            if c1 < c2 + T:
                delta = (c2 + T) - c1
                dct_block[i1, j1] = c1 + delta
        else:
            # Make c2 >= c1 + T
            if c2 < c1 + T:
                delta = (c1 + T) - c2
                dct_block[i2, j2] = c2 + delta
        return dct_block

    def embed(self,
              cover_path: Path = COVER_IMAGE_PATH,
              watermark_path: Path = WATERMARK_IMAGE_PATH,
              stego_out_path: Path = STEGO_IMAGE_PATH) -> Path:
        cover_arr, wm_bits = self._prepare_images(cover_path, watermark_path)

        # 1-level DWT
        (LL, (LH, HL, HH)) = dwt2(cover_arr, self.wavelet)

        # Prepare block permutation for LL blocks
        h, w = LL.shape
        bs = self.block_size
        # Permutation for within-block pixel scrambling
        perm_block = chaotic_permutation((bs, bs), self.chaos_map, self.keys.block_perm_seed, self.keys.block_perm_r)

        # Prepare blocks
        LL_blocks = block_view(LL, bs)  # view: (bh, bw, bs, bs)
        bh, bw = LL_blocks.shape[0], LL_blocks.shape[1]

        # Flatten watermark bits to match number of blocks
        num_blocks = bh * bw
        wm_flat = wm_bits.reshape(-1)
        if wm_flat.size < num_blocks:
            reps = int(np.ceil(num_blocks / wm_flat.size))
            wm_flat = np.tile(wm_flat, reps)
        wm_flat = wm_flat[:num_blocks]

        # Embed bit per block
        LL_mod = LL.copy()
        for bi in range(bh):
            for bj in range(bw):
                block = LL_blocks[bi, bj].copy()
                block_scrambled = scramble_block(block, perm_block)
                dct_b = dct2(block_scrambled)

                bit = int(wm_flat[bi * bw + bj])
                dct_b = self._embed_block(dct_b, bit)
                idct_b = idct2(dct_b)

                # Unscrambling is not applied here; we leave LL spatial block in scrambled state to benefit diffusion
                LL_mod[bi * bs:(bi + 1) * bs, bj * bs:(bj + 1) * bs] = idct_b

        # Reconstruct image via inverse DWT
        stego = idwt2((LL_mod, (LH, HL, HH)), self.wavelet)
        stego = np.clip(stego, 0, 255)

        stego_img = Image.fromarray(stego.astype(np.uint8))
        stego_img.save(stego_out_path)
        return Path(stego_out_path)


