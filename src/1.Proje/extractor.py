from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from config import (
    STEGO_IMAGE_PATH,
    EXTRACTED_WATERMARK_PATH,
    WATERMARK_WIDTH,
    WATERMARK_HEIGHT,
    BLOCK_SIZE,
    EMBED_STRENGTH,
    DWT_WAVELET,
    CHAOS_MAP,
    CHAOS_SEED_COVER,
    CHAOS_R_COVER,
)
from utils import dwt2, block_view, dct2
from chaos import chaotic_permutation, scramble_block


@dataclass
class ExtractionKeys:
    block_perm_seed: float
    block_perm_r: float


class DwtHybridExtractor:
    def __init__(self,
                 block_size: int = BLOCK_SIZE,
                 embed_strength: float = EMBED_STRENGTH,
                 wavelet: str = DWT_WAVELET,
                 chaos_map: str = CHAOS_MAP,
                 keys: ExtractionKeys | None = None):
        self.block_size = block_size
        self.embed_strength = embed_strength
        self.wavelet = wavelet
        self.chaos_map = chaos_map
        if keys is None:
            keys = ExtractionKeys(
                block_perm_seed=CHAOS_SEED_COVER,
                block_perm_r=CHAOS_R_COVER,
            )
        self.keys = keys

    def _extract_bit(self, dct_block: np.ndarray) -> int:
        i1, j1 = 2, 4
        i2, j2 = 4, 4
        c1 = dct_block[i1, j1]
        c2 = dct_block[i2, j2]
        return 1 if (c1 - c2) >= 0 else 0

    def extract(self,
                stego_path: Path = STEGO_IMAGE_PATH,
                out_w: int = WATERMARK_WIDTH,
                out_h: int = WATERMARK_HEIGHT,
                save_path: Path = EXTRACTED_WATERMARK_PATH) -> Path:
        stego = Image.open(stego_path).convert("L")
        arr = np.array(stego, dtype=np.float64)

        (LL, (LH, HL, HH)) = dwt2(arr, self.wavelet)
        bs = self.block_size
        perm_block = chaotic_permutation((bs, bs), self.chaos_map, self.keys.block_perm_seed, self.keys.block_perm_r)

        LL_blocks = block_view(LL, bs)
        bh, bw = LL_blocks.shape[0], LL_blocks.shape[1]

        bits = []
        for bi in range(bh):
            for bj in range(bw):
                block = LL_blocks[bi, bj].copy()
                block_scrambled = scramble_block(block, perm_block)
                dct_b = dct2(block_scrambled)
                bit = self._extract_bit(dct_b)
                bits.append(bit)

        bits = np.array(bits, dtype=np.uint8)
        total = out_w * out_h
        if bits.size < total:
            reps = int(np.ceil(total / bits.size))
            bits = np.tile(bits, reps)
        bits = bits[:total]
        wm = (bits.reshape(out_h, out_w) * 255).astype(np.uint8)
        Image.fromarray(wm, mode="L").save(save_path)
        return Path(save_path)


