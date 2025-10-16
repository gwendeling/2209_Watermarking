from __future__ import annotations

import numpy as np
from typing import Tuple


def _logistic_sequence(length: int, seed: float, r: float) -> np.ndarray:
    x = np.empty(length, dtype=np.float64)
    x0 = np.clip(seed, 1e-9, 1 - 1e-9)
    x[0] = x0
    for i in range(1, length):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def _tent_sequence(length: int, seed: float, mu: float) -> np.ndarray:
    # mu typically in (1, 2]
    x = np.empty(length, dtype=np.float64)
    x0 = np.clip(seed, 1e-9, 1 - 1e-9)
    x[0] = x0
    for i in range(1, length):
        x_prev = x[i - 1]
        if x_prev < 0.5:
            x[i] = mu * x_prev
        else:
            x[i] = mu * (1.0 - x_prev)
    # Normalize to (0,1)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x


def chaotic_permutation(shape: Tuple[int, int], map_name: str, seed: float, r_or_mu: float) -> np.ndarray:
    """
    Create a deterministic permutation array of size N (flattened) using chaotic map sorting.
    Returns an index array perm such that arr_flat[perm] gives scrambled order.
    """
    h, w = shape
    n = h * w
    if map_name.lower() == "logistic":
        seq = _logistic_sequence(n, seed, r_or_mu)
    elif map_name.lower() == "tent":
        seq = _tent_sequence(n, seed, r_or_mu)
    else:
        raise ValueError("Unsupported chaotic map: %s" % map_name)
    # argsort gives permutation indices; tie-break with stable kind
    perm = np.argsort(seq, kind="mergesort")
    return perm.astype(np.int64)


def scramble_block(block: np.ndarray, perm: np.ndarray) -> np.ndarray:
    h, w = block.shape
    flat = block.reshape(-1)
    scrambled = flat[perm]
    return scrambled.reshape(h, w)


def unscramble_block(block: np.ndarray, perm: np.ndarray) -> np.ndarray:
    h, w = block.shape
    flat = block.reshape(-1)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size)
    descrambled = flat[inv_perm]
    return descrambled.reshape(h, w)


