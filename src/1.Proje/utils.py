from __future__ import annotations

import numpy as np
import pywt
from scipy.fftpack import dct, idct


def psnr(original: np.ndarray, compared: np.ndarray) -> float:
    original = original.astype(np.float64)
    compared = compared.astype(np.float64)
    mse = np.mean((original - compared) ** 2)
    if mse == 0:
        return 99.0
    PIXEL_MAX = 255.0
    return 20.0 * np.log10(PIXEL_MAX / np.sqrt(mse))


def dwt2(image: np.ndarray, wavelet: str):
    return pywt.dwt2(image, wavelet)


def idwt2(coeffs, wavelet: str):
    return pywt.idwt2(coeffs, wavelet)


def block_view(arr: np.ndarray, block_size: int):
    h, w = arr.shape
    bh, bw = block_size, block_size
    h_trim = (h // bh) * bh
    w_trim = (w // bw) * bw
    arr = arr[:h_trim, :w_trim]
    shape = (h_trim // bh, w_trim // bw, bh, bw)
    strides = (arr.strides[0] * bh, arr.strides[1] * bw, arr.strides[0], arr.strides[1])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


