from __future__ import annotations

import hashlib

import numpy as np


def pixel_msb_key(array: np.ndarray, *, dual_plane: bool = False) -> bytes:
    """从像素高 7 位（MSB）派生 AES 密钥；LSB 隐写不改变此值。"""
    del dual_plane
    stable = array.astype(np.uint8, copy=False) & 0xFE
    return hashlib.sha256(stable.tobytes()).digest()


def spread_rng_seed(pixel_key: bytes, width: int, height: int, stream_id: int) -> bytes:
    return hashlib.sha256(
        pixel_key
        + b"KDCJC-SPREAD-V1"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + stream_id.to_bytes(4, "big")
    ).digest()
