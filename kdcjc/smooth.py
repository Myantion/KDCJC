from __future__ import annotations

import struct
import zlib

import numpy as np

from kdcjc.core import Block

SMOOTH_EXACT_ZSTD = 0
SMOOTH_POOL = 3

_HALF_LO = -65
_HALF_HI = 66


def _block_mean_even(even: np.ndarray) -> np.ndarray:
    mean = even.mean(axis=(0, 1)).astype(np.int16)
    mean = np.clip(mean, 0, 254)
    mean = mean - (mean & 1)
    return np.broadcast_to(mean.reshape(1, 1, 3), even.shape).copy()


def smooth_block_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lsb = data & 1
    even = (data & 0xFE).astype(np.int16)
    flattened = _block_mean_even(even)
    residual = (even - flattened).astype(np.int16)
    smoothed = flattened.astype(np.uint8) | lsb.astype(np.uint8)
    return smoothed, residual


def apply_block_smooth(blocks: list[Block]) -> np.ndarray:
    residuals = np.zeros((len(blocks),) + blocks[0].data.shape, dtype=np.int16)
    for block_id, block in enumerate(blocks):
        block.data, residuals[block_id] = smooth_block_data(block.data)
    return residuals


def remove_block_smooth_tiles(
    tiles: list[Block],
    tile_block_ids: list[int],
    residuals: np.ndarray,
) -> None:
    for tile_index, block_id in enumerate(tile_block_ids):
        lsb = tiles[tile_index].data & 1
        even = (tiles[tile_index].data & 0xFE).astype(np.int16)
        even = even + residuals[block_id].astype(np.int16)
        tiles[tile_index].data = np.clip(even, 0, 254).astype(np.uint8) | lsb.astype(np.uint8)


def _residuals_to_half(residuals: np.ndarray) -> np.ndarray:
    if not np.all(residuals.astype(np.int16) % 2 == 0):
        raise ValueError("块内平滑残差必须为偶数")
    return (residuals.astype(np.int16) // 2).astype(np.int16)


def _half_to_residuals(half: np.ndarray) -> np.ndarray:
    return (half.astype(np.int16) * 2).astype(np.int16)


def _pool_half(half: np.ndarray, pool: int) -> np.ndarray:
    block_count, block_size, _, channels = half.shape
    grid = block_size // pool
    pooled = np.zeros((block_count, grid, grid, channels), dtype=np.int8)
    for block_id in range(block_count):
        block = half[block_id].astype(np.int16)
        for gy in range(grid):
            for gx in range(grid):
                cell = block[gy * pool : (gy + 1) * pool, gx * pool : (gx + 1) * pool]
                pooled[block_id, gy, gx] = np.clip(np.rint(cell.mean()), -127, 127).astype(np.int8)
    return pooled


def _upsample_pooled(pooled: np.ndarray, block_size: int, pool: int) -> np.ndarray:
    block_count, grid, _, channels = pooled.shape
    up = np.repeat(np.repeat(pooled.astype(np.int16), pool, axis=1), pool, axis=2)
    return up[:, :block_size, :block_size, :]


def _pick_pool_size(residuals: np.ndarray, max_bytes: int) -> int:
    half = _residuals_to_half(residuals)
    for pool in (4, 8, 16):
        if block_size := half.shape[1]:
            if block_size % pool != 0:
                continue
            pooled = _pool_half(half, pool)
            body = struct.pack(">B", pool) + pooled.tobytes()
            packed = struct.pack(">BI", SMOOTH_POOL, len(body)) + zlib.compress(body, level=9)
            if len(packed) <= max_bytes:
                return pool
    raise ValueError(
        f"像素 LSB 容量不足（平滑元数据超出约 {max_bytes} 字节）。"
        "请关闭块内平滑或增大图片尺寸。"
    )


def pack_smooth_residuals(residuals: np.ndarray, max_bytes: int = 90000) -> bytes:
    half = _residuals_to_half(residuals)
    try:
        import zstandard as zstd

        exact_body = zstd.ZstdCompressor(level=19).compress(half.astype(np.int8).tobytes())
    except ImportError:
        exact_body = zlib.compress(half.astype(np.int8).tobytes(), level=9)
    meta_exact = struct.pack(">BI", SMOOTH_EXACT_ZSTD, len(exact_body)) + exact_body
    if len(meta_exact) <= max_bytes:
        return meta_exact

    pool = _pick_pool_size(residuals, max_bytes)
    pooled = _pool_half(half, pool)
    body = struct.pack(">B", pool) + pooled.tobytes()
    return struct.pack(">BI", SMOOTH_POOL, len(body)) + zlib.compress(body, level=9)


def unpack_smooth_residuals(
    payload: bytes,
    block_count: int,
    block_size: int,
    plane_payload: bytes | None = None,
) -> np.ndarray:
    del plane_payload
    if len(payload) < 5:
        raise ValueError("块内平滑残差数据已损坏")
    mode = payload[0]
    expected = block_count * block_size * block_size * 3

    if mode == SMOOTH_EXACT_ZSTD:
        body_len = struct.unpack_from(">I", payload, 1)[0]
        body = payload[5 : 5 + body_len]
        if len(body) != body_len:
            raise ValueError("块内平滑残差数据已损坏")
        try:
            import zstandard as zstd

            raw = zstd.ZstdDecompressor().decompress(body)
        except ImportError:
            raw = zlib.decompress(body)
        half = np.frombuffer(raw, dtype=np.int8).astype(np.int16)
        if half.size != expected:
            raise ValueError("块内平滑残差数据已损坏")
        return _half_to_residuals(half.reshape(block_count, block_size, block_size, 3))

    if mode == SMOOTH_POOL:
        body_len = struct.unpack_from(">I", payload, 1)[0]
        body = zlib.decompress(payload[5 : 5 + body_len])
        pool = body[0]
        if pool <= 0 or block_size % pool != 0:
            raise ValueError("块内平滑残差数据已损坏")
        grid = block_size // pool
        pooled = np.frombuffer(body[1:], dtype=np.int8).reshape(block_count, grid, grid, 3)
        half = _upsample_pooled(pooled, block_size, pool)
        return _half_to_residuals(half)

    raise ValueError("块内平滑残差数据已损坏")
