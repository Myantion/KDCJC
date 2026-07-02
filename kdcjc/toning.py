from __future__ import annotations

import numpy as np

from kdcjc.core import Block


def toning_alpha_from_seed(seed: int) -> float:
    """由 seed 派生分堆色调强度。"""
    return 0.30 + (seed % 21) / 100.0


def _pile_means_from_block_means(
    block_means: np.ndarray,
    pile_labels: list[int],
    n_piles: int,
) -> np.ndarray:
    pile_sums = np.zeros((n_piles, 3), dtype=np.float64)
    counts = np.zeros(n_piles, dtype=np.float64)
    for block_id, pile_id in enumerate(pile_labels):
        pile_sums[pile_id] += block_means[block_id]
        counts[pile_id] += 1.0
    for pile_id in range(n_piles):
        if counts[pile_id] > 0:
            pile_sums[pile_id] /= counts[pile_id]
    return pile_sums


def compute_block_deltas(blocks: list[Block], pile_labels: list[int], alpha: float) -> np.ndarray:
    """计算每块 RGB 偏移（int8，偶数位/高7位），与稀疏 LSB 隐写兼容。"""
    n_piles = max(pile_labels) + 1
    block_means = np.array(
        [(block.data & 0xFE).mean(axis=(0, 1)) for block in blocks],
        dtype=np.float64,
    )
    pile_means = _pile_means_from_block_means(block_means, pile_labels, n_piles)
    deltas = np.zeros((len(blocks), 3), dtype=np.int32)
    for block_id, block in enumerate(blocks):
        pile_id = pile_labels[block_id]
        raw = np.rint(alpha * (pile_means[pile_id] - block_means[block_id])).astype(np.int32)
        even = (block.data & 0xFE).astype(np.int32)
        for channel in range(3):
            values = even[:, :, channel]
            raw[channel] = int(np.clip(raw[channel], values.min() - 0, 254 - values.max()))
            if raw[channel] % 2 != 0:
                raw[channel] -= 1 if raw[channel] > 0 else -1
        deltas[block_id] = np.clip(raw, -126, 126)
    return deltas.astype(np.int8)


def apply_pile_toning(blocks: list[Block], deltas: np.ndarray) -> None:
    for block_id, block in enumerate(blocks):
        lsb = block.data & 1
        even = block.data.astype(np.int32) & 0xFE
        shift = deltas[block_id].astype(np.int32).reshape(1, 1, 3)
        block.data = (even + shift).astype(np.uint8) | lsb.astype(np.uint8)


def remove_pile_toning_tiles(tiles: list[Block], tile_block_ids: list[int], deltas: np.ndarray) -> None:
    for tile_index, block_id in enumerate(tile_block_ids):
        lsb = tiles[tile_index].data & 1
        even = tiles[tile_index].data.astype(np.int32) & 0xFE
        shift = deltas[block_id].astype(np.int32).reshape(1, 1, 3)
        tiles[tile_index].data = (even - shift).astype(np.uint8) | lsb.astype(np.uint8)


def pack_toning_deltas(deltas: np.ndarray) -> bytes:
    return deltas.astype(np.int8, copy=False).reshape(-1).tobytes()


def unpack_toning_deltas(payload: bytes, block_count: int) -> np.ndarray:
    expected = block_count * 3
    if len(payload) < expected:
        raise ValueError("像素内色调偏移数据已损坏")
    return np.frombuffer(payload[:expected], dtype=np.int8).reshape(block_count, 3)
