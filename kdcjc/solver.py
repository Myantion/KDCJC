from __future__ import annotations

import hashlib
import random

import numpy as np

from kdcjc.core import (
    Block,
    KDCJCParams,
    ProgressCallback,
    SOLVER_MAX_BLOCKS,
    _edge_band_cost,
    _holistic_pair_cost,
    _split_blocks,
)

_RESTARTS = 6
_CANDIDATE_POOL = 64


def _rotate_data(data: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(data, turns % 4)


def _image_hash(array: np.ndarray) -> bytes:
    return hashlib.sha256((array & 0xFE).tobytes()).digest()


def _compose(
    tiles: list[np.ndarray],
    assign: list[int],
    rots: list[int],
    grid_w: int,
    grid_h: int,
    block_size: int,
) -> np.ndarray:
    height = grid_h * block_size
    width = grid_w * block_size
    output = np.zeros((height, width, 3), dtype=np.uint8)
    for pos in range(len(assign)):
        row = pos // grid_w
        col = pos % grid_w
        data = _rotate_data(tiles[assign[pos]], rots[pos])
        y0 = row * block_size
        x0 = col * block_size
        output[y0 : y0 + block_size, x0 : x0 + block_size] = data
    return output


def _edge_cost_rotated(a: np.ndarray, b: np.ndarray, direction: str, band: int = 3) -> float:
    band = max(1, min(band, a.shape[0], a.shape[1], b.shape[0], b.shape[1]))
    if direction == "horizontal":
        return float(np.mean(np.abs(a[:, -band:, :].astype(np.float32) - b[:, :band, :].astype(np.float32))))
    return float(np.mean(np.abs(a[-band:, :, :].astype(np.float32) - b[:band, :, :].astype(np.float32))))


def _pair_cost(a: np.ndarray, b: np.ndarray, params: KDCJCParams) -> float:
    ma = a.mean(axis=(0, 1))
    mb = b.mean(axis=(0, 1))
    sa = a.std(axis=(0, 1))
    sb = b.std(axis=(0, 1))
    return params.holistic_weight * (
        float(np.linalg.norm(ma - mb)) + 0.5 * float(np.linalg.norm(sa - sb))
    )


def _placement_cost(
    pos: int,
    tile_id: int,
    rot: int,
    assign: list[int],
    rots: list[int],
    tiles: list[np.ndarray],
    grid_w: int,
    params: KDCJCParams,
) -> float:
    row = pos // grid_w
    col = pos % grid_w
    data = _rotate_data(tiles[tile_id], rot)
    cost = 0.0
    if col > 0:
        left = _rotate_data(tiles[assign[pos - 1]], rots[pos - 1])
        cost += params.edge_weight * _edge_cost_rotated(left, data, "horizontal")
        cost += _pair_cost(left, data, params)
    if row > 0:
        top = _rotate_data(tiles[assign[pos - grid_w]], rots[pos - grid_w])
        cost += params.edge_weight * _edge_cost_rotated(top, data, "vertical")
        cost += _pair_cost(top, data, params)
    return cost


def _seam_total(
    tiles: list[np.ndarray],
    assign: list[int],
    rots: list[int],
    grid_w: int,
    grid_h: int,
    params: KDCJCParams,
) -> float:
    blocks: list[Block] = []
    for pos in range(len(assign)):
        data = _rotate_data(tiles[assign[pos]], rots[pos])
        blocks.append(
            Block(
                block_id=pos,
                data=data,
                top=data[0, :, :],
                bottom=data[-1, :, :],
                left=data[:, 0, :],
                right=data[:, -1, :],
                mean=data.mean(axis=(0, 1)),
                std=data.std(axis=(0, 1)),
            )
        )
    cost = 0.0
    for row in range(grid_h):
        for col in range(grid_w):
            idx = row * grid_w + col
            block = blocks[idx]
            if col + 1 < grid_w:
                right = blocks[idx + 1]
                cost += params.edge_weight * _edge_band_cost(block, right, "horizontal")
                cost += _holistic_pair_cost(block, right, params)
            if row + 1 < grid_h:
                bottom = blocks[idx + grid_w]
                cost += params.edge_weight * _edge_band_cost(block, bottom, "vertical")
                cost += _holistic_pair_cost(block, bottom, params)
    return cost


def _greedy_assign(
    tiles: list[np.ndarray],
    grid_w: int,
    grid_h: int,
    params: KDCJCParams,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    count = len(tiles)
    assign = [-1] * count
    rots = [0] * count
    remaining = set(range(count))

    for pos in range(count):
        candidates = list(remaining)
        if len(candidates) > _CANDIDATE_POOL:
            rng.shuffle(candidates)
            candidates = candidates[:_CANDIDATE_POOL]

        scored: list[tuple[float, int, int]] = []
        for tid in candidates:
            for rot in range(4):
                if pos == 0:
                    data = _rotate_data(tiles[tid], rot)
                    score = float(data.std()) + float(np.linalg.norm(data.mean(axis=(0, 1))))
                    scored.append((-score, tid, rot))
                else:
                    cost = _placement_cost(pos, tid, rot, assign, rots, tiles, grid_w, params)
                    scored.append((cost, tid, rot))

        scored.sort(key=lambda item: item[0])
        top = scored[: min(8, len(scored))]
        pick = top[rng.randrange(len(top))]
        assign[pos] = pick[1]
        rots[pos] = pick[2]
        remaining.remove(pick[1])

    return assign, rots


def _optimize_rotations(
    tiles: list[np.ndarray],
    assign: list[int],
    rots: list[int],
    grid_w: int,
    grid_h: int,
    params: KDCJCParams,
) -> list[int]:
    count = len(assign)
    best = rots[:]
    best_seam = _seam_total(tiles, assign, best, grid_w, grid_h, params)
    improved = True
    while improved:
        improved = False
        for pos in range(count):
            current = best[pos]
            for rot in range(4):
                if rot == current:
                    continue
                trial = best[:]
                trial[pos] = rot
                seam = _seam_total(tiles, assign, trial, grid_w, grid_h, params)
                if seam + 1e-6 < best_seam:
                    best = trial
                    best_seam = seam
                    improved = True
                    break
    return best


def _refine_annealing(
    tiles: list[np.ndarray],
    assign: list[int],
    rots: list[int],
    grid_w: int,
    grid_h: int,
    block_size: int,
    target_hash: bytes,
    params: KDCJCParams,
    rng: random.Random,
    max_iters: int,
) -> tuple[list[int], list[int], bool]:
    count = len(assign)
    current_a = assign[:]
    current_r = _optimize_rotations(tiles, current_a, rots[:], grid_w, grid_h, params)
    current_seam = _seam_total(tiles, current_a, current_r, grid_w, grid_h, params)

    best_a = current_a[:]
    best_r = current_r[:]
    best_seam = current_seam

    image = _compose(tiles, best_a, best_r, grid_w, grid_h, block_size)
    if _image_hash(image) == target_hash:
        return best_a, best_r, True

    for step in range(max_iters):
        trial_a = current_a[:]
        trial_r = current_r[:]
        move = rng.randrange(4)
        if move == 0:
            i = rng.randrange(count)
            j = rng.randrange(count)
            trial_a[i], trial_a[j] = trial_a[j], trial_a[i]
        elif move == 1:
            i = rng.randrange(count)
            trial_r[i] = (trial_r[i] + 1) % 4
        elif move == 2:
            i = rng.randrange(count)
            j = rng.randrange(count)
            trial_a[i], trial_a[j] = trial_a[j], trial_a[i]
            trial_r[i] = (trial_r[i] + 1) % 4
        else:
            i = rng.randrange(count)
            trial_r[i] = rng.randrange(4)

        trial_seam = _seam_total(tiles, trial_a, trial_r, grid_w, grid_h, params)
        temperature = max(0.01, 1.0 - step / max_iters)
        delta = trial_seam - current_seam
        if delta <= 0 or rng.random() < np.exp(-delta / (temperature * 80.0 + 1e-6)):
            current_a = trial_a
            current_r = trial_r
            current_seam = trial_seam
            if current_seam < best_seam:
                best_a = current_a[:]
                best_r = current_r[:]
                best_seam = current_seam
                image = _compose(tiles, best_a, best_r, grid_w, grid_h, block_size)
                if _image_hash(image) == target_hash:
                    return best_a, best_r, True

        if step % 128 == 0:
            image = _compose(tiles, best_a, best_r, grid_w, grid_h, block_size)
            if _image_hash(image) == target_hash:
                return best_a, best_r, True

    image = _compose(tiles, best_a, best_r, grid_w, grid_h, block_size)
    return best_a, best_r, _image_hash(image) == target_hash


def solve_jigsaw(
    scrambled: np.ndarray,
    block_size: int,
    grid_width: int,
    grid_height: int,
    original_width: int,
    original_height: int,
    target_hash: bytes,
    params: KDCJCParams | None = None,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    params = params or KDCJCParams()
    count = grid_width * grid_height
    if count > SOLVER_MAX_BLOCKS:
        raise ValueError(
            f"求解模式块数过多（{count} 块，上限 {SOLVER_MAX_BLOCKS}）。"
            f"请增大块大小后重新加密。"
        )

    tiles_obj = _split_blocks(scrambled, block_size)
    if len(tiles_obj) != count:
        raise ValueError("块数量与网格不匹配")

    tiles = [tile.data.copy() for tile in tiles_obj]
    max_iters = max(15_000, count * 80)
    progress_step = 100 // _RESTARTS

    for restart in range(_RESTARTS):
        if progress_callback:
            progress_callback(min(95, restart * progress_step), 100)

        rng = random.Random(0x4B44434A + restart * 7919)
        assign, rots = _greedy_assign(tiles, grid_width, grid_height, params, rng)
        rots = _optimize_rotations(tiles, assign, rots, grid_width, grid_height, params)

        image = _compose(tiles, assign, rots, grid_width, grid_height, block_size)
        if _image_hash(image) == target_hash:
            if progress_callback:
                progress_callback(100, 100)
            return image[:original_height, :original_width]

        assign, rots, ok = _refine_annealing(
            tiles,
            assign,
            rots,
            grid_width,
            grid_height,
            block_size,
            target_hash,
            params,
            rng,
            max_iters,
        )
        if ok:
            image = _compose(tiles, assign, rots, grid_width, grid_height, block_size)
            if progress_callback:
                progress_callback(100, 100)
            return image[:original_height, :original_width]

    raise ValueError(
        "拼图求解失败：无法在合理时间内找到匹配排列。"
        "请增大块大小（减少块数）后重新加密。"
    )
