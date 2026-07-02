from __future__ import annotations

import random
import secrets
import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

ProgressCallback = Callable[[int, int], None]

MAX_BLOCKS = 12000
BUILD_PROGRESS_RATIO = 0.72


@dataclass
class Block:
    block_id: int
    data: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    edge_richness: float = 0.0


@dataclass
class KDCJCParams:
    block_size: int = 64
    edge_weight: float = 2.0
    holistic_weight: float = 0.4
    global_weight: float = 0.12
    top_k: int = 3
    candidate_pool: int = 128
    seed_top_k: int = 8
    swap_iterations: int = 5000
    hard_mode: bool = True
    cluster_method: str = "none"
    cluster_piles: int = 12
    pile_layout: str = "voronoi"
    solver_mode: bool = True  # 深改：不存排列表，由 seed+分堆标签推导
    soften_strip: int = 0  # 块接缝像素柔化宽度，0=关闭（实验）
    pile_toning: bool = True  # 分堆内色调协调
    block_smooth: bool = True  # 大图自动使用 Q7 压缩 + 双平面 LSB


@dataclass
class KDCJCResult:
    image: Image.Image
    block_size: int
    permutation: list[int]
    grid_width: int
    grid_height: int
    padded_width: int
    padded_height: int
    seed: int
    rotations: list[int] = field(default_factory=list)
    content_hash: bytes = field(default_factory=bytes)
    pile_labels: list[int] = field(default_factory=list)
    cluster_method: str = "pca"
    pile_layout: str = "voronoi"
    soften_strip: int = 0
    pile_toning: bool = True
    block_smooth: bool = True
    toning_deltas: np.ndarray | None = field(default=None, repr=False)
    smooth_residuals: np.ndarray | None = field(default=None, repr=False)


def estimate_block_count(width: int, height: int, block_size: int) -> tuple[int, int, int]:
    if block_size <= 0:
        raise ValueError("块大小必须大于 0")
    pad_w = (block_size - width % block_size) % block_size
    pad_h = (block_size - height % block_size) % block_size
    grid_w = (width + pad_w) // block_size
    grid_h = (height + pad_h) // block_size
    return grid_w, grid_h, grid_w * grid_h


def suggest_block_size(width: int, height: int, target_blocks: int = 2500) -> int:
    block_size = 32
    while block_size <= 256:
        _, _, count = estimate_block_count(width, height, block_size)
        if count <= target_blocks:
            return block_size
        block_size += 8
    return 256


def _seed_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _pad_image(image: Image.Image, block_size: int) -> tuple[Image.Image, int, int]:
    width, height = image.size
    pad_w = (block_size - width % block_size) % block_size
    pad_h = (block_size - height % block_size) % block_size
    if pad_w == 0 and pad_h == 0:
        return image, width, height
    padded = Image.new("RGB", (width + pad_w, height + pad_h))
    padded.paste(image, (0, 0))
    if pad_w:
        edge = image.crop((width - 1, 0, width, height))
        for x in range(width, width + pad_w):
            padded.paste(edge, (x, 0))
    if pad_h:
        edge = padded.crop((0, height - 1, width + pad_w, height))
        for y in range(height, height + pad_h):
            padded.paste(edge, (0, y))
    return padded, width, height


def _edge_richness(data: np.ndarray) -> float:
    top = data[0, :, :]
    bottom = data[-1, :, :]
    left = data[:, 0, :]
    right = data[:, -1, :]
    score = 0.0
    for edge in (top, bottom, left, right):
        edge_f = edge.astype(np.float32)
        score += float(edge_f.std())
        if edge_f.shape[0] > 1:
            score += float(np.mean(np.abs(np.diff(edge_f, axis=0))))
        if edge_f.shape[1] > 1:
            score += float(np.mean(np.abs(np.diff(edge_f, axis=1))))
    return score


def _split_blocks(array: np.ndarray, block_size: int) -> list[Block]:
    height, width, _ = array.shape
    grid_h = height // block_size
    grid_w = width // block_size
    usable_h = grid_h * block_size
    usable_w = grid_w * block_size
    tiles = (
        array[:usable_h, :usable_w]
        .reshape(grid_h, block_size, grid_w, block_size, 3)
        .transpose(0, 2, 1, 3, 4)
    )

    blocks: list[Block] = []
    block_id = 0
    for row in range(grid_h):
        for col in range(grid_w):
            data = tiles[row, col].copy()
            blocks.append(
                Block(
                    block_id=block_id,
                    data=data,
                    top=data[0, :, :],
                    bottom=data[-1, :, :],
                    left=data[:, 0, :],
                    right=data[:, -1, :],
                    mean=data.mean(axis=(0, 1)),
                    std=data.std(axis=(0, 1)),
                    edge_richness=_edge_richness(data),
                )
            )
            block_id += 1
    return blocks


def _edge_cost(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    a = edge_a.astype(np.float32)
    b = edge_b.astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def _edge_band_cost(block_a: Block, block_b: Block, direction: str, band: int = 3) -> float:
    band = max(1, min(band, block_a.data.shape[0], block_a.data.shape[1]))
    if direction == "horizontal":
        left_band = block_a.data[:, :band, :]
        right_band = block_b.data[:, -band:, :]
    else:
        left_band = block_a.data[:band, :, :]
        right_band = block_b.data[-band:, :, :]
    return _edge_cost(left_band, right_band)


def _holistic_pair_cost(block_a: Block, block_b: Block, params: KDCJCParams) -> float:
    return params.holistic_weight * (
        float(np.linalg.norm(block_a.mean - block_b.mean))
        + 0.5 * float(np.linalg.norm(block_a.std - block_b.std))
    )


def _seam_cost_horizontal(blocks: list[Block], left_id: int, right_id: int, params: KDCJCParams) -> float:
    left = blocks[left_id]
    right = blocks[right_id]
    band = min(3, left.data.shape[1], right.data.shape[1])
    edge = _edge_cost(left.data[:, -band:, :], right.data[:, :band, :])
    return params.edge_weight * edge + _holistic_pair_cost(left, right, params)


def _seam_cost_vertical(blocks: list[Block], top_id: int, bottom_id: int, params: KDCJCParams) -> float:
    top = blocks[top_id]
    bottom = blocks[bottom_id]
    band = min(3, top.data.shape[0], bottom.data.shape[0])
    edge = _edge_cost(top.data[-band:, :, :], bottom.data[:band, :, :])
    return params.edge_weight * edge + _holistic_pair_cost(top, bottom, params)


def _quick_costs(
    remaining_ids: np.ndarray,
    means: np.ndarray,
    global_target: np.ndarray,
    left_mean: np.ndarray | None,
    top_mean: np.ndarray | None,
    params: KDCJCParams,
) -> np.ndarray:
    rem_means = means[remaining_ids]
    costs = params.global_weight * np.linalg.norm(rem_means - global_target, axis=1)
    if left_mean is not None:
        costs += params.holistic_weight * np.linalg.norm(rem_means - left_mean, axis=1)
    if top_mean is not None:
        costs += params.holistic_weight * np.linalg.norm(rem_means - top_mean, axis=1)
    return costs


def _full_cost(
    candidate: Block,
    blocks: list[Block],
    col: int,
    row: int,
    grid: list[list[int | None]],
    global_target: np.ndarray,
    params: KDCJCParams,
) -> float:
    cost = params.global_weight * float(np.linalg.norm(candidate.mean - global_target))
    if col > 0:
        left_id = grid[row][col - 1]
        assert left_id is not None
        left = blocks[left_id]
        cost += params.edge_weight * _edge_band_cost(candidate, left, "horizontal")
        cost += _holistic_pair_cost(candidate, left, params)
    if row > 0:
        top_id = grid[row - 1][col]
        assert top_id is not None
        top = blocks[top_id]
        cost += params.edge_weight * _edge_band_cost(candidate, top, "vertical")
        cost += _holistic_pair_cost(candidate, top, params)
    return cost


def _refresh_block(block: Block) -> None:
    data = block.data
    block.top = data[0, :, :]
    block.bottom = data[-1, :, :]
    block.left = data[:, 0, :]
    block.right = data[:, -1, :]
    block.mean = data.mean(axis=(0, 1))
    block.std = data.std(axis=(0, 1))
    block.edge_richness = _edge_richness(data)


def _rotate_block(block: Block, turns: int) -> None:
    block.data = np.rot90(block.data, turns % 4)
    _refresh_block(block)


def _random_grid(block_count: int, grid_w: int, grid_h: int, rng: random.Random) -> list[list[int]]:
    ids = list(range(block_count))
    rng.shuffle(ids)
    grid: list[list[int]] = []
    index = 0
    for _row in range(grid_h):
        row: list[int] = []
        for _col in range(grid_w):
            row.append(ids[index])
            index += 1
        grid.append(row)
    return grid


def _choose_seed_block(blocks: list[Block], rng: random.Random, params: KDCJCParams) -> int:
    ranked = sorted(range(len(blocks)), key=lambda idx: blocks[idx].edge_richness, reverse=True)
    pool = ranked[: min(params.seed_top_k, len(ranked))]
    return pool[rng.randrange(len(pool))]


def _compose_from_grid(
    blocks: list[Block],
    grid: list[list[int | None]],
    block_size: int,
) -> np.ndarray:
    grid_h = len(grid)
    grid_w = len(grid[0])
    height = grid_h * block_size
    width = grid_w * block_size
    output = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(grid_h):
        for col in range(grid_w):
            block_id = grid[row][col]
            if block_id is None:
                continue
            y0 = row * block_size
            x0 = col * block_size
            output[y0 : y0 + block_size, x0 : x0 + block_size] = blocks[block_id].data
    return output


def _incident_seams(row: int, col: int, grid_h: int, grid_w: int) -> list[tuple[str, int, int]]:
    seams: list[tuple[str, int, int]] = []
    if col > 0:
        seams.append(("h", row, col - 1))
    if col < grid_w - 1:
        seams.append(("h", row, col))
    if row > 0:
        seams.append(("v", row - 1, col))
    if row < grid_h - 1:
        seams.append(("v", row, col))
    return seams


def _seam_value(
    grid: list[list[int]],
    blocks: list[Block],
    seam: tuple[str, int, int],
    params: KDCJCParams,
) -> float:
    kind, row, col = seam
    if kind == "h":
        return _seam_cost_horizontal(blocks, grid[row][col], grid[row][col + 1], params)
    return _seam_cost_vertical(blocks, grid[row][col], grid[row + 1][col], params)


def _patch_seam_cost(
    grid: list[list[int]],
    blocks: list[Block],
    positions: tuple[tuple[int, int], tuple[int, int]],
    params: KDCJCParams,
) -> float:
    grid_h = len(grid)
    grid_w = len(grid[0])
    pos_a, pos_b = positions
    seam_keys = set(_incident_seams(pos_a[0], pos_a[1], grid_h, grid_w))
    seam_keys.update(_incident_seams(pos_b[0], pos_b[1], grid_h, grid_w))
    return sum(_seam_value(grid, blocks, seam, params) for seam in seam_keys)


def _optimize_by_swaps(
    grid: list[list[int]],
    blocks: list[Block],
    seed: int,
    params: KDCJCParams,
    progress_callback: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int = 100,
) -> list[list[int]]:
    if params.swap_iterations <= 0:
        return grid

    rng = _seed_rng(seed ^ 0xA5A5A5A5)
    grid_h = len(grid)
    grid_w = len(grid[0])
    working = [row[:] for row in grid]

    for step in range(params.swap_iterations):
        if grid_w > 1 and (grid_h == 1 or rng.random() < 0.5):
            row = rng.randrange(grid_h)
            col = rng.randrange(grid_w - 1)
            pos_a = (row, col)
            pos_b = (row, col + 1)
        elif grid_h > 1:
            row = rng.randrange(grid_h - 1)
            col = rng.randrange(grid_w)
            pos_a = (row, col)
            pos_b = (row + 1, col)
        else:
            break

        before = _patch_seam_cost(working, blocks, (pos_a, pos_b), params)
        ar, ac = pos_a
        br, bc = pos_b
        working[ar][ac], working[br][bc] = working[br][bc], working[ar][ac]
        after = _patch_seam_cost(working, blocks, (pos_a, pos_b), params)
        if after > before:
            working[ar][ac], working[br][bc] = working[br][bc], working[ar][ac]

        if progress_callback and (step % 64 == 0 or step + 1 == params.swap_iterations):
            done = progress_offset + int((step + 1) / params.swap_iterations * progress_total)
            progress_callback(done, progress_offset + progress_total)

    return working


def _build_permutation(
    blocks: list[Block],
    grid_w: int,
    grid_h: int,
    seed: int,
    params: KDCJCParams,
    progress_callback: ProgressCallback | None = None,
    progress_total: int = 100,
) -> list[list[int]]:
    rng = _seed_rng(seed)
    remaining = set(range(len(blocks)))
    grid: list[list[int | None]] = [[None for _ in range(grid_w)] for _ in range(grid_h)]
    means = np.stack([block.mean for block in blocks], axis=0)
    global_target = means.mean(axis=0)
    total_cells = grid_w * grid_h
    processed = 0

    start_idx = _choose_seed_block(blocks, rng, params)
    grid[0][0] = start_idx
    remaining.remove(start_idx)

    for row in range(grid_h):
        for col in range(grid_w):
            if grid[row][col] is not None:
                continue

            left_mean = means[grid[row][col - 1]] if col > 0 else None
            top_mean = means[grid[row - 1][col]] if row > 0 else None
            remaining_list = np.fromiter(remaining, dtype=np.int32)
            quick = _quick_costs(
                remaining_list, means, global_target, left_mean, top_mean, params
            )

            pool_size = min(params.candidate_pool, len(remaining_list))
            if pool_size < len(remaining_list):
                shortlist = remaining_list[np.argsort(quick)[:pool_size]]
            else:
                shortlist = remaining_list

            scored: list[tuple[float, int]] = []
            for block_id in shortlist:
                cost = _full_cost(
                    blocks[int(block_id)],
                    blocks,
                    col,
                    row,
                    grid,
                    global_target,
                    params,
                )
                scored.append((cost, int(block_id)))

            scored.sort(key=lambda item: item[0])
            pick_index = rng.randrange(min(params.top_k, len(scored)))
            chosen = scored[pick_index][1]
            grid[row][col] = chosen
            remaining.remove(chosen)

            processed += 1
            if progress_callback and (processed % 8 == 0 or processed == total_cells):
                done = int(processed / total_cells * progress_total)
                progress_callback(done, 100)

    return [[cell for cell in row if cell is not None] for row in grid]


def _grid_to_permutation(grid: list[list[int]], grid_w: int, grid_h: int) -> list[int]:
    permutation: list[int] = []
    for row in range(grid_h):
        for col in range(grid_w):
            permutation.append(grid[row][col])
    return permutation


def _make_progress_bridge(
    progress_callback: ProgressCallback | None,
) -> tuple[ProgressCallback | None, int, int]:
    if progress_callback is None:
        return None, 72, 28

    build_total = int(100 * BUILD_PROGRESS_RATIO)
    swap_total = 100 - build_total

    def bridge(done: int, total: int) -> None:
        progress_callback(done, total)

    return bridge, build_total, swap_total


def encrypt_image(
    image: Image.Image,
    params: KDCJCParams | None = None,
    seed: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> KDCJCResult:
    params = params or KDCJCParams()
    if params.block_size < 4:
        raise ValueError("块大小至少为 4")

    if seed is None:
        seed = secrets.randbits(63)

    rgb = image.convert("RGB")
    padded, orig_w, orig_h = _pad_image(rgb, params.block_size)
    array = np.asarray(padded, dtype=np.uint8)
    content_hash = hashlib.sha256((array & 0xFE).tobytes()).digest()
    blocks = _split_blocks(array, params.block_size)
    grid_w = padded.size[0] // params.block_size
    grid_h = padded.size[1] // params.block_size

    if len(blocks) < 2:
        raise ValueError("图像尺寸过小，无法分块加密")

    if len(blocks) > MAX_BLOCKS:
        suggested = suggest_block_size(orig_w, orig_h)
        raise ValueError(
            f"块数过多（{len(blocks)} 块），计算会非常慢。"
            f"建议将块大小调到至少 {suggested}（约 {estimate_block_count(orig_w, orig_h, suggested)[2]} 块）"
        )

    rng = _seed_rng(seed)
    rotations = [0] * len(blocks)
    if params.hard_mode:
        rotations = [rng.randrange(4) for _ in range(len(blocks))]
        for block, turns in zip(blocks, rotations):
            if turns:
                _rotate_block(block, turns)

    bridge, build_total, swap_total = _make_progress_bridge(progress_callback)
    pile_labels: list[int] = []

    if params.hard_mode:
        if bridge:
            bridge(100, 100)
        from kdcjc.cluster import assign_pile_labels, piled_grid

        pile_labels = assign_pile_labels(
            blocks,
            params.cluster_method,
            params.cluster_piles,
            seed,
        )
        if params.cluster_method == "none":
            grid = _random_grid(len(blocks), grid_w, grid_h, rng)
        else:
            grid = piled_grid(
                len(blocks),
                grid_w,
                grid_h,
                pile_labels,
                rng,
                layout=params.pile_layout,
            )
    else:
        def build_progress(done: int, _total: int) -> None:
            if bridge:
                bridge(done, 100)

        grid = _build_permutation(
            blocks,
            grid_w,
            grid_h,
            seed,
            params,
            progress_callback=build_progress,
            progress_total=build_total,
        )

        if params.swap_iterations > 0:
            swap_offset = build_total

            def swap_progress(done: int, total: int) -> None:
                if bridge:
                    bridge(done, 100)

            grid = _optimize_by_swaps(
                grid,
                blocks,
                seed,
                params,
                progress_callback=swap_progress,
                progress_offset=swap_offset,
                progress_total=swap_total,
            )
        elif bridge:
            bridge(100, 100)

    permutation = _grid_to_permutation(grid, grid_w, grid_h)
    pile_toning = bool(params.pile_toning) and bool(pile_labels) and params.cluster_method != "none"
    toning_deltas = None
    if pile_toning:
        from kdcjc.toning import apply_pile_toning, compute_block_deltas, toning_alpha_from_seed

        toning_deltas = compute_block_deltas(blocks, pile_labels, toning_alpha_from_seed(seed))
        apply_pile_toning(blocks, toning_deltas)

    smooth_residuals = None
    if params.block_smooth:
        from kdcjc.smooth import apply_block_smooth

        smooth_residuals = apply_block_smooth(blocks)

    scrambled = _compose_from_grid(blocks, grid, params.block_size)
    result_image = Image.fromarray(scrambled, mode="RGB")
    soften_strip = max(0, min(3, int(params.soften_strip)))

    return KDCJCResult(
        image=result_image,
        block_size=params.block_size,
        permutation=permutation,
        grid_width=grid_w,
        grid_height=grid_h,
        padded_width=padded.size[0],
        padded_height=padded.size[1],
        seed=seed,
        rotations=rotations,
        content_hash=content_hash,
        pile_labels=pile_labels,
        cluster_method=params.cluster_method,
        pile_layout=params.pile_layout,
        soften_strip=soften_strip,
        pile_toning=pile_toning,
        block_smooth=bool(params.block_smooth),
        toning_deltas=toning_deltas,
        smooth_residuals=smooth_residuals,
    )


def decrypt_image(
    scrambled: Image.Image,
    block_size: int,
    permutation: list[int],
    grid_width: int,
    grid_height: int,
    original_width: int,
    original_height: int,
    rotations: list[int] | None = None,
    pile_labels: list[int] | None = None,
    pile_toning: bool = False,
    toning_deltas: np.ndarray | None = None,
    block_smooth: bool = False,
    smooth_residuals: np.ndarray | None = None,
    seed: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> Image.Image:
    def report(done: int, total: int = 100) -> None:
        if progress_callback:
            progress_callback(done, total)

    report(5, 100)
    padded, _, _ = _pad_image(scrambled.convert("RGB"), block_size)
    array = np.asarray(padded, dtype=np.uint8)
    blocks = _split_blocks(array, block_size)

    if len(blocks) != len(permutation):
        raise ValueError("块数量与排列信息不匹配")

    report(15, 100)
    if bool(block_smooth) and smooth_residuals is not None and len(smooth_residuals) > 0:
        from kdcjc.smooth import remove_block_smooth_tiles

        remove_block_smooth_tiles(blocks, permutation, smooth_residuals)

    report(30, 100)
    if bool(pile_toning) and toning_deltas is not None and len(toning_deltas) > 0:
        from kdcjc.toning import remove_pile_toning_tiles

        remove_pile_toning_tiles(blocks, permutation, toning_deltas)

    report(40, 100)
    inverse = [0] * len(permutation)
    for new_index, original_index in enumerate(permutation):
        inverse[original_index] = new_index

    restored_grid: list[list[int | None]] = [
        [None for _ in range(grid_width)] for _ in range(grid_height)
    ]
    for original_index, new_index in enumerate(inverse):
        row = original_index // grid_width
        col = original_index % grid_width
        restored_grid[row][col] = new_index

    if rotations:
        working = [Block(
            block_id=block.block_id,
            data=block.data.copy(),
            top=block.top,
            bottom=block.bottom,
            left=block.left,
            right=block.right,
            mean=block.mean,
            std=block.std,
            edge_richness=block.edge_richness,
        ) for block in blocks]
        block_total = max(len(inverse), 1)
        for source_index, tile_index in enumerate(inverse):
            turns = (4 - rotations[source_index]) % 4
            if turns:
                _rotate_block(working[tile_index], turns)
            if source_index % 4 == 0 or source_index + 1 == block_total:
                report(40 + 45 * (source_index + 1) // block_total, 100)
        blocks = working
    else:
        report(85, 100)

    report(90, 100)
    restored = _compose_from_grid(blocks, restored_grid, block_size)
    report(100, 100)
    return Image.fromarray(restored, mode="RGB").crop((0, 0, original_width, original_height))
