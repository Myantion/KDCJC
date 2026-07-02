from __future__ import annotations

import random
from collections import deque

import numpy as np

from kdcjc.core import Block


def block_features(blocks: list[Block]) -> np.ndarray:
    rows: list[list[float]] = []
    for block in blocks:
        rows.append(
            [
                float(block.mean[0]),
                float(block.mean[1]),
                float(block.mean[2]),
                float(block.std[0]),
                float(block.std[1]),
                float(block.std[2]),
                float(block.edge_richness),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return (matrix - mean) / std


def _pca_embed(matrix: np.ndarray, components: int) -> np.ndarray:
    if matrix.shape[0] < 2:
        return matrix
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(components, vt.shape[0])
    return centered @ vt[:rank].T


def _umap_embed(matrix: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError("未安装 umap-learn，请执行 pip install umap-learn，或改用 PCA") from exc

    n_neighbors = max(2, min(15, matrix.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=min(2, matrix.shape[0] - 1),
        n_neighbors=n_neighbors,
        random_state=seed & 0xFFFFFFFF,
    )
    return reducer.fit_transform(matrix)


def _kmeans_labels(matrix: np.ndarray, k: int, rng: random.Random, max_iter: int = 40) -> list[int]:
    count = matrix.shape[0]
    k = max(1, min(k, count))
    if k == 1:
        return [0] * count

    indices = rng.sample(range(count), k)
    centers = matrix[indices].copy()
    labels = np.zeros(count, dtype=np.int32)

    for _ in range(max_iter):
        dist = np.sum((matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = dist.argmin(axis=1).astype(np.int32)
        new_centers = centers.copy()
        for cluster in range(k):
            members = matrix[labels == cluster]
            if members.size:
                new_centers[cluster] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels.tolist()


def assign_pile_labels(
    blocks: list[Block],
    method: str,
    n_piles: int,
    seed: int,
) -> list[int]:
    features = _standardize(block_features(blocks))
    rng = random.Random(seed)

    if method == "none" or len(blocks) <= 1:
        return [0] * len(blocks)

    if method == "umap":
        try:
            embedded = _umap_embed(features, seed)
        except ImportError:
            embedded = _pca_embed(features, min(3, features.shape[1]))
    else:
        embedded = _pca_embed(features, min(3, features.shape[1]))

    piles = max(2, min(n_piles, len(blocks)))
    return _kmeans_labels(embedded, piles, rng)


def _spread_seeds(grid_h: int, grid_w: int, n_piles: int, rng: random.Random) -> list[tuple[int, int]]:
    candidates = [(row, col) for row in range(grid_h) for col in range(grid_w)]
    first = rng.choice(candidates)
    seeds = [first]

    while len(seeds) < n_piles:
        best_cell = candidates[0]
        best_dist = -1.0
        for cell in candidates:
            dist = min((cell[0] - s[0]) ** 2 + (cell[1] - s[1]) ** 2 for s in seeds)
            if dist > best_dist:
                best_dist = dist
                best_cell = cell
        seeds.append(best_cell)

    return seeds


def _neighbors(row: int, col: int, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid_h and 0 <= nc < grid_w:
            result.append((nr, nc))
    return result


def _grow_heap_regions(
    grid_h: int,
    grid_w: int,
    pile_sizes: list[int],
    seeds: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    n_piles = len(pile_sizes)
    targets = pile_sizes[:]
    regions: list[list[tuple[int, int]]] = [[] for _ in range(n_piles)]
    assigned: set[tuple[int, int]] = set()
    queues = [deque([seeds[p]]) for p in range(n_piles)]

    while sum(len(region) for region in regions) < grid_h * grid_w:
        progress = False
        order = list(range(n_piles))
        for pile_id in order:
            if len(regions[pile_id]) >= targets[pile_id]:
                continue
            while queues[pile_id] and len(regions[pile_id]) < targets[pile_id]:
                row, col = queues[pile_id].popleft()
                if (row, col) in assigned:
                    continue
                assigned.add((row, col))
                regions[pile_id].append((row, col))
                progress = True
                for nr, nc in _neighbors(row, col, grid_h, grid_w):
                    if (nr, nc) not in assigned:
                        queues[pile_id].append((nr, nc))
                break
            if len(regions[pile_id]) >= targets[pile_id]:
                continue
            if not progress and len(regions[pile_id]) < targets[pile_id]:
                for row in range(grid_h):
                    for col in range(grid_w):
                        cell = (row, col)
                        if cell not in assigned:
                            assigned.add(cell)
                            regions[pile_id].append(cell)
                            progress = True
                            break
                    if progress:
                        break
        if not progress:
            break

    leftover = [
        (row, col)
        for row in range(grid_h)
        for col in range(grid_w)
        if (row, col) not in assigned
    ]
    for cell in leftover:
        nearest = min(
            range(n_piles),
            key=lambda pile_id: (cell[0] - seeds[pile_id][0]) ** 2 + (cell[1] - seeds[pile_id][1]) ** 2,
        )
        regions[nearest].append(cell)

    return regions


def _balanced_voronoi_regions(
    grid_h: int,
    grid_w: int,
    pile_sizes: list[int],
    seeds: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """泰森多边形分域，并按目标块数平衡各堆区域大小。"""
    n_piles = len(pile_sizes)
    cells = [(row, col) for row in range(grid_h) for col in range(grid_w)]

    def dist(pile_id: int, row: int, col: int) -> tuple[int, int, int]:
        sr, sc = seeds[pile_id]
        return ((row - sr) ** 2 + (col - sc) ** 2, row, col)

    def nearest_piles(row: int, col: int) -> list[int]:
        return sorted(range(n_piles), key=lambda pile_id: dist(pile_id, row, col))

    regions: list[list[tuple[int, int]]] = [[] for _ in range(n_piles)]
    for row, col in cells:
        regions[nearest_piles(row, col)[0]].append((row, col))

    changed = True
    while changed:
        changed = False
        for pile_id in range(n_piles):
            target = pile_sizes[pile_id]
            while len(regions[pile_id]) > target:
                row, col = max(regions[pile_id], key=lambda rc: dist(pile_id, rc[0], rc[1]))
                regions[pile_id].remove((row, col))
                moved = False
                for alt in nearest_piles(row, col):
                    if alt != pile_id and len(regions[alt]) < pile_sizes[alt]:
                        regions[alt].append((row, col))
                        moved = True
                        changed = True
                        break
                if not moved:
                    regions[pile_id].append((row, col))
                    break

        for pile_id in range(n_piles):
            target = pile_sizes[pile_id]
            while len(regions[pile_id]) < target:
                best: tuple[tuple[int, int, int], int, tuple[int, int]] | None = None
                for donor in range(n_piles):
                    if len(regions[donor]) <= pile_sizes[donor]:
                        continue
                    for row, col in regions[donor]:
                        score = dist(pile_id, row, col)
                        if best is None or score < best[0]:
                            best = (score, donor, (row, col))
                if best is None:
                    break
                _, donor, (row, col) = best
                regions[donor].remove((row, col))
                regions[pile_id].append((row, col))
                changed = True

    return regions


def voronoi_grid(
    block_count: int,
    grid_w: int,
    grid_h: int,
    pile_labels: list[int],
    rng: random.Random,
) -> list[list[int]]:
    """把同色块放进泰森多边形区域，边界呈不规则多边形。"""
    n_piles = max(pile_labels) + 1 if pile_labels else 1
    by_pile: list[list[int]] = [[] for _ in range(n_piles)]
    for block_id, pile_id in enumerate(pile_labels):
        by_pile[pile_id].append(block_id)

    pile_sizes = [len(by_pile[pile_id]) for pile_id in range(n_piles)]
    if sum(pile_sizes) != block_count:
        raise ValueError("分堆块数与网格不匹配")

    seeds = _spread_seeds(grid_h, grid_w, n_piles, rng)
    regions = _balanced_voronoi_regions(grid_h, grid_w, pile_sizes, seeds)

    grid: list[list[int | None]] = [[None for _ in range(grid_w)] for _ in range(grid_h)]
    for pile_id in range(n_piles):
        cells = regions[pile_id][:]
        rng.shuffle(cells)
        blocks = by_pile[pile_id][:]
        rng.shuffle(blocks)
        for (row, col), block_id in zip(cells, blocks):
            grid[row][col] = block_id

    for row in range(grid_h):
        for col in range(grid_w):
            if grid[row][col] is None:
                raise ValueError("泰森多边形布局未覆盖全部网格")

    return [[int(grid[row][col]) for col in range(grid_w)] for row in range(grid_h)]


def heap_grid(
    block_count: int,
    grid_w: int,
    grid_h: int,
    pile_labels: list[int],
    rng: random.Random,
) -> list[list[int]]:
    """把同色块放进二维连通区域（堆），而不是按行排成条带。"""
    n_piles = max(pile_labels) + 1 if pile_labels else 1
    by_pile: list[list[int]] = [[] for _ in range(n_piles)]
    for block_id, pile_id in enumerate(pile_labels):
        by_pile[pile_id].append(block_id)

    pile_sizes = [len(by_pile[pile_id]) for pile_id in range(n_piles)]
    if sum(pile_sizes) != block_count:
        raise ValueError("分堆块数与网格不匹配")

    seeds = _spread_seeds(grid_h, grid_w, n_piles, rng)
    regions = _grow_heap_regions(grid_h, grid_w, pile_sizes, seeds)

    grid: list[list[int | None]] = [[None for _ in range(grid_w)] for _ in range(grid_h)]
    for pile_id in range(n_piles):
        cells = regions[pile_id][:]
        rng.shuffle(cells)
        blocks = by_pile[pile_id][:]
        rng.shuffle(blocks)
        for (row, col), block_id in zip(cells, blocks):
            grid[row][col] = block_id

    for row in range(grid_h):
        for col in range(grid_w):
            if grid[row][col] is None:
                raise ValueError("分堆布局未覆盖全部网格")

    return [[int(grid[row][col]) for col in range(grid_w)] for row in range(grid_h)]


def strip_grid(
    block_count: int,
    grid_w: int,
    grid_h: int,
    pile_labels: list[int],
    rng: random.Random,
) -> list[list[int]]:
    """旧版：按行填充，视觉上呈色条。"""
    n_piles = max(pile_labels) + 1 if pile_labels else 1
    by_pile: list[list[int]] = [[] for _ in range(n_piles)]
    for block_id, pile_id in enumerate(pile_labels):
        by_pile[pile_id].append(block_id)

    ordered_blocks: list[int] = []
    pile_order = list(range(n_piles))
    rng.shuffle(pile_order)
    for pile_id in pile_order:
        chunk = by_pile[pile_id][:]
        rng.shuffle(chunk)
        ordered_blocks.extend(chunk)

    grid: list[list[int]] = []
    index = 0
    for _row in range(grid_h):
        row: list[int] = []
        for _col in range(grid_w):
            row.append(ordered_blocks[index])
            index += 1
        grid.append(row)
    return grid


def piled_grid(
    block_count: int,
    grid_w: int,
    grid_h: int,
    pile_labels: list[int],
    rng: random.Random,
    layout: str = "heap",
) -> list[list[int]]:
    if layout == "strip":
        return strip_grid(block_count, grid_w, grid_h, pile_labels, rng)
    if layout == "voronoi":
        return voronoi_grid(block_count, grid_w, grid_h, pile_labels, rng)
    return heap_grid(block_count, grid_w, grid_h, pile_labels, rng)
