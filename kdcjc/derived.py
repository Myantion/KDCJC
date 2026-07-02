from __future__ import annotations

import random

from kdcjc.core import KDCJCParams, _seed_rng


def permutation_from_derived(
    block_count: int,
    grid_w: int,
    grid_h: int,
    seed: int,
    pile_labels: list[int],
    params: KDCJCParams,
) -> list[int]:
    from kdcjc.cluster import piled_grid
    from kdcjc.core import _grid_to_permutation, _random_grid

    rng = _seed_rng(seed)
    for _ in range(block_count):
        rng.randrange(4)

    if params.cluster_method == "none" or not pile_labels:
        grid = _random_grid(block_count, grid_w, grid_h, rng)
    else:
        grid = piled_grid(
            block_count,
            grid_w,
            grid_h,
            pile_labels,
            rng,
            layout=params.pile_layout,
        )
    return _grid_to_permutation(grid, grid_w, grid_h)
