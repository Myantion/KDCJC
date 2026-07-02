from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

RESTORE_KEY_LENGTH = 16
EDGE_STRIP = 1
MAX_STRENGTH = 48
MIN_STRENGTH = 12


def derive_restore_key(seed: int) -> str:
    """由拼图排列 seed 派生第二密钥，专用于接缝像素柔化（与排列 seed 分离）。"""
    seed_bytes = seed.to_bytes(8, "big", signed=False)
    return hashlib.sha256(b"KDCJC-SOFTEN-V1" + seed_bytes).hexdigest()[:RESTORE_KEY_LENGTH].upper()


def _keystream(restore_key: str, length: int) -> np.ndarray:
    seed = hashlib.sha256(restore_key.encode("ascii")).digest()
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return np.frombuffer(bytes(out[:length]), dtype=np.uint8)


def _seam_count(grid_w: int, grid_h: int, height: int, width: int, strip: int) -> int:
    vertical = max(grid_w - 1, 0) * height * strip
    horizontal = max(grid_h - 1, 0) * width * strip
    return vertical + horizontal


def _strength_from_byte(value: int) -> int:
    span = MAX_STRENGTH - MIN_STRENGTH + 1
    return MIN_STRENGTH + int(value % span)


def _soften_pair(left: np.ndarray, right: np.ndarray, strength: int) -> tuple[np.ndarray, np.ndarray]:
    diff = (right.astype(np.int32) - left.astype(np.int32)) * strength // 256
    left_new = np.clip(left.astype(np.int32) + diff, 0, 255).astype(np.uint8)
    right_new = np.clip(right.astype(np.int32) - diff, 0, 255).astype(np.uint8)
    return left_new, right_new


def _unsoften_pair(left_new: np.ndarray, right_new: np.ndarray, strength: int) -> tuple[np.ndarray, np.ndarray]:
    denom = 256 - 2 * strength
    if denom <= 0:
        raise ValueError("接缝参数异常，无法还原")
    total = left_new.astype(np.int32) + right_new.astype(np.int32)
    delta = left_new.astype(np.int32) - right_new.astype(np.int32)
    original_delta = (delta * 256) // denom
    left = np.clip((total + original_delta) // 2, 0, 255).astype(np.uint8)
    right = np.clip((total - original_delta) // 2, 0, 255).astype(np.uint8)
    return left, right


def _process_vertical_seams(
    out: np.ndarray,
    grid_width: int,
    block_size: int,
    height: int,
    width: int,
    strip: int,
    stream: np.ndarray,
    idx: int,
    reverse: bool,
) -> int:
    cols = reversed(range(1, grid_width)) if reverse else range(1, grid_width)
    for col in cols:
        x = col * block_size
        offsets = reversed(range(strip)) if reverse else range(strip)
        for offset in offsets:
            x_left = x - offset - 1
            x_right = x + offset
            if x_left < 0 or x_right >= width:
                continue
            ys = reversed(range(height)) if reverse else range(height)
            for y in ys:
                if reverse:
                    idx -= 1
                    strength = _strength_from_byte(int(stream[idx]))
                    left, right = _unsoften_pair(out[y, x_left], out[y, x_right], strength)
                else:
                    strength = _strength_from_byte(int(stream[idx]))
                    idx += 1
                    left, right = _soften_pair(out[y, x_left], out[y, x_right], strength)
                out[y, x_left] = left
                out[y, x_right] = right
    return idx


def _process_horizontal_seams(
    out: np.ndarray,
    grid_height: int,
    block_size: int,
    height: int,
    width: int,
    strip: int,
    stream: np.ndarray,
    idx: int,
    reverse: bool,
) -> int:
    rows = reversed(range(1, grid_height)) if reverse else range(1, grid_height)
    for row in rows:
        y = row * block_size
        offsets = reversed(range(strip)) if reverse else range(strip)
        for offset in offsets:
            y_top = y - offset - 1
            y_bottom = y + offset
            if y_top < 0 or y_bottom >= height:
                continue
            xs = reversed(range(width)) if reverse else range(width)
            for x in xs:
                if reverse:
                    idx -= 1
                    strength = _strength_from_byte(int(stream[idx]))
                    top, bottom = _unsoften_pair(out[y_top, x], out[y_bottom, x], strength)
                else:
                    strength = _strength_from_byte(int(stream[idx]))
                    idx += 1
                    top, bottom = _soften_pair(out[y_top, x], out[y_bottom, x], strength)
                out[y_top, x] = top
                out[y_bottom, x] = bottom
    return idx


def apply_edge_cipher_from_seed(
    image: Image.Image,
    block_size: int,
    grid_width: int,
    grid_height: int,
    seed: int,
    strip: int = EDGE_STRIP,
) -> Image.Image:
    """移位后在块接缝做可逆像素柔化（实验性，默认关闭）。"""
    restore_key = derive_restore_key(seed)
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8).copy()
    height, width, _ = array.shape
    stream = _keystream(restore_key, _seam_count(grid_width, grid_height, height, width, strip))
    idx = 0
    idx = _process_vertical_seams(array, grid_width, block_size, height, width, strip, stream, idx, False)
    _process_horizontal_seams(array, grid_height, block_size, height, width, strip, stream, idx, False)
    return Image.fromarray(array, mode="RGB")


def remove_edge_cipher_from_seed(
    image: Image.Image,
    block_size: int,
    grid_width: int,
    grid_height: int,
    seed: int,
    strip: int = EDGE_STRIP,
) -> Image.Image:
    restore_key = derive_restore_key(seed)
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8).copy()
    height, width, _ = array.shape
    stream = _keystream(restore_key, _seam_count(grid_width, grid_height, height, width, strip))
    idx = _seam_count(grid_width, grid_height, height, width, strip)
    idx = _process_horizontal_seams(array, grid_height, block_size, height, width, strip, stream, idx, True)
    _process_vertical_seams(array, grid_width, block_size, height, width, strip, stream, idx, True)
    return Image.fromarray(array, mode="RGB")
