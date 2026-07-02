from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import zlib

import numpy as np
from PIL import Image

from kdcjc.crypto_key import pixel_msb_key, spread_rng_seed
from kdcjc.meta_pack import unpack_meta

PNG_META_KEY = "KDCJC"

_STEGO_MAGIC = b"KDCJ"
_FOOTER_MAGIC = b"KDCZ"
_PACKET_HEADER = struct.Struct(">4sII")
_FOOTER = struct.Struct(">4sIII")
_INTERIOR_MARGIN = 2
_SPARSE_VERSION = 3
_LSB_VERSION = 2
_SALT_BYTES = 16
_LENGTH_BYTES = 4

def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        bits = np.pad(bits, (0, 8 - bits.size % 8), constant_values=0)
    return np.packbits(bits).tobytes()


def _build_packet(payload: bytes, version: int = _SPARSE_VERSION) -> bytes:
    return _PACKET_HEADER.pack(_STEGO_MAGIC, version, len(payload)) + payload


def _parse_packet(packet: bytes) -> tuple[bytes | None, int]:
    if len(packet) < _PACKET_HEADER.size:
        return None, 0
    magic, version, length = _PACKET_HEADER.unpack_from(packet)
    if magic != _STEGO_MAGIC:
        return None, 0
    end = _PACKET_HEADER.size + length
    if length <= 0 or end > len(packet):
        return None, 0
    return packet[_PACKET_HEADER.size : end], version


def _valid_block_sizes(width: int, height: int) -> set[int]:
    return {
        block_size
        for block_size in range(4, 257)
        if width % block_size == 0 and height % block_size == 0
    }


def _candidate_block_sizes(width: int, height: int) -> list[int]:
    valid = _valid_block_sizes(width, height)
    preferred = [48, 40, 56, 32, 64, 72, 80, 24, 88, 96, 104, 112, 120, 128]
    ordered = [block_size for block_size in preferred if block_size in valid]
    ordered.extend(sorted(valid - set(ordered)))
    return ordered


def _fast_probe_block_sizes(width: int, height: int) -> list[int]:
    """还原时优先尝试常用块大小，非加密图可快速失败。"""
    valid = _valid_block_sizes(width, height)
    preferred = [32, 48, 40, 56, 64, 24, 72, 80, 88, 96, 104, 112, 120, 128]
    return [block_size for block_size in preferred if block_size in valid]


def _gui_block_sizes(width: int, height: int, skip: set[int] | None = None) -> list[int]:
    """GUI 可选的 4–128 步进 4 块大小，覆盖非常用尺寸。"""
    valid = _valid_block_sizes(width, height)
    skip = skip or set()
    return [block_size for block_size in range(4, 129, 4) if block_size in valid and block_size not in skip]


def _payload_matches_grid(payload: bytes, width: int, height: int, block_size: int) -> bool:
    try:
        meta = unpack_meta(payload)
    except ValueError:
        try:
            meta = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(meta, dict):
            return False

    grid_w = width // block_size
    grid_h = height // block_size
    if meta.get("block_size") != block_size:
        return False
    if meta.get("grid_width") not in (None, grid_w) and meta.get("grid_width") != grid_w:
        return False
    if meta.get("grid_height") not in (None, grid_h) and meta.get("grid_height") != grid_h:
        return False

    if meta.get("storage") == "derived":
        return len(meta.get("content_hash", b"")) == 32 and isinstance(meta.get("pile_labels"), list)

    if meta.get("storage") == "encrypted_restore":
        return len(meta.get("content_hash", b"")) == 32 and isinstance(meta.get("permutation"), list)

    if meta.get("storage") == "solver":
        return len(meta.get("content_hash", b"")) == 32

    permutation = meta.get("permutation")
    return isinstance(permutation, list) and len(permutation) == grid_w * grid_h


def _interior_coords(height: int, width: int, block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_w = width // block_size
    grid_h = height // block_size
    margin = _INTERIOR_MARGIN
    inner = block_size - 2 * margin
    if inner <= 0:
        margin = 0
        inner = block_size

    grid_rows = np.arange(grid_h, dtype=np.int32)
    grid_cols = np.arange(grid_w, dtype=np.int32)
    dy = np.arange(inner, dtype=np.int32)
    dx = np.arange(inner, dtype=np.int32)
    rows, cols, ddy, ddx = np.meshgrid(grid_rows, grid_cols, dy, dx, indexing="ij")
    ys = rows * block_size + margin + ddy
    xs = cols * block_size + margin + ddx
    return ys.ravel(), xs.ravel(), np.asarray([0, 1, 2], dtype=np.int32)


def _image_grad(array: np.ndarray) -> np.ndarray:
    stable = array.astype(np.uint16) & 0xFE
    gray = (
        stable[:, :, 0].astype(np.float32) * 0.299
        + stable[:, :, 1].astype(np.float32) * 0.587
        + stable[:, :, 2].astype(np.float32) * 0.114
    )
    gy, gx = np.gradient(gray)
    return gx * gx + gy * gy


def _texture_scores(array: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return _image_grad(array)[ys, xs]


def _top_sparse_flat_indices(
    grad: np.ndarray,
    width: int,
    height: int,
    block_size: int,
    bit_count: int,
) -> np.ndarray:
    ys, xs, _ = _interior_coords(height, width, block_size)
    scores = grad[ys, xs]
    channels = np.arange(ys.size, dtype=np.int64) % 3
    flat = (ys.astype(np.int64) * width + xs.astype(np.int64)) * 3 + channels

    if bit_count > flat.size:
        raise ValueError(
            f"块内像素容量不足（需要 {bit_count // 8} 字节，"
            f"可用约 {flat.size // 8} 字节）"
        )
    if bit_count == flat.size:
        order = np.argsort(scores, kind="stable")[::-1]
        return flat[order]

    top = np.argpartition(scores, -bit_count)[-bit_count:]
    order = top[np.argsort(scores[top], kind="stable")[::-1]]
    return flat[order]


def _sparse_indices(
    array: np.ndarray,
    block_size: int,
    bit_count: int,
    grad: np.ndarray | None = None,
) -> np.ndarray:
    height, width = array.shape[:2]
    if grad is None:
        grad = _image_grad(array)
    return _top_sparse_flat_indices(grad, width, height, block_size, bit_count)


def _embed_bits_at(flat: np.ndarray, indices: np.ndarray, bits: np.ndarray) -> None:
    flat[indices] &= 0xFE
    flat[indices] = (flat[indices] & 0xFE) | bits


def _extract_bits_at(flat: np.ndarray, indices: np.ndarray, bit_count: int) -> np.ndarray:
    return flat[indices[:bit_count]] & 1


def _embed_sparse(array: np.ndarray, block_size: int, payload: bytes) -> None:
    packet = _build_packet(payload, _SPARSE_VERSION)
    bits = _bytes_to_bits(packet)
    indices = _sparse_indices(array, block_size, bits.size)
    _embed_bits_at(array.reshape(-1), indices, bits)


def _extract_sparse(
    array: np.ndarray,
    block_size: int,
    grad: np.ndarray | None = None,
) -> bytes | None:
    height, width = array.shape[:2]
    flat = array.reshape(-1)
    header_bits = _PACKET_HEADER.size * 8
    try:
        header_indices = _sparse_indices(array, block_size, header_bits, grad=grad)
    except ValueError:
        return None
    header = _bits_to_bytes(_extract_bits_at(flat, header_indices, header_bits))
    if len(header) < _PACKET_HEADER.size:
        return None
    magic, version, length = _PACKET_HEADER.unpack(header)
    if magic != _STEGO_MAGIC or version != _SPARSE_VERSION or length <= 0:
        return None

    total_bits = header_bits + length * 8
    try:
        indices = _sparse_indices(array, block_size, total_bits, grad=grad)
    except ValueError:
        return None
    packet = _bits_to_bytes(_extract_bits_at(flat, indices, total_bits))[: _PACKET_HEADER.size + length]
    payload, _ = _parse_packet(packet)
    return payload


def _strip_sparse(array: np.ndarray, block_size: int, payload_len: int) -> None:
    packet = _build_packet(b"\x00" * payload_len, _SPARSE_VERSION)
    bits = _bytes_to_bits(packet)
    indices = _sparse_indices(array, block_size, bits.size)
    array.reshape(-1)[indices] &= 0xFE


def _stego_indices_lsb(height: int, width: int, block_size: int) -> np.ndarray:
    ys, xs, channels = _interior_coords(height, width, block_size)
    return (ys.astype(np.int64) * width + xs.astype(np.int64)) * 3 + (np.arange(ys.size) % 3)


def _embed_lsb(array: np.ndarray, block_size: int, payload: bytes) -> None:
    packet = _build_packet(payload, _LSB_VERSION)
    bits = _bytes_to_bits(packet)
    indices = _stego_indices_lsb(array.shape[0], array.shape[1], block_size)
    _embed_bits_at(array.reshape(-1), indices, bits)


def _extract_lsb(array: np.ndarray, block_size: int) -> bytes | None:
    height, width = array.shape[:2]
    flat = array.reshape(-1)
    indices = _stego_indices_lsb(height, width, block_size)
    header_bits = _PACKET_HEADER.size * 8
    if header_bits > indices.size:
        return None
    header = _bits_to_bytes(_extract_bits_at(flat, indices, header_bits))
    magic, version, length = _PACKET_HEADER.unpack(header)
    if magic != _STEGO_MAGIC or version != _LSB_VERSION or length <= 0:
        return None
    total_bits = header_bits + length * 8
    if total_bits > indices.size:
        return None
    packet = _bits_to_bytes(_extract_bits_at(flat, indices, total_bits))[: _PACKET_HEADER.size + length]
    payload, _ = _parse_packet(packet)
    return payload


def _strip_lsb(array: np.ndarray, block_size: int, payload_len: int) -> None:
    packet = _build_packet(b"\x00" * payload_len, _LSB_VERSION)
    bits = _bytes_to_bits(packet)
    indices = _stego_indices_lsb(array.shape[0], array.shape[1], block_size)
    array.reshape(-1)[indices[: bits.size]] &= 0xFE


def estimate_sparse_capacity(width: int, height: int, block_size: int) -> int:
    ys, xs, _ = _interior_coords(height, width, block_size)
    return ys.size // 8


def estimate_full_lsb_capacity(width: int, height: int, lsb_depth: int = 1) -> int:
    depth = max(1, min(7, int(lsb_depth)))
    total_bits = width * height * 3 * depth
    return max(0, (total_bits - _LENGTH_BYTES * 8) // 8)


def estimate_dual_plane_capacity(width: int, height: int) -> tuple[int, int]:
    """返回 (plane0 元数据容量, planes1-6 平滑容量)。"""
    plane0 = estimate_full_lsb_capacity(width, height, lsb_depth=1)
    smooth_bytes = width * height * 3 * 6 // 8
    return plane0, smooth_bytes


def _embed_at_plane(flat: np.ndarray, indices: np.ndarray, bits: np.ndarray, plane: int) -> None:
    mask = np.uint8(~(1 << plane) & 0xFF)
    flat[indices] = (flat[indices].astype(np.uint8) & mask) | (bits.astype(np.uint8) << plane)


def _read_at_plane(flat: np.ndarray, indices: np.ndarray, plane: int) -> np.ndarray:
    return (flat[indices].astype(np.uint8) >> plane) & 1


def _embed_plane_payload(
    flat: np.ndarray,
    width: int,
    height: int,
    pixel_key: bytes,
    payload: bytes,
    plane: int,
    stream_id: int,
    reserved: set[int] | None = None,
) -> None:
    body = struct.pack(">I", len(payload)) + payload
    body_bits = _bytes_to_bits(body)
    length_indices = _spread_indices(width, height, pixel_key, _LENGTH_BYTES * 8, stream_id, reserved)
    local_reserved = set(length_indices.tolist())
    if reserved:
        local_reserved |= reserved
    _embed_at_plane(flat, length_indices, body_bits[: _LENGTH_BYTES * 8], plane)
    payload_indices = _spread_indices(
        width,
        height,
        pixel_key,
        body_bits.size - _LENGTH_BYTES * 8,
        stream_id + 1000,
        local_reserved,
    )
    _embed_at_plane(flat, payload_indices, body_bits[_LENGTH_BYTES * 8 :], plane)


def _extract_plane_payload(
    flat: np.ndarray,
    width: int,
    height: int,
    pixel_key: bytes,
    plane: int,
    stream_id: int,
    reserved: set[int] | None = None,
    max_len: int | None = None,
) -> bytes | None:
    try:
        length_indices = _spread_indices(width, height, pixel_key, _LENGTH_BYTES * 8, stream_id, reserved)
    except ValueError:
        return None
    local_reserved = set(length_indices.tolist())
    if reserved:
        local_reserved |= reserved
    payload_len = struct.unpack(">I", _bits_to_bytes(_read_at_plane(flat, length_indices, plane)))[0]
    if max_len is not None and (payload_len <= 0 or payload_len > max_len):
        return None
    if max_len is None and payload_len <= 0:
        return None
    try:
        payload_indices = _spread_indices(
            width,
            height,
            pixel_key,
            payload_len * 8,
            stream_id + 1000,
            local_reserved,
        )
    except ValueError:
        return None
    return _bits_to_bytes(_read_at_plane(flat, payload_indices, plane))[:payload_len]


def _embed_interleaved_planes(
    flat: np.ndarray,
    width: int,
    height: int,
    pixel_key: bytes,
    payload: bytes,
    planes: tuple[int, ...],
    reserved_plane0: set[int] | None = None,
) -> None:
    bits = _bytes_to_bits(payload)
    if bits.size == 0:
        return
    plane_count = len(planes)
    for slot, plane in enumerate(planes):
        plane_bits = bits[slot::plane_count]
        if plane_bits.size == 0:
            continue
        reserved = reserved_plane0 if plane == 0 else None
        indices = _spread_indices(width, height, pixel_key, plane_bits.size, 2000 + plane, reserved)
        _embed_at_plane(flat, indices, plane_bits, plane)


def _extract_interleaved_planes(
    flat: np.ndarray,
    width: int,
    height: int,
    pixel_key: bytes,
    planes: tuple[int, ...],
    byte_len: int,
    reserved_plane0: set[int] | None = None,
) -> bytes | None:
    if byte_len <= 0:
        return b""
    bit_count = byte_len * 8
    plane_count = len(planes)
    chunks: list[np.ndarray] = []
    for slot, plane in enumerate(planes):
        slot_count = (bit_count - slot + plane_count - 1) // plane_count
        if slot_count <= 0:
            chunks.append(np.empty(0, dtype=np.uint8))
            continue
        reserved = reserved_plane0 if plane == 0 else None
        try:
            indices = _spread_indices(width, height, pixel_key, slot_count, 2000 + plane, reserved)
        except ValueError:
            return None
        chunks.append(_read_at_plane(flat, indices, plane))
    bits = np.zeros(bit_count, dtype=np.uint8)
    for slot, plane_bits in enumerate(chunks):
        if plane_bits.size:
            bits[slot::plane_count] = plane_bits[: bits[slot::plane_count].size]
    return _bits_to_bytes(bits)[:byte_len]


def embed_dual_spread_payload(
    array: np.ndarray,
    meta_payload: bytes,
    smooth_payload: bytes,
    pixel_key: bytes,
) -> None:
    embed_spread_payload(array, meta_payload, pixel_key)
    flat = array.reshape(-1)
    height, width, _ = array.shape
    _embed_interleaved_planes(
        flat,
        width,
        height,
        pixel_key,
        smooth_payload,
        planes=(1, 2, 3, 4, 5, 6),
    )


def extract_dual_spread_payload(
    array: np.ndarray,
    pixel_key: bytes,
    smooth_byte_len: int,
) -> tuple[bytes | None, bytes | None]:
    meta_payload = extract_spread_payload(array, pixel_key)
    if meta_payload is None:
        return None, None
    height, width, _ = array.shape
    flat = array.reshape(-1)
    smooth_payload = _extract_interleaved_planes(
        flat,
        width,
        height,
        pixel_key,
        planes=(1, 2, 3, 4, 5, 6),
        byte_len=smooth_byte_len,
    )
    return meta_payload, smooth_payload


def strip_dual_spread_payload(
    array: np.ndarray,
    pixel_key: bytes,
    meta_len: int,
    smooth_len: int,
) -> None:
    strip_spread_payload(array, pixel_key)
    if smooth_len > 0:
        height, width, _ = array.shape
        flat = array.reshape(-1)
        _embed_interleaved_planes(
            flat,
            width,
            height,
            pixel_key,
            b"\x00" * smooth_len,
            planes=(1, 2, 3, 4, 5, 6),
        )


def _spread_indices(
    width: int,
    height: int,
    pixel_key: bytes,
    bit_count: int,
    stream_id: int,
    reserved: set[int] | None = None,
) -> np.ndarray:
    if bit_count <= 0:
        return np.empty(0, dtype=np.int64)
    digest = spread_rng_seed(pixel_key, width, height, stream_id)
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    total = width * height * 3
    if reserved:
        pool = np.setdiff1d(np.arange(total, dtype=np.int64), np.fromiter(reserved, dtype=np.int64))
    else:
        pool = np.arange(total, dtype=np.int64)
    if bit_count > pool.size:
        raise ValueError(
            f"像素 LSB 容量不足（需要 {bit_count // 8} 字节，"
            f"可用约 {pool.size // 8} 字节）。"
            "请关闭块内平滑或换更大图片。"
        )
    return rng.choice(pool, size=bit_count, replace=False)


def _embed_at_indices(flat: np.ndarray, indices: np.ndarray, bits: np.ndarray) -> None:
    flat[indices] &= 0xFE
    flat[indices] = (flat[indices] & 0xFE) | bits.astype(np.uint8)


def _read_at_indices(flat: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return flat[indices] & 1


def embed_spread_payload(array: np.ndarray, payload: bytes, pixel_key: bytes) -> None:
    height, width, _ = array.shape
    body = struct.pack(">I", len(payload)) + payload
    body_bits = _bytes_to_bits(body)
    flat = array.reshape(-1)
    length_indices = _spread_indices(width, height, pixel_key, _LENGTH_BYTES * 8, 0)
    reserved = set(length_indices.tolist())
    _embed_at_indices(flat, length_indices, body_bits[: _LENGTH_BYTES * 8])
    payload_indices = _spread_indices(
        width, height, pixel_key, body_bits.size - _LENGTH_BYTES * 8, 1, reserved
    )
    _embed_at_indices(flat, payload_indices, body_bits[_LENGTH_BYTES * 8 :])


def extract_spread_payload(array: np.ndarray, pixel_key: bytes) -> bytes | None:
    height, width, _ = array.shape
    flat = array.reshape(-1)
    try:
        length_indices = _spread_indices(width, height, pixel_key, _LENGTH_BYTES * 8, 0)
    except ValueError:
        return None
    reserved = set(length_indices.tolist())
    payload_len = struct.unpack(">I", _bits_to_bytes(_read_at_plane(flat, length_indices, 0)))[0]
    max_len = estimate_full_lsb_capacity(width, height)
    if payload_len <= 0 or payload_len > max_len:
        return None
    try:
        payload_indices = _spread_indices(width, height, pixel_key, payload_len * 8, 1, reserved)
    except ValueError:
        return None
    return _bits_to_bytes(_read_at_plane(flat, payload_indices, 0))[:payload_len]


def strip_spread_payload(array: np.ndarray, pixel_key: bytes) -> None:
    payload = extract_spread_payload(array, pixel_key)
    if payload is None:
        return
    height, width, _ = array.shape
    flat = array.reshape(-1)
    body_bits = _bytes_to_bits(struct.pack(">I", len(payload)) + payload)
    length_indices = _spread_indices(width, height, pixel_key, _LENGTH_BYTES * 8, 0)
    reserved = set(length_indices.tolist())
    body_bits = _bytes_to_bits(struct.pack(">I", len(payload)) + payload)
    payload_indices = _spread_indices(
        width, height, pixel_key, body_bits.size - _LENGTH_BYTES * 8, 1, reserved
    )
    all_indices = np.concatenate([length_indices, payload_indices])
    flat[all_indices] &= 0xFE


def _extract_png_payload(image: Image.Image) -> bytes | None:
    text = getattr(image, "text", None) or {}
    encoded = text.get(PNG_META_KEY)
    if not encoded:
        return None
    try:
        return zlib.decompress(base64.b64decode(encoded.encode("ascii")))
    except (ValueError, zlib.error, OSError):
        return None


def has_png_payload(image: Image.Image) -> bool:
    return _extract_png_payload(image) is not None


def embed_payload(image: Image.Image, payload: bytes, pixel_key: bytes) -> Image.Image:
    """密钥隐含在像素 MSB 中，密文分散写入全图 LSB。"""
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8).copy()
    needed = _LENGTH_BYTES + len(payload)
    capacity = estimate_full_lsb_capacity(array.shape[1], array.shape[0])
    if needed > capacity:
        raise ValueError(
            f"像素 LSB 容量不足（需要 {needed} 字节，可用约 {capacity} 字节）。"
            "请关闭「块内均值平滑」或增大图片尺寸。"
        )
    embed_spread_payload(array, payload, pixel_key)
    return Image.fromarray(array, mode="RGB")


def strip_stego(
    image: Image.Image,
    block_size: int,
    pixel_key: bytes | None = None,
    payload_len: int | None = None,
    spread: bool = False,
) -> Image.Image:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8).copy()
    if spread and pixel_key:
        strip_spread_payload(array, pixel_key)
        return Image.fromarray(array, mode="RGB")
    if payload_len is None:
        payload = _extract_sparse(array, block_size)
        if payload is None:
            payload = _extract_lsb(array, block_size)
        payload_len = len(payload) if payload else 0

    if payload_len > 0:
        try:
            _strip_sparse(array, block_size, payload_len)
        except ValueError:
            _strip_lsb(array, block_size, payload_len)
    return Image.fromarray(array, mode="RGB")


def has_footer_payload(image: Image.Image) -> bool:
    return _extract_pixel_strip(image) is not None


def _extract_pixel_strip(image: Image.Image) -> bytes | None:
    rgb = image.convert("RGB")
    flat = np.asarray(rgb, dtype=np.uint8).reshape(-1)
    if flat.size < _FOOTER.size:
        return None

    magic, puzzle_w, puzzle_h, packet_len = _FOOTER.unpack(flat[-_FOOTER.size :].tobytes())
    if magic != _FOOTER_MAGIC:
        return None
    if puzzle_w != rgb.size[0] or puzzle_h <= 0 or packet_len <= 0:
        return None

    puzzle_bytes = puzzle_w * puzzle_h * 3
    if flat.size < puzzle_bytes + packet_len:
        return None

    packet = flat[puzzle_bytes : puzzle_bytes + packet_len].tobytes()
    payload, _ = _parse_packet(packet)
    return payload


def extract_payload(image: Image.Image) -> tuple[bytes | None, bool]:
    """返回 (wire_payload, is_spread)。spread 模式密钥由像素 MSB 派生。"""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pixel_key = pixel_msb_key(array)
    wire = extract_spread_payload(array, pixel_key)
    if wire is not None:
        try:
            unpack_meta(wire, pixel_key=pixel_key)
            return wire, True
        except ValueError:
            pass

    png_payload = _extract_png_payload(image)
    if png_payload is not None:
        return png_payload, False

    width, height = image.size

    footer_payload = _extract_pixel_strip(image)
    if footer_payload is not None:
        return footer_payload, False

    fast_sizes = _fast_probe_block_sizes(width, height)
    for block_size in fast_sizes:
        payload = _extract_lsb(array, block_size)
        if payload is not None and _payload_matches_grid(payload, width, height, block_size):
            return payload, False

    grad = _image_grad(array)
    saw_sparse_header = False
    for block_size in fast_sizes:
        payload = _extract_sparse(array, block_size, grad=grad)
        if payload is not None:
            if _payload_matches_grid(payload, width, height, block_size):
                return payload, False
            saw_sparse_header = True

    fast_set = set(fast_sizes)
    extra_sizes = _gui_block_sizes(width, height, skip=fast_set)
    if saw_sparse_header:
        extra_sizes.extend(
            block_size
            for block_size in _candidate_block_sizes(width, height)
            if block_size not in fast_set and block_size not in extra_sizes
        )

    for block_size in extra_sizes:
        payload = _extract_sparse(array, block_size, grad=grad)
        if payload is not None and _payload_matches_grid(payload, width, height, block_size):
            return payload, False
        payload = _extract_lsb(array, block_size)
        if payload is not None and _payload_matches_grid(payload, width, height, block_size):
            return payload, False

    return None, False


def puzzle_region(image: Image.Image, block_size: int | None = None, payload_len: int | None = None) -> Image.Image:
    rgb = image.convert("RGB")
    flat = np.asarray(rgb, dtype=np.uint8).reshape(-1)
    if flat.size >= _FOOTER.size:
        magic, puzzle_w, puzzle_h, _packet_len = _FOOTER.unpack(flat[-_FOOTER.size :].tobytes())
        if magic == _FOOTER_MAGIC and puzzle_w == rgb.size[0] and 0 < puzzle_h < rgb.size[1]:
            return rgb.crop((0, 0, puzzle_w, puzzle_h))

    if block_size is None:
        return rgb
    return strip_stego(rgb, block_size, payload_len)
