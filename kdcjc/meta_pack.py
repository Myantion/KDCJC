from __future__ import annotations

import hashlib
import os
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kdcjc.core import KDCJCResult

_APP_KEY = hashlib.sha256(b"KDCJC-INTERNAL-RESTORE-KEY-V1").digest()
_META_MAGIC = b"KDCB"
_HEADER_V11 = struct.Struct(">4sBHHHHH12s")
_HEADER_V12 = struct.Struct(">4sBHHHHHQ")
_FLAG_PERM16 = 1
_FORMAT_VERSION_DERIVED = 13
FORMAT_VERSION_DERIVED = 13
FORMAT_VERSION_PIXEL = 14
_INNER_HEADER_V14 = struct.Struct(">BHHHHH")


def _encrypt_blob_with_key(key: bytes, plain: bytes) -> tuple[bytes, bytes]:
    nonce = os.urandom(12)
    encrypted = AESGCM(key).encrypt(nonce, plain, _META_MAGIC)
    return nonce, encrypted


def _decrypt_blob_with_key(key: bytes, nonce: bytes, encrypted: bytes) -> bytes:
    return AESGCM(key).decrypt(nonce, encrypted, _META_MAGIC)


def _encrypt_blob(plain: bytes) -> tuple[bytes, bytes]:
    return _encrypt_blob_with_key(_APP_KEY, plain)


def _decrypt_blob(nonce: bytes, encrypted: bytes) -> bytes:
    return _decrypt_blob_with_key(_APP_KEY, nonce, encrypted)


_METHOD_CODES = {"pca": 0, "umap": 1, "none": 2}
_LAYOUT_CODES = {"heap": 0, "strip": 1, "voronoi": 2}
_METHOD_NAMES = {v: k for k, v in _METHOD_CODES.items()}
_LAYOUT_NAMES = {v: k for k, v in _LAYOUT_CODES.items()}


def _pack_soften_byte(result: KDCJCResult) -> int:
    value = max(0, min(3, int(result.soften_strip)))
    if result.pile_toning:
        value |= 4
    if result.block_smooth:
        value |= 8
    return value


def _unpack_soften_byte(value: int) -> tuple[int, bool, bool]:
    return value & 3, bool(value & 4), bool(value & 8)


def _pack_derived_blob(result: KDCJCResult, smooth_max_bytes: int = 90000) -> bytes:
    method = _METHOD_CODES.get(result.cluster_method, 0)
    layout = _LAYOUT_CODES.get(result.pile_layout, 0)
    soften = _pack_soften_byte(result)
    piles = bytes(result.pile_labels)
    rotations = bytes(result.rotations)
    payload = (
        struct.pack(">QBBB", result.seed & 0xFFFFFFFFFFFFFFFF, method, layout, soften)
        + piles
        + rotations
    )
    if result.pile_toning and result.toning_deltas is not None:
        from kdcjc.toning import pack_toning_deltas

        payload += pack_toning_deltas(result.toning_deltas)
    if result.block_smooth and result.smooth_residuals is not None:
        from kdcjc.smooth import pack_smooth_residuals

        smooth_blob = pack_smooth_residuals(result.smooth_residuals, max_bytes=smooth_max_bytes)
        payload += struct.pack(">I", len(smooth_blob)) + smooth_blob
    return payload


def _unpack_derived_blob(
    payload: bytes, block_count: int
) -> tuple[int, str, str, int, bool, bool, list[int], list[int], bytes, bytes]:
    if len(payload) < 10 + block_count * 2:
        raise ValueError("像素内还原数据已损坏")
    if len(payload) >= 11 + block_count * 2:
        seed, method_code, layout_code, soften_code = struct.unpack_from(">QBBB", payload, 0)
        offset = 11
        soften_strip, pile_toning, block_smooth = _unpack_soften_byte(soften_code)
    else:
        seed, method_code, layout_code = struct.unpack_from(">QBB", payload, 0)
        offset = 10
        soften_strip = 0
        pile_toning = False
        block_smooth = False
    pile_labels = list(payload[offset : offset + block_count])
    offset += block_count
    rotations = list(payload[offset : offset + block_count])
    offset += block_count
    tail = payload[offset:]
    toning_tail = tail
    smooth_blob = b""
    if pile_toning:
        toning_bytes = block_count * 3
        if len(tail) < toning_bytes:
            raise ValueError("像素内色调偏移数据已损坏")
        toning_tail = tail[toning_bytes:]
        tail = tail[:toning_bytes]
    if block_smooth:
        if len(toning_tail) < 4:
            raise ValueError("块内平滑残差数据已损坏")
        smooth_len = struct.unpack_from(">I", toning_tail, 0)[0]
        smooth_start = 4
        smooth_end = smooth_start + smooth_len
        if smooth_len <= 0 or smooth_end > len(toning_tail):
            raise ValueError("块内平滑残差数据已损坏")
        smooth_blob = toning_tail[smooth_start:smooth_end]
    return (
        seed,
        _METHOD_NAMES.get(method_code, "pca"),
        _LAYOUT_NAMES.get(layout_code, "heap"),
        soften_strip,
        pile_toning,
        block_smooth,
        pile_labels,
        rotations,
        tail,
        smooth_blob,
    )


def pack_meta(
    result: KDCJCResult,
    original_width: int,
    original_height: int,
    pixel_key: bytes,
    smooth_max_bytes: int = 90000,
) -> bytes:
    """v14：密钥由像素 MSB 隐含派生，元数据 AES 加密后写入 LSB。"""
    if not result.content_hash:
        raise ValueError("缺少内容校验信息")
    if len(pixel_key) != 32:
        raise ValueError("像素密钥无效")

    inner_plain = (
        _INNER_HEADER_V14.pack(
            FORMAT_VERSION_PIXEL,
            result.block_size,
            result.grid_width,
            result.grid_height,
            original_width,
            original_height,
        )
        + _pack_derived_blob(result, smooth_max_bytes=smooth_max_bytes)
        + result.content_hash
    )
    nonce, encrypted = _encrypt_blob_with_key(pixel_key, inner_plain)
    return nonce + encrypted + result.content_hash


def _unpack_v14_inner(plain: bytes) -> dict:
    if len(plain) < _INNER_HEADER_V14.size + 32:
        raise ValueError("像素内还原数据已损坏")
    version, block_size, grid_w, grid_h, orig_w, orig_h = _INNER_HEADER_V14.unpack_from(plain)
    if version != FORMAT_VERSION_PIXEL:
        raise ValueError("像素内还原数据已损坏")
    content_hash = plain[-32:]
    derived_end = len(plain) - 32
    derived_start = _INNER_HEADER_V14.size
    block_count = grid_w * grid_h
    (
        seed,
        cluster_method,
        pile_layout,
        soften_strip,
        pile_toning,
        block_smooth,
        pile_labels,
        rotations,
        toning_tail,
        smooth_blob,
    ) = _unpack_derived_blob(plain[derived_start:derived_end], block_count)
    toning_deltas = None
    if pile_toning:
        from kdcjc.toning import unpack_toning_deltas

        toning_deltas = unpack_toning_deltas(toning_tail, block_count)
    smooth_residuals = None
    if block_smooth:
        from kdcjc.smooth import unpack_smooth_residuals

        smooth_residuals = unpack_smooth_residuals(smooth_blob, block_count, block_size)
    return {
        "version": version,
        "storage": "derived",
        "edge_cipher": soften_strip > 0,
        "soften_strip": soften_strip,
        "pile_toning": pile_toning,
        "block_smooth": block_smooth,
        "smooth_mode": smooth_blob[0] if block_smooth and smooth_blob else 0,
        "toning_deltas": toning_deltas,
        "smooth_residuals": smooth_residuals,
        "block_size": block_size,
        "grid_width": grid_w,
        "grid_height": grid_h,
        "original_width": orig_w,
        "original_height": orig_h,
        "padded_width": grid_w * block_size,
        "padded_height": grid_h * block_size,
        "content_hash": content_hash,
        "seed": seed,
        "pile_labels": pile_labels,
        "rotations": rotations,
        "cluster_method": cluster_method,
        "pile_layout": pile_layout,
        "pixel_key_derived": True,
    }


def unpack_meta(payload: bytes, pixel_key: bytes | None = None) -> dict:
    if pixel_key and len(payload) >= 44:
        nonce = payload[:12]
        encrypted = payload[12:-32]
        wire_hash = payload[-32:]
        try:
            plain = _decrypt_blob_with_key(pixel_key, nonce, encrypted)
        except Exception as exc:
            raise ValueError("像素内还原数据已损坏（密钥与图像不匹配）") from exc
        if plain[-32:] != wire_hash:
            raise ValueError("像素内还原数据已损坏")
        return _unpack_v14_inner(plain)

    if len(payload) >= _HEADER_V12.size + 12 + 16 + 32 and payload.startswith(_META_MAGIC):
        return _unpack_meta_legacy(payload)

    raise ValueError("不是 KDCJC 加密图片（像素内无有效还原信息）")


def _unpack_meta_legacy(payload: bytes) -> dict:
    if len(payload) < _HEADER_V12.size + 12 + 16 + 32:
        raise ValueError("像素内还原数据已损坏")

    magic, version, block_size, grid_w, grid_h, orig_w, orig_h, seed = _HEADER_V12.unpack_from(payload)
    if magic != _META_MAGIC:
        raise ValueError("不是 KDCJC 加密图片（像素内无有效还原信息）")

    if version >= FORMAT_VERSION_DERIVED:
        offset = _HEADER_V12.size
        nonce = payload[offset : offset + 12]
        encrypted_end = len(payload) - 32
        encrypted = payload[offset + 12 : encrypted_end]
        content_hash = payload[encrypted_end:]
        block_count = grid_w * grid_h
        plain = _decrypt_blob(nonce, encrypted)
        (
            blob_seed,
            cluster_method,
            pile_layout,
            soften_strip,
            pile_toning,
            block_smooth,
            pile_labels,
            rotations,
            toning_tail,
            smooth_blob,
        ) = _unpack_derived_blob(plain, block_count)
        if blob_seed != seed:
            raise ValueError("像素内还原数据已损坏")
        toning_deltas = None
        if pile_toning:
            from kdcjc.toning import unpack_toning_deltas

            toning_deltas = unpack_toning_deltas(toning_tail, block_count)
        smooth_residuals = None
        if block_smooth:
            from kdcjc.smooth import unpack_smooth_residuals

            smooth_residuals = unpack_smooth_residuals(smooth_blob, block_count, block_size)
        return {
            "version": version,
            "storage": "derived",
            "edge_cipher": soften_strip > 0,
            "soften_strip": soften_strip,
            "pile_toning": pile_toning,
            "block_smooth": block_smooth,
            "toning_deltas": toning_deltas,
            "smooth_residuals": smooth_residuals,
            "block_size": block_size,
            "grid_width": grid_w,
            "grid_height": grid_h,
            "original_width": orig_w,
            "original_height": orig_h,
            "padded_width": grid_w * block_size,
            "padded_height": grid_h * block_size,
            "content_hash": content_hash,
            "seed": seed,
            "pile_labels": pile_labels,
            "rotations": rotations,
            "cluster_method": cluster_method,
            "pile_layout": pile_layout,
            "pixel_key_derived": False,
        }

    if len(payload) < _HEADER_V11.size + 16 + 32:
        raise ValueError("像素内还原数据已损坏")

    magic11, version11, block_size, grid_w, grid_h, orig_w, orig_h, nonce = _HEADER_V11.unpack_from(payload)
    encrypted_end = len(payload) - 32
    encrypted = payload[_HEADER_V11.size : encrypted_end]
    content_hash = payload[encrypted_end:]
    plain = _decrypt_blob(nonce, encrypted)
    permutation, rotations = _unpack_restore_blob_v11(plain)
    return {
        "version": version11,
        "storage": "encrypted_restore",
        "edge_cipher": False,
        "block_size": block_size,
        "grid_width": grid_w,
        "grid_height": grid_h,
        "original_width": orig_w,
        "original_height": orig_h,
        "padded_width": grid_w * block_size,
        "padded_height": grid_h * block_size,
        "permutation": permutation,
        "rotations": rotations,
        "content_hash": content_hash,
        "seed": 0,
    }


def _unpack_restore_blob_v11(payload: bytes) -> tuple[list[int], list[int]]:
    flag = payload[0]
    count = payload[1]
    offset = 2
    if flag == 0:
        permutation = list(payload[offset : offset + count])
        offset += count
        rotations = list(payload[offset : offset + count])
        return permutation, rotations
    if flag == _FLAG_PERM16:
        count = payload[1] | (payload[2] << 8)
        offset = 3
        permutation = list(struct.unpack_from(f">{count}H", payload, offset))
        offset += count * 2
        rotations = list(payload[offset : offset + count])
        return permutation, rotations
    raise ValueError("像素内还原数据已损坏")
