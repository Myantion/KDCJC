from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from kdcjc.core import KDCJCParams, KDCJCResult, ProgressCallback, _pad_image, decrypt_image, encrypt_image
from kdcjc.crypto_key import pixel_msb_key
from kdcjc.meta_pack import pack_meta, unpack_meta
from kdcjc.derived import permutation_from_derived
from kdcjc.mask import apply_edge_cipher_from_seed, remove_edge_cipher_from_seed
from kdcjc.stego import embed_payload, estimate_full_lsb_capacity, extract_payload, has_footer_payload, has_png_payload, puzzle_region, strip_stego

KDCJC_MAGIC = "KDCJC1"
META_KEY = "KDCJC"
IMAGE_NAME = "image.png"
META_NAME = "meta.json"
LEGACY_META_NAME = "meta.enc"
FORMAT_VERSION = 14


def build_meta(result: KDCJCResult, original_width: int, original_height: int) -> dict:
    return {
        "version": FORMAT_VERSION,
        "storage": "derived",
        "edge_cipher": False,
        "block_size": result.block_size,
        "grid_width": result.grid_width,
        "grid_height": result.grid_height,
        "original_width": original_width,
        "original_height": original_height,
        "padded_width": result.padded_width,
        "padded_height": result.padded_height,
        "content_hash": result.content_hash,
        "seed": result.seed,
    }


def meta_to_bytes(
    result: KDCJCResult,
    original_width: int,
    original_height: int,
    pixel_key: bytes,
    smooth_max_bytes: int = 90000,
) -> bytes:
    return pack_meta(result, original_width, original_height, pixel_key, smooth_max_bytes=smooth_max_bytes)


def meta_from_bytes(payload: bytes, pixel_key: bytes | None = None) -> dict:
    if payload.startswith(b"KDCB"):
        try:
            return unpack_meta(payload)
        except ValueError as exc:
            raise ValueError("像素内还原数据已损坏") from exc
    try:
        return unpack_meta(payload, pixel_key=pixel_key)
    except ValueError as exc:
        raise ValueError("像素内还原数据已损坏") from exc


def _parse_meta_text(payload: str) -> dict:
    try:
        meta = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("不是有效的 KDCJC 加密图片") from exc

    if isinstance(meta, dict) and "enc" in meta:
        raise ValueError("这是旧版需密码的文件，请用旧版本解密后重新加密")

    if not isinstance(meta, dict) or "permutation" not in meta:
        raise ValueError("不是 KDCJC 加密图片（未找到还原信息）")
    return meta


def _map_progress(
    progress_callback: ProgressCallback | None,
    start: int,
    end: int,
) -> ProgressCallback | None:
    if progress_callback is None:
        return None
    span = max(end - start, 1)

    def mapped(done: int, total: int) -> None:
        overall = start + int(done * span / max(total, 1))
        progress_callback(min(overall, end), 100)

    return mapped


def _restore_from_meta(
    scrambled: Image.Image,
    meta: dict,
    progress_callback: ProgressCallback | None = None,
) -> Image.Image:
    def report(percent: int) -> None:
        if progress_callback:
            progress_callback(percent, 100)

    if meta.get("storage") == "derived":
        params = KDCJCParams(
            cluster_method=meta.get("cluster_method", "pca"),
            pile_layout=meta.get("pile_layout", "heap"),
        )
        block_count = int(meta["grid_width"]) * int(meta["grid_height"])
        seed = int(meta["seed"])
        pile_labels = list(meta["pile_labels"])
        rotations = list(meta["rotations"])
        report(52)
        permutation = permutation_from_derived(
            block_count,
            int(meta["grid_width"]),
            int(meta["grid_height"]),
            seed,
            pile_labels,
            params,
        )
        report(55)
        restored = decrypt_image(
            scrambled=scrambled,
            block_size=int(meta["block_size"]),
            permutation=permutation,
            grid_width=int(meta["grid_width"]),
            grid_height=int(meta["grid_height"]),
            original_width=int(meta["original_width"]),
            original_height=int(meta["original_height"]),
            rotations=rotations,
            pile_toning=bool(meta.get("pile_toning", False)),
            toning_deltas=meta.get("toning_deltas"),
            block_smooth=bool(meta.get("block_smooth", False)),
            smooth_residuals=meta.get("smooth_residuals"),
            seed=seed,
            progress_callback=_map_progress(progress_callback, 55, 98),
        )
        if meta.get("content_hash"):
            import hashlib
            from kdcjc.smooth import SMOOTH_POOL

            report(99)
            padded = _pad_image_for_hash(restored, int(meta["block_size"]), meta)
            digest = hashlib.sha256((padded & 0xFE).tobytes()).digest()
            lossy_smooth = int(meta.get("smooth_mode", 0)) == SMOOTH_POOL
            if digest != meta["content_hash"] and not lossy_smooth:
                raise ValueError("还原结果校验失败，图片可能已损坏或被篡改")
        report(100)
        return restored

    if meta.get("storage") == "solver":
        raise ValueError("旧版求解格式已不再支持，请用当前版本重新加密")

    rotations = meta.get("rotations")
    report(55)
    restored = decrypt_image(
        scrambled=scrambled,
        block_size=int(meta["block_size"]),
        permutation=list(meta["permutation"]),
        grid_width=int(meta["grid_width"]),
        grid_height=int(meta["grid_height"]),
        original_width=int(meta["original_width"]),
        original_height=int(meta["original_height"]),
        rotations=list(rotations) if rotations else None,
        progress_callback=_map_progress(progress_callback, 55, 98),
    )
    if meta.get("content_hash"):
        import hashlib

        report(99)
        padded = _pad_image_for_hash(restored, int(meta["block_size"]), meta)
        digest = hashlib.sha256((padded & 0xFE).tobytes()).digest()
        if digest != meta["content_hash"]:
            raise ValueError("还原结果校验失败，图片可能已损坏或被篡改")
    report(100)
    return restored


def _pad_image_for_hash(restored: Image.Image, block_size: int, meta: dict):
    import numpy as np

    rgb = restored.convert("RGB")
    padded, _, _ = _pad_image(rgb, block_size)
    return np.asarray(padded, dtype=np.uint8)


def _read_meta_from_image(image: Image.Image) -> tuple[dict, bytes, bool]:
    pixel_payload, spread = extract_payload(image)
    if pixel_payload is None:
        text = getattr(image, "text", None) or {}
        legacy = text.get(META_KEY)
        if legacy:
            return _parse_meta_text(legacy), b"", False
        raise ValueError("不是 KDCJC 加密图片（像素内未找到还原信息）")
    import numpy as np

    pixel_key = pixel_msb_key(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if spread:
        meta = meta_from_bytes(pixel_payload, pixel_key=pixel_key)
    elif pixel_payload.startswith(b"KDCB"):
        meta = meta_from_bytes(pixel_payload)
    else:
        meta = meta_from_bytes(pixel_payload, pixel_key=pixel_key)
    return meta, pixel_payload, spread


def _open_rgb_with_meta(path: Path) -> tuple[Image.Image, bytes | None, dict | None, bool, bool]:
    opened = Image.open(path)
    opened.load()
    from_png = has_png_payload(opened)
    meta, pixel_payload, spread = _read_meta_from_image(opened)
    return opened.convert("RGB"), pixel_payload, meta, spread, from_png


def _remove_edge_layer(image: Image.Image, meta: dict) -> Image.Image:
    soften_strip = int(meta.get("soften_strip", 0))
    if soften_strip <= 0 and not meta.get("edge_cipher", False):
        return image
    if soften_strip <= 0:
        soften_strip = 1
    return remove_edge_cipher_from_seed(
        image,
        int(meta["block_size"]),
        int(meta["grid_width"]),
        int(meta["grid_height"]),
        int(meta["seed"]),
        strip=soften_strip,
    )


def save_encrypted_image(
    path: str | Path,
    result: KDCJCResult,
    original_width: int,
    original_height: int,
) -> Image.Image:
    path = Path(path)
    suffix = path.suffix.lower()
    import numpy as np

    rgb = result.image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8)
    pixel_key = pixel_msb_key(array)
    capacity = estimate_full_lsb_capacity(array.shape[1], array.shape[0])
    smooth_budget = max(8000, capacity - 4096)
    meta_bytes = meta_to_bytes(
        result,
        original_width,
        original_height,
        pixel_key,
        smooth_max_bytes=smooth_budget,
    )
    output_image = embed_payload(result.image, meta_bytes, pixel_key)
    if result.soften_strip > 0:
        output_image = apply_edge_cipher_from_seed(
            output_image,
            result.block_size,
            result.grid_width,
            result.grid_height,
            result.seed,
            strip=result.soften_strip,
        )

    if suffix == ".kdcjc":
        _save_kdcjc_archive(path, output_image)
    elif suffix == ".png":
        output_image.save(path, format="PNG")
    else:
        output_image.save(path)

    return output_image


def _save_kdcjc_archive(path: Path, image: Image.Image) -> None:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("magic.txt", KDCJC_MAGIC)
        archive.writestr(IMAGE_NAME, buffer.getvalue())


def load_encrypted_image(
    path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Image.Image, Image.Image, dict]:
    def report(percent: int) -> None:
        if progress_callback:
            progress_callback(percent, 100)

    report(2)
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".kdcjc":
        stored, meta = _load_kdcjc_archive(path)
        pixel_payload = b""
        spread = meta.get("pixel_key_derived", False)
        from_png = False
    else:
        opened = Image.open(path)
        opened.load()
        report(8)
        from_png = has_png_payload(opened)
        report(10)
        meta, pixel_payload, spread = _read_meta_from_image(opened)
        report(33)
        stored = opened.convert("RGB")

    report(35)
    import numpy as np

    pixel_key = pixel_msb_key(np.asarray(stored.convert("RGB"), dtype=np.uint8))

    report(40)
    if pixel_payload or spread or meta.get("pixel_key_derived"):
        block_size = int(meta.get("block_size", 0))
        storage = meta.get("storage", "")
        softened_source = _remove_edge_layer(stored, meta)
        report(45)
        if spread or meta.get("pixel_key_derived"):
            scrambled_source = strip_stego(
                softened_source,
                block_size,
                pixel_key=pixel_key,
                spread=True,
            )
        elif from_png or has_png_payload(softened_source):
            scrambled_source = puzzle_region(softened_source, block_size or None)
        elif has_footer_payload(softened_source):
            scrambled_source = puzzle_region(softened_source, block_size, len(pixel_payload))
        elif storage in ("block_sparse", "block_lsb", "solver", "encrypted_restore", "derived") and block_size > 0:
            scrambled_source = strip_stego(softened_source, block_size, payload_len=len(pixel_payload))
        else:
            scrambled_source = puzzle_region(softened_source, block_size or None)
    else:
        scrambled_source = _remove_edge_layer(stored, meta)
    report(50)
    scrambled = scrambled_source
    restored = _restore_from_meta(scrambled, meta, progress_callback=progress_callback)
    puzzle_w = int(meta.get("padded_width", scrambled.width))
    puzzle_h = int(meta.get("padded_height", scrambled.height))
    scrambled = scrambled.crop((0, 0, puzzle_w, puzzle_h))
    stored = stored.crop((0, 0, puzzle_w, puzzle_h))
    report(100)
    return stored, restored, meta


def _load_kdcjc_archive(path: Path) -> tuple[Image.Image, dict]:
    with zipfile.ZipFile(path, "r") as archive:
        magic = archive.read("magic.txt").decode("utf-8").strip()
        if magic != KDCJC_MAGIC:
            raise ValueError("不是有效的 KDCJC 文件")
        if IMAGE_NAME not in archive.namelist():
            raise ValueError("KDCJC 文件缺少图像")
        image_bytes = archive.read(IMAGE_NAME)

    stored = Image.open(BytesIO(image_bytes))
    stored.load()
    meta, _, _spread = _read_meta_from_image(stored)
    return stored.convert("RGB"), meta


def save_kdcjc(
    path: str | Path,
    result: KDCJCResult,
    original_width: int,
    original_height: int,
) -> None:
    save_encrypted_image(path, result, original_width, original_height)


def load_kdcjc(path: str | Path) -> tuple[Image.Image, Image.Image, dict]:
    return load_encrypted_image(path)


def encrypt_file(
    input_path: str | Path,
    output_path: str | Path,
    params: KDCJCParams | None = None,
    progress_callback: ProgressCallback | None = None,
) -> KDCJCResult:
    image = Image.open(input_path).convert("RGB")
    original_size = image.size
    result = encrypt_image(image, params, progress_callback=progress_callback)
    result.image = save_encrypted_image(output_path, result, original_size[0], original_size[1])
    return result


def preview_encrypted_image(result: KDCJCResult, original_width: int, original_height: int) -> Image.Image:
    import numpy as np

    rgb = result.image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8)
    pixel_key = pixel_msb_key(array)
    capacity = estimate_full_lsb_capacity(array.shape[1], array.shape[0])
    smooth_budget = max(8000, capacity - 4096)
    meta_bytes = meta_to_bytes(
        result,
        original_width,
        original_height,
        pixel_key,
        smooth_max_bytes=smooth_budget,
    )
    output = embed_payload(result.image, meta_bytes, pixel_key)
    if result.soften_strip > 0:
        output = apply_edge_cipher_from_seed(
            output,
            result.block_size,
            result.grid_width,
            result.grid_height,
            result.seed,
            strip=result.soften_strip,
        )
    return output
