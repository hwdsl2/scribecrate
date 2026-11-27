"""
Pinned sherpa-onnx model selection and cache handling (standard library only).

https://github.com/hwdsl2/scribecrate

Copyright (C) 2026 Lin Song <linsongui@gmail.com>

This work is licensed under the MIT License
See: https://opensource.org/licenses/MIT

Model weights have their own upstream licenses; see the README.
"""

import argparse
import hashlib
import logging
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import NamedTuple
from urllib.parse import quote


logger = logging.getLogger("whisper_server.diarizer.models")


class ModelAsset(NamedTuple):
    size: int
    sha256: str


DEFAULT_EMBEDDING_MODEL = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
_EMBEDDING_URL = "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/"
_EMBEDDING_REVISION = "0743f301363dec56491a490f6d6cbc9d67f9a3bf"
_CAMPP_REVISION = "a8055d6daef723a6a4ecf4de233708575bf4994f"
# Sizes and SHA-256 values from the pinned revisions. Only CAM++ uses the
# newer exports. The other models retain the bytes used by existing caches.
EMBEDDING_MODELS = {
    '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx': ModelAsset(
        29596979, '98daf9bcf1c8abf3db374fec4c27a5eda7c58705bb3f180d057b085a72a88bb5'),
    '3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx': ModelAsset(
        28281139, 'e6af4d4c444f3dab01d991bb9b7d5af98eb979e329f6c8880a19d5553561ed25'),
    '3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx': ModelAsset(
        28281165, '53cab20ba8f2eb2f6e2d89ecc18e88ef3222169b61b4419cbc13666744a4574b'),
    '3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx': ModelAsset(
        39593765, 'e2d2048292e055f7b61cdec3db010503f35369b245bf0b3bbad021c9a91e4053'),
    '3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx': ModelAsset(
        39593761, '1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b'),
    '3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx': ModelAsset(
        116058710, '19547e85b6c14ec44b8add4e7cb9ce353c7e995d4f1c9ffd408176ac3a2d6895'),
    '3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx': ModelAsset(
        26485263, 'c59158379255ad66e161679cca6af8d52d51e389e3224ab7d7a7baae295c2db5'),
    '3dspeaker_speech_eres2net_sv_zh-cn_16k-common.onnx': ModelAsset(
        220623485, '2b9c4219b25326473524f006f1a09050ac28ccaf58c1f7dbc53e7631fa2fb1df'),
    '3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx': ModelAsset(
        71441526, 'bf1a75b9930474cf3389ef415e6e5d38ca96fea4a3a00f7e301d080a58ee2239'),
    'nemo_en_speakerverification_speakernet.onnx': ModelAsset(
        23411863, 'd204dc8aac0014b8543f05fc8e310510c7022bc65b6452c203ec205ef7a66b23'),
    'nemo_en_titanet_large.onnx': ModelAsset(
        101405493, 'd51abcf31717ef28162f26acb9d44dd4127c3d44c9b8624f699f3425daca8e77'),
    'nemo_en_titanet_small.onnx': ModelAsset(
        40257283, 'ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e'),
    'wespeaker_en_voxceleb_CAM++.onnx': ModelAsset(
        29281636, '7acd352a0372de23ce49379b02864833b7f3f501b4ea25248fc977389f08e5fc'),
    'wespeaker_en_voxceleb_CAM++_LM.onnx': ModelAsset(
        29281639, '231fe146b6ee64ffc39042e9953a1640fa3bd28c043ff3fcf0ea619e3aacb7ae'),
    'wespeaker_en_voxceleb_resnet152_LM.onnx': ModelAsset(
        79158470, '3e1c2c7a02097bdddc27c1cf7382237399abdc3b5b682e59455830673c6686e7'),
    'wespeaker_en_voxceleb_resnet221_LM.onnx': ModelAsset(
        95037599, '182f4ae144d70dfeb78064f6d507f8aada35c732a782631152a2626a4f20a60a'),
    'wespeaker_en_voxceleb_resnet293_LM.onnx': ModelAsset(
        114336527, 'f65dbc820e534eef64ae12d1e289e20244d60e60f7f00d7b092092b1c458be2e'),
    'wespeaker_en_voxceleb_resnet34.onnx': ModelAsset(
        26534365, '5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94'),
    'wespeaker_en_voxceleb_resnet34_LM.onnx': ModelAsset(
        26530550, 'e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012'),
    'wespeaker_zh_cnceleb_resnet34.onnx': ModelAsset(
        26534363, 'f86cd6c509f331f0e20b07bd48d1b2eb7de54202643401c4e84695ac861a0e5a'),
    'wespeaker_zh_cnceleb_resnet34_LM.onnx': ModelAsset(
        26530548, '87d1d5068397f3792c730570b53d66cd8be1da7ea22dd04f5b6706d96a3cd168'),
}
# Exact bytes from _EMBEDDING_REVISION, retained for verified offline reuse.
_LEGACY_CAMPP_MODELS = {
    '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx': ModelAsset(
        29596978, '357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b'),
    '3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx': ModelAsset(
        28281138, 'f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11'),
    '3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx': ModelAsset(
        28281164, 'aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2'),
    'wespeaker_en_voxceleb_CAM++.onnx': ModelAsset(
        29292684, 'c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef'),
    'wespeaker_en_voxceleb_CAM++_LM.onnx': ModelAsset(
        29292687, 'e197af7e9d473030cf486b3124149a19bf37014d0e4485e4c70c483b0ec10cb2'),
}

SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
_SEGMENTATION_URL = (
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/"
    "9403a6902bb58e3d5ae8c7e77c3422de279db2e0/"
)
# Model and license bytes match the original sherpa-onnx GitHub archive.
_SEGMENTATION_ASSET = ModelAsset(
    5992913, "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079")
_SEGMENTATION_LICENSE = ModelAsset(
    1061, "14d7016ad68e7394d6e6b78d96cc2ae431c905287b89674cfdf021e79e62b8ba")

_DOWNLOAD_MAX_TIME = 900
_LICENSE_MAX_TIME = 30
_DOWNLOAD_RETRIES = 2


def _download_process_timeout(max_time: int) -> int:
    return max_time * (_DOWNLOAD_RETRIES + 1) + 30


def startup_timeout_seconds(*, diarization_enabled: bool, local_files_only: bool) -> int:
    """Retain the normal startup window, adding online ONNX download budgets."""
    timeout = 300
    if diarization_enabled and not local_files_only:
        # Budget both model downloads and a missing segmentation license.
        # Retain the Whisper startup allowance and a minute for native initialization.
        timeout += (2 * _download_process_timeout(_DOWNLOAD_MAX_TIME)
                    + _download_process_timeout(_LICENSE_MAX_TIME) + 60)
    return timeout


def resolve_embedding_model(value: str = "") -> str:
    """Accept only a known filename; callers normalize environment values first."""
    if type(value) is not str or (value and value not in EMBEDDING_MODELS):
        raise ValueError(
            "WHISPER_DIARIZE_EMBEDDING_MODEL must be one of: "
            + ", ".join(EMBEDDING_MODELS) + "."
        )
    return value or DEFAULT_EMBEDDING_MODEL


def _verified(path: Path, asset: ModelAsset) -> bool:
    """Verify a regular file or symlink target; preserve invalid cache entries."""
    try:
        # Follow cache links, then inspect the opened target before reading it.
        # Nonblocking opens prevent FIFO targets from hanging model startup.
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError as exc:
        if path.is_symlink():
            raise RuntimeError(
                f"Cached model symbolic link has no existing target: {path}. "
                "Repair the link or move it aside and retry with downloads enabled."
            ) from exc
        return False
    except OSError as exc:
        raise RuntimeError(f"Cannot safely read cached model {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != asset.size:
            valid = False
        else:
            digest = hashlib.sha256()
            with os.fdopen(fd, "rb", closefd=False) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            valid = digest.hexdigest() == asset.sha256
    finally:
        os.close(fd)
    if not valid:
        raise RuntimeError(
            f"Cached model failed size/SHA-256 verification: {path}. "
            "Move this file aside and retry with downloads enabled."
        )
    return True


def _download_file(url: str, dest: Path, asset: ModelAsset, *,
                   max_time: int = _DOWNLOAD_MAX_TIME) -> None:
    logger.info("Downloading %s", url)
    subprocess.run([
        "curl", "--disable", "--fail", "--location", "--silent", "--show-error",
        "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", "15", "--max-time", str(max_time),
        "--retry", str(_DOWNLOAD_RETRIES), "--retry-delay", "2", "--max-filesize", str(asset.size),
        "--output", str(dest), url,
    ], check=True, timeout=_download_process_timeout(max_time))
    if not _verified(dest, asset):
        raise RuntimeError(f"Download did not produce the expected file: {dest}")


def _temporary(parent: Path):
    # Clean up staged downloads on transfer, verification or publication failure.
    return tempfile.TemporaryDirectory(prefix=".diarization-", dir=parent)


def _publish(source: Path, destination: Path) -> None:
    os.chmod(source, 0o644)
    os.replace(source, destination)


def ensure_models(cache_dir: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                  *, local_files_only: bool = False) -> tuple[str, str]:
    """Verify/reuse the selected models, downloading atomically when permitted.

    Cache directory ownership is trusted, as with the Whisper model cache.
    The corrected CAM++ files use digest-specific paths for offline reuse and rollback.
    Model-file and segmentation-directory symlinks are supported; model targets
    must be existing regular files that pass verification. Invalid links remain
    untouched so shared cache files are not silently replaced.
    """
    name = resolve_embedding_model(embedding_model)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    seg_path = root / SEGMENTATION_MODEL
    flat_emb_path = root / name
    asset = EMBEDDING_MODELS[name]
    legacy_asset = _LEGACY_CAMPP_MODELS.get(name)
    emb_path = flat_emb_path
    if legacy_asset is not None:
        emb_path = root / "diarization" / "embeddings" / asset.sha256 / name
    if seg_path.parent.is_symlink() and not seg_path.parent.is_dir():
        raise RuntimeError(
            f"Segmentation cache directory link must point to an existing directory: {seg_path.parent}. "
            "Repair the link or move it aside and retry."
        )
    seg_path.parent.mkdir(exist_ok=True)
    missing_seg = not _verified(seg_path, _SEGMENTATION_ASSET)
    missing_emb = not _verified(emb_path, asset)
    if missing_emb and legacy_asset is not None:
        legacy_cached = _verified(flat_emb_path, legacy_asset)
        if legacy_cached and local_files_only and not missing_seg:
            logger.warning(
                "Using the older cached CAM++ model %s in offline mode. "
                "Prefetch the corrected model with 'whisper_manage --downloaddiarize %s'.",
                name, name,
            )
            return str(seg_path), str(flat_emb_path)
    if local_files_only and (missing_seg or missing_emb):
        missing = seg_path if missing_seg else emb_path
        raise RuntimeError(f"WHISPER_LOCAL_ONLY prevents downloading missing diarization model: {missing}")

    if missing_seg:
        with _temporary(seg_path.parent) as directory:
            work = Path(directory)
            download = work / seg_path.name
            license_path = seg_path.with_name("LICENSE")
            license_download = work / license_path.name
            _download_file(_SEGMENTATION_URL + seg_path.name, download, _SEGMENTATION_ASSET)
            missing_license = not os.path.lexists(license_path)
            if missing_license:
                _download_file(_SEGMENTATION_URL + license_path.name, license_download,
                               _SEGMENTATION_LICENSE, max_time=_LICENSE_MAX_TIME)
            if not _verified(seg_path, _SEGMENTATION_ASSET):
                # Publish after both files verify. Preserve any existing license.
                if missing_license and not os.path.lexists(license_path):
                    _publish(license_download, license_path)
                _publish(download, seg_path)

    if missing_emb:
        revision = _CAMPP_REVISION if legacy_asset is not None else _EMBEDDING_REVISION
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        with _temporary(emb_path.parent) as directory:
            download = Path(directory) / name
            _download_file(f"{_EMBEDDING_URL}{revision}/{quote(name, safe='')}", download, asset)
            if not _verified(emb_path, asset):
                _publish(download, emb_path)
    return str(seg_path), str(emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="/var/lib/whisper")
    parser.add_argument("--embedding-model", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        paths = ensure_models(args.cache_dir, args.embedding_model)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print("Speaker embedding model: " + resolve_embedding_model(args.embedding_model))
    print("Diarization models verified: " + ", ".join(paths))
