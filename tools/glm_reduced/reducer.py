# SPDX-License-Identifier: Apache-2.0
"""Checkpoint reduction engine: plan, execute, verify.

The engine operates on raw safetensors bytes (see ``safetensors_io``), so it
streams arbitrarily large shards without loading the whole checkpoint and
preserves tensor payloads bit-for-bit. It never mutates the source directory,
refuses unsafe output paths, stages output in a sibling directory and
publishes atomically with ``os.rename``.

Tensor classification is driven by the profile's exact naming rules (verified
against pinned upstream index snapshots), not by generic substring guesses:
vision subtrees are matched before decoder-layer roots, and anything
unrecognized aborts the build unless explicitly allowed and recorded.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import regex as re

from . import MANIFEST_FILENAME, MANIFEST_SCHEMA, TOOL_NAME, TOOL_VERSION
from .errors import (
    ManifestError,
    ReductionError,
    SafetyError,
    UnknownTensorError,
    UnsupportedFormatError,
    UnsupportedQuantError,
)
from .profiles import (
    PER_LAYER_ARRAY_KEYS,
    NamingRules,
    ReductionProfile,
    read_layer_count,
    truncate_layer_config,
    validate_layer_arrays,
)
from .safetensors_io import (
    OutTensor,
    SafetensorsFile,
    plan_shards,
    read_safetensors,
    sha256_of_file,
    shard_filename,
    write_shard,
)

# Index files recognized at checkpoint top level. The second name is the one
# Ascend ModelSlim-quantized checkpoints actually ship (see lab weight share).
INDEX_FILENAMES = ("model.safetensors.index.json", "quant_model_weights.safetensors.index.json")
QUANT_DESCRIPTION_FILENAME = "quant_model_description.json"


def resolve_index_filename(files: dict) -> str:
    """Return the weight-index name a source descriptor pins; fail closed if neither is present."""
    for name in INDEX_FILENAMES:
        if name in files:
            return name
    raise UnsupportedFormatError(f"source descriptor pins no known index file ({', '.join(INDEX_FILENAMES)})")


DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3

# Non-safetensors weight containers we refuse explicitly rather than guessing.
_REFUSED_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".gguf", ".ckpt", ".msgpack", ".h5")

# One of these must exist, otherwise the checkpoint cannot be served standalone.
_TOKENIZER_CANDIDATES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")

# Projector / rotation tensors that live outside layer and vision subtrees.
_EXTRA_GLOBAL_RE = re.compile(r"(^|\.)(mm_projector|multi_modal_projector|mapping_proj|projector)\.")

# quantization_config schemes whose per-tensor metadata is a plain companion
# tensor (weight_scale_inv) that follows its weight through any crop.
_SUPPORTED_CONFIG_QUANT = {"fp8"}

# Global (non-layer) keys allowed in a quant_model_description.json with
# non-string values; everything else must map a module path to a scheme string.
_QUANT_GLOBAL_KEYS = {"group_size", "metadata", "optional", "version", "is_rot_used"}


@dataclass
class SourceCheckpoint:
    src_dir: str
    config: dict
    config_sha256: str
    index_sha256: str | None
    shards: dict[str, SafetensorsFile]  # shard filename -> parsed header
    weight_map: dict[str, str]  # tensor name -> shard filename
    aux_files: list[str]  # relative paths copied verbatim
    quant_description: dict | None
    quant_description_sha256: str | None
    warnings: list[str] = field(default_factory=list)
    # Set when loaded via load_source_selective: {"missing_shards": [...]}.
    selective: dict | None = None


@dataclass
class TensorAction:
    src_name: str
    action: str  # "keep" | "remap" | "drop-layer"
    dst_name: str | None
    src_layer: int | None = None
    dst_layer: int | None = None


@dataclass
class Plan:
    profile: ReductionProfile
    keep_layers: int
    source_layers: int
    mtp_layers: int
    actions: list[TensorAction]
    new_config: dict
    new_quant_description: dict | None
    warnings: list[str] = field(default_factory=list)

    @property
    def kept(self) -> list[TensorAction]:
        return [a for a in self.actions if a.action in ("keep", "remap")]

    @property
    def dropped(self) -> list[TensorAction]:
        return [a for a in self.actions if a.action.startswith("drop")]


def _layer_anchor_ok(name: str) -> bool:
    """Index-declared shard names must be plain relative basenames."""
    if not name or name != PurePosixPath(name).name:
        return False
    if name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts:
        return False
    return not os.path.isabs(name) and not re.match(r"^[A-Za-z]:", name)


def _resolve_aux(path: Path, src: Path) -> Path:
    """Resolve a possibly-symlinked aux file read-only (HF caches symlink into
    blob stores outside the snapshot dir); the resolved target must exist."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SafetyError(f"auxiliary file {path.relative_to(src)} is an unresolvable symlink: {exc}") from exc
    if not resolved.is_file():
        raise SafetyError(f"auxiliary file {path.relative_to(src)} does not resolve to a regular file")
    return resolved


def load_source(src_dir: str) -> SourceCheckpoint:
    src = Path(src_dir)
    if not src.is_dir():
        raise UnsupportedFormatError(f"source checkpoint directory {src_dir!r} does not exist")
    refused = [
        str(p.relative_to(src)) for p in src.rglob("*") if p.is_file() and p.suffix.lower() in _REFUSED_WEIGHT_SUFFIXES
    ]
    if refused:
        raise UnsupportedFormatError(
            f"source contains non-safetensors weight files {sorted(refused)}; convert the checkpoint to "
            "safetensors (sharded or single-file) before reducing it"
        )
    config_path = src / "config.json"
    if not config_path.is_file():
        raise UnsupportedFormatError(f"source checkpoint {src_dir!r} has no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    warnings: list[str] = []
    shards: dict[str, SafetensorsFile] = {}
    weight_map: dict[str, str] = {}
    index_sha256: str | None = None
    index_name = next((name for name in INDEX_FILENAMES if (src / name).is_file()), None)
    if index_name is not None:
        index_path = src / index_name
        index_sha256 = sha256_of_file(str(index_path))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise UnsupportedFormatError(f"{index_name} has no usable weight_map")
        for name, shard_name in raw_map.items():
            if not isinstance(shard_name, str) or not _layer_anchor_ok(shard_name):
                raise UnsupportedFormatError(
                    f"{index_name} maps {name!r} to unsafe shard path {shard_name!r}; "
                    "shard names must be plain basenames inside the checkpoint directory"
                )
            if name in weight_map:
                raise UnsupportedFormatError(f"duplicate tensor {name!r} in {index_name}")
            if shard_name not in shards:
                shard_path = src / shard_name
                if not shard_path.is_file():
                    raise UnsupportedFormatError(f"index references missing shard {shard_name!r}")
                shards[shard_name] = read_safetensors(str(shard_path))
            if name not in shards[shard_name].tensors:
                raise UnsupportedFormatError(f"index maps {name!r} to {shard_name!r} but the shard lacks it")
            weight_map[name] = shard_name
        # Every shard declared by the index is fully covered, and no top-level
        # shard may exist outside the index (it would be silently dropped).
        tensors_by_shard: dict[str, set[str]] = defaultdict(set)
        for tensor_name, shard_name in weight_map.items():
            tensors_by_shard[shard_name].add(tensor_name)
        for shard_name, parsed in shards.items():
            extra = set(parsed.tensors) - tensors_by_shard[shard_name]
            if extra:
                raise UnsupportedFormatError(
                    f"shard {shard_name!r} contains tensors missing from {index_name}: {sorted(extra)[:5]}"
                )
        unindexed = sorted(p.name for p in src.glob("*.safetensors") if p.name not in shards)
        if unindexed:
            raise UnsupportedFormatError(
                f"top-level safetensors shards not covered by {index_name}: {unindexed}; refusing to silently drop them"
            )
    else:
        safetensors = sorted(p.name for p in src.glob("*.safetensors"))
        if not safetensors:
            raise UnsupportedFormatError(
                f"source checkpoint {src_dir!r} has no top-level safetensors weights and no index file "
                f"({', '.join(INDEX_FILENAMES)})"
            )
        for shard_name in safetensors:
            parsed = read_safetensors(str(src / shard_name))
            overlap = set(parsed.tensors) & set(weight_map)
            if overlap:
                raise UnsupportedFormatError(f"duplicate tensor names across unindexed shards: {sorted(overlap)[:5]}")
            shards[shard_name] = parsed
            for name in parsed.tensors:
                weight_map[name] = shard_name

    quant_description = None
    quant_description_sha256 = None
    quant_path = src / QUANT_DESCRIPTION_FILENAME
    if quant_path.is_file():
        quant_description = json.loads(quant_path.read_text(encoding="utf-8"))
        if not isinstance(quant_description, dict):
            raise UnsupportedQuantError(
                f"{QUANT_DESCRIPTION_FILENAME} must be a JSON object of module path -> scheme; "
                f"got {type(quant_description).__name__}"
            )
        quant_description_sha256 = sha256_of_file(str(quant_path))

    special = {QUANT_DESCRIPTION_FILENAME, MANIFEST_FILENAME, "config.json", *INDEX_FILENAMES}
    aux_files = []
    symlinked = 0
    for path in sorted(src.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        rel = path.relative_to(src).as_posix()
        if rel in special or (path.parent == src and path.suffix == ".safetensors"):
            continue
        # Safetensors in subdirectories (e.g. optional/quarot.safetensors) are
        # auxiliary assets referenced by the quant metadata; copy them verbatim
        # rather than treating them as layer weights or dropping them.
        _resolve_aux(path, src)  # raises on broken/unresolvable symlinks
        if path.is_symlink():
            symlinked += 1
        aux_files.append(rel)
    if symlinked:
        warnings.append(f"{symlinked} auxiliary files were symlinks; copied dereferenced read-only content")
    _validate_referenced_aux(src, quant_description)
    if not any(name in aux_files for name in _TOKENIZER_CANDIDATES):
        raise UnsupportedFormatError(
            f"source checkpoint {src_dir!r} has none of {_TOKENIZER_CANDIDATES}; a reduced checkpoint must "
            "remain standalone-servable, so the tokenizer/processor files are required"
        )
    return SourceCheckpoint(
        src_dir=str(src),
        config=config,
        config_sha256=sha256_of_file(str(config_path)),
        index_sha256=index_sha256,
        shards=shards,
        weight_map=weight_map,
        aux_files=aux_files,
        quant_description=quant_description,
        quant_description_sha256=quant_description_sha256,
        warnings=warnings,
    )


def _load_index(src: Path) -> tuple[str, dict, str]:
    """Locate and parse the top-level safetensors index; returns (name, index, sha256)."""
    index_name = next((name for name in INDEX_FILENAMES if (src / name).is_file()), None)
    if index_name is None:
        raise UnsupportedFormatError(f"source checkpoint {str(src)!r} has no index file ({', '.join(INDEX_FILENAMES)})")
    index_path = src / index_name
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return index_name, index, sha256_of_file(str(index_path))


def load_source_selective(src_dir: str, profile: ReductionProfile, keep_layers: int) -> SourceCheckpoint:
    """Selective-download mode: build from only the shards a reduction needs.

    The full original index is classified first; every shard containing at
    least one retained or remapped (MTP) tensor must be present and its full
    header is validated, and every needed tensor must exist in its shard.
    Shards containing only dropped-layer tensors may be absent (recorded in
    the manifest and warnings). The original index digest is preserved, so a
    selectively staged source cannot silently masquerade as a full one.
    """
    src = Path(src_dir)
    if not src.is_dir():
        raise UnsupportedFormatError(f"source checkpoint directory {src_dir!r} does not exist")
    config_path = src / "config.json"
    if not config_path.is_file():
        raise UnsupportedFormatError(f"source checkpoint {src_dir!r} has no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layer_config = _layer_config_of(profile, config)
    source_layers = read_layer_count(profile, layer_config)
    profile.validate_keep_layers(keep_layers, source_layers)
    problems = validate_layer_arrays(profile, layer_config, keep_layers, source_layers)
    if problems:
        raise ReductionError("config rejected:\n- " + "\n- ".join(problems))
    mtp_layers = int(layer_config.get("num_nextn_predict_layers") or 0)

    index_name, index, index_sha256 = _load_index(src)
    raw_map = index.get("weight_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise UnsupportedFormatError(f"{index_name} has no usable weight_map")
    needed_shards: set[str] = set()
    all_shards: set[str] = set()
    needed_tensors: dict[str, str] = {}
    for name, shard_name in raw_map.items():
        if not isinstance(shard_name, str) or not _layer_anchor_ok(shard_name):
            raise UnsupportedFormatError(
                f"{index_name} maps {name!r} to unsafe shard path {shard_name!r}; "
                "shard names must be plain basenames inside the checkpoint directory"
            )
        all_shards.add(shard_name)
        action = _classify(name, profile.naming, source_layers, keep_layers, mtp_layers, profile.keep_mtp)
        if action.action in ("keep", "remap"):
            needed_shards.add(shard_name)
            needed_tensors[name] = shard_name
    missing_needed = sorted(shard for shard in needed_shards if not (src / shard).is_file())
    if missing_needed:
        raise UnsupportedFormatError(
            f"selective source is missing {len(missing_needed)} REQUIRED shards: {missing_needed}; "
            "download them before building (see `required-shards`)"
        )
    missing_dropped = sorted(shard for shard in all_shards - needed_shards if not (src / shard).is_file())
    present_extra = sorted(p.name for p in src.glob("*.safetensors") if p.name not in all_shards)
    if present_extra:
        raise UnsupportedFormatError(
            f"top-level safetensors shards not covered by {index_name}: {present_extra}; refusing to silently drop them"
        )
    shards: dict[str, SafetensorsFile] = {}
    weight_map: dict[str, str] = {}
    for shard_name in sorted(needed_shards):
        parsed = read_safetensors(str(src / shard_name))
        shards[shard_name] = parsed
    for name, shard_name in needed_tensors.items():
        if name not in shards[shard_name].tensors:
            raise UnsupportedFormatError(f"index maps needed tensor {name!r} to {shard_name!r} but the shard lacks it")
        weight_map[name] = shard_name

    quant_description, quant_description_sha256 = None, None
    quant_path = src / QUANT_DESCRIPTION_FILENAME
    if quant_path.is_file():
        quant_description = json.loads(quant_path.read_text(encoding="utf-8"))
        if not isinstance(quant_description, dict):
            raise UnsupportedQuantError(
                f"{QUANT_DESCRIPTION_FILENAME} must be a JSON object of module path -> scheme; "
                f"got {type(quant_description).__name__}"
            )
        quant_description_sha256 = sha256_of_file(str(quant_path))

    warnings = [
        f"selective source: {len(needed_shards)}/{len(all_shards)} shards present; "
        f"{len(missing_dropped)} dropped-only shards absent (not needed)"
    ]
    if missing_dropped:
        warnings.append(f"absent dropped-only shards: {missing_dropped}")
    special = {QUANT_DESCRIPTION_FILENAME, MANIFEST_FILENAME, "config.json", *INDEX_FILENAMES}
    aux_files = []
    symlinked = 0
    for path in sorted(src.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        rel = path.relative_to(src).as_posix()
        if rel in special or (path.parent == src and path.suffix == ".safetensors"):
            continue
        _resolve_aux(path, src)
        if path.is_symlink():
            symlinked += 1
        aux_files.append(rel)
    if symlinked:
        warnings.append(f"{symlinked} auxiliary files were symlinks; copied dereferenced read-only content")
    _validate_referenced_aux(src, quant_description)
    if not any(name in aux_files for name in _TOKENIZER_CANDIDATES):
        raise UnsupportedFormatError(
            f"source checkpoint {src_dir!r} has none of {_TOKENIZER_CANDIDATES}; a reduced checkpoint must "
            "remain standalone-servable, so the tokenizer/processor files are required"
        )
    return SourceCheckpoint(
        src_dir=str(src),
        config=config,
        config_sha256=sha256_of_file(str(config_path)),
        index_sha256=index_sha256,
        shards=shards,
        weight_map=weight_map,
        aux_files=aux_files,
        quant_description=quant_description,
        quant_description_sha256=quant_description_sha256,
        warnings=warnings,
        selective={"missing_shards": missing_dropped, "present_shards": sorted(needed_shards)},
    )


def _validate_referenced_aux(src: Path, quant_description: dict | None) -> None:
    """Auxiliary safetensors referenced by quant metadata (e.g. QuaRot
    rotation maps) must exist; a dangling reference means a corrupt output."""

    def _walk(value):
        if isinstance(value, str) and value.endswith(".safetensors"):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from _walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from _walk(child)

    if not quant_description:
        return
    for ref in _walk(quant_description.get("optional")):
        if not (src / ref).is_file():
            raise UnsupportedQuantError(
                f"quant metadata references auxiliary file {ref!r} which is missing from the source; "
                "the reduced checkpoint would be unusable"
            )


def _layer_config_of(profile: ReductionProfile, config: dict) -> dict:
    if profile.config_layout == "nested_text":
        text_config = config.get("text_config")
        if not isinstance(text_config, dict):
            raise UnsupportedFormatError(
                f"profile {profile.name!r} expects a nested text_config object, but config.json lacks one; "
                "refusing to guess a flat override (see GLM-5.3-Flash layout)"
            )
        return text_config
    return config


def _classify(
    name: str,
    naming: NamingRules,
    source_layers: int,
    keep_layers: int,
    mtp_layers: int,
    keep_mtp: bool,
) -> TensorAction:
    # Vision subtrees first: their inner numbering must never be read as
    # decoder-layer indices.
    if any(name.startswith(prefix) for prefix in naming.vision_roots):
        return TensorAction(name, "keep", name)
    for root in naming.layer_roots:
        if not name.startswith(root):
            continue
        rest = name[len(root) :]
        match = re.match(r"(\d+)\.(.*)$", rest)
        if not match:
            raise UnknownTensorError(
                f"tensor {name!r} starts with the decoder-layer root {root!r} but has no numeric "
                "layer index; the checkpoint layout is not understood"
            )
        idx = int(match.group(1))
        if idx < source_layers:
            if idx < keep_layers:
                return TensorAction(name, "keep", name, src_layer=idx, dst_layer=idx)
            return TensorAction(name, "drop-layer", None, src_layer=idx)
        if idx < source_layers + mtp_layers:
            if not keep_mtp:
                return TensorAction(name, "drop-layer", None, src_layer=idx)
            dst_idx = keep_layers + (idx - source_layers)
            return TensorAction(name, "remap", f"{root}{dst_idx}.{match.group(2)}", src_layer=idx, dst_layer=dst_idx)
        raise UnknownTensorError(
            f"tensor {name!r} references layer {idx}, beyond {source_layers} decoder layers + "
            f"{mtp_layers} MTP layers; the checkpoint layout is not understood"
        )
    if name in naming.embed_names or name in naming.final_norm_names or name in naming.lm_head_names:
        return TensorAction(name, "keep", name)
    if name in naming.keep_names:
        return TensorAction(name, "keep", name)
    if _EXTRA_GLOBAL_RE.search(name) or name.startswith("rot.") or name.endswith("rotary_emb.inv_freq"):
        return TensorAction(name, "keep", name)
    raise UnknownTensorError(
        f"tensor {name!r} matches none of profile's known layer/global/vision names; re-run with "
        "--allow-extra-tensors to copy it verbatim (recorded in the manifest), or extend the "
        "profile naming rules in tools/glm_reduced/profiles.py after inspecting its role"
    )


_LAYER_ANCHOR_RE = re.compile(r"\.layers\.(\d+)\.")


def _remap_module_listing(entries: list, source_layers: int, keep_layers: int, mtp_layers: int, what: str) -> list:
    """Filter/remap per-layer module names in quantization listings.

    Entries without a ``.layers.<i>.`` anchor (scheme suffixes like ``lm_head``
    or vision paths) pass through; layer entries for dropped layers are removed
    and MTP entries are remapped onto the new layer indices.
    """
    result = []
    for entry in entries:
        if not isinstance(entry, str):
            raise UnsupportedQuantError(f"{what} entry {entry!r} is not a string; unsupported listing format")
        match = _LAYER_ANCHOR_RE.search(entry)
        if not match:
            result.append(entry)
            continue
        idx = int(match.group(1))
        if idx < source_layers:
            if idx < keep_layers:
                result.append(entry)
            continue
        if idx < source_layers + mtp_layers:
            new_idx = keep_layers + (idx - source_layers)
            result.append(entry[: match.start(1)] + str(new_idx) + entry[match.end(1) :])
            continue
        raise UnsupportedQuantError(
            f"{what} entry {entry!r} references layer {idx} beyond {source_layers}+{mtp_layers}; "
            "refusing to emit inconsistent quantization metadata"
        )
    return result


def _plan_config_quant(config: dict, source_layers: int, keep_layers: int, mtp_layers: int) -> None:
    quant = config.get("quantization_config")
    if quant is None:
        return
    if not isinstance(quant, dict):
        raise UnsupportedQuantError(f"quantization_config must be a dict, got {type(quant).__name__}")
    method = quant.get("quant_method")
    if method not in _SUPPORTED_CONFIG_QUANT:
        raise UnsupportedQuantError(
            f"quantization_config.quant_method={method!r} is not supported for layer reduction "
            f"(supported: {sorted(_SUPPORTED_CONFIG_QUANT)}). Per-tensor scale layouts such as FP8 "
            "blockwise survive a crop unchanged; schemes with cross-layer or packed state do not. "
            "Convert or pre-process the checkpoint first, or extend _plan_config_quant after review."
        )
    not_convert = quant.get("modules_to_not_convert")
    if isinstance(not_convert, list):
        quant["modules_to_not_convert"] = _remap_module_listing(
            not_convert, source_layers, keep_layers, mtp_layers, "modules_to_not_convert"
        )
    ignored = quant.get("ignored_layers")
    if isinstance(ignored, list):
        quant["ignored_layers"] = _remap_module_listing(
            ignored, source_layers, keep_layers, mtp_layers, "ignored_layers"
        )


def _plan_quant_description(
    source: SourceCheckpoint, source_layers: int, keep_layers: int, mtp_layers: int
) -> dict | None:
    """Filter/remap a ModelSlim quant_model_description.json atomically.

    Each key is classified on its own: per-layer module keys are kept, dropped
    or remapped individually; validated global metadata keys (group_size,
    metadata, optional, version, is_rot_used) are preserved untouched. Values
    are never re-paired with other keys.
    """
    description = source.quant_description
    if description is None:
        return None
    new_description = {}
    for key, value in description.items():
        if not isinstance(key, str):
            raise UnsupportedQuantError(f"{QUANT_DESCRIPTION_FILENAME} has a non-string key {key!r}")
        if key in _QUANT_GLOBAL_KEYS:
            if not isinstance(value, (str, int, float, bool, dict, list)) and value is not None:
                raise UnsupportedQuantError(
                    f"{QUANT_DESCRIPTION_FILENAME} global key {key!r} has unsupported value type {type(value).__name__}"
                )
            new_description[key] = value
            continue
        if not isinstance(value, str):
            raise UnsupportedQuantError(
                f"{QUANT_DESCRIPTION_FILENAME} entry {key!r} has unsupported value type "
                f"{type(value).__name__}; per-module entries must map to a scheme string, and only the "
                f"known global keys {sorted(_QUANT_GLOBAL_KEYS)} may carry metadata values"
            )
        match = _LAYER_ANCHOR_RE.search(key)
        if match:
            idx = int(match.group(1))
            if idx < source_layers:
                if idx < keep_layers:
                    new_description[key] = value
                # dropped decoder layer: key omitted
            elif idx < source_layers + mtp_layers:
                new_idx = keep_layers + (idx - source_layers)
                new_description[key[: match.start(1)] + str(new_idx) + key[match.end(1) :]] = value
            else:
                raise UnsupportedQuantError(
                    f"{QUANT_DESCRIPTION_FILENAME} entry {key!r} references layer {idx} beyond "
                    f"{source_layers}+{mtp_layers}; refusing to emit inconsistent quantization metadata"
                )
        else:
            new_description[key] = value
    return new_description


def _check_completeness(plan: Plan, weight_map: dict[str, str]) -> None:
    """Fail the build if any retained decoder/MTP layer or required global is missing."""
    kept_layers = {a.src_layer for a in plan.kept if a.src_layer is not None and a.action == "keep"}
    missing = [i for i in range(plan.keep_layers) if i not in kept_layers]
    if missing:
        raise UnsupportedFormatError(
            f"source checkpoint is missing all tensors for retained decoder layers {missing}; "
            "the checkpoint is incomplete"
        )
    if plan.mtp_layers and plan.profile.keep_mtp:
        remapped = {a.src_layer for a in plan.kept if a.action == "remap"}
        missing_mtp = [i for i in range(plan.source_layers, plan.source_layers + plan.mtp_layers) if i not in remapped]
        if missing_mtp:
            raise UnsupportedFormatError(
                f"config declares num_nextn_predict_layers={plan.mtp_layers} but MTP layers {missing_mtp} "
                "have no tensors in the checkpoint"
            )
    naming = plan.profile.naming
    kept_names = {a.dst_name for a in plan.kept}
    missing_globals = []
    if not any(name in kept_names for name in naming.embed_names):
        missing_globals.append(f"embedding ({'|'.join(naming.embed_names)})")
    if not any(name in kept_names for name in naming.final_norm_names):
        missing_globals.append(f"final norm ({'|'.join(naming.final_norm_names)})")
    tied = bool(_layer_config_of(plan.profile, plan.new_config).get("tie_word_embeddings", False))
    if not tied and not any(name in kept_names for name in naming.lm_head_names):
        missing_globals.append(f"lm_head ({'|'.join(naming.lm_head_names)}, tie_word_embeddings=false)")
    if missing_globals:
        raise UnsupportedFormatError(
            "source checkpoint is missing required global tensors: " + ", ".join(missing_globals)
        )


def plan_reduction(
    source: SourceCheckpoint,
    profile: ReductionProfile,
    keep_layers: int | None = None,
    *,
    allow_extra_tensors: bool = False,
    truncate_unknown_arrays: bool = False,
) -> Plan:
    config = source.config
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise UnsupportedFormatError("config.json has no architectures list; cannot identify the model family")
    if not set(architectures).issubset(set(profile.architectures)):
        raise ReductionError(
            f"checkpoint architectures {architectures} are not covered by profile {profile.name!r} "
            f"({list(profile.architectures)}); pick the matching profile or add a new one"
        )
    layer_config = _layer_config_of(profile, config)
    source_layers = read_layer_count(profile, layer_config)
    keep = keep_layers if keep_layers is not None else profile.default_keep_layers
    profile.validate_keep_layers(keep, source_layers)
    problems = validate_layer_arrays(profile, layer_config, keep, source_layers)
    if problems:
        raise ReductionError(
            f"profile {profile.name!r} rejected the source config for keep_layers={keep}:\n- " + "\n- ".join(problems)
        )

    mtp_layers = int(layer_config.get("num_nextn_predict_layers") or 0)
    warnings: list[str] = list(source.warnings)
    if profile.official_layers and profile.official_layers != source_layers:
        warnings.append(
            f"source layer count={source_layers} differs from the official {profile.official_layers} "
            f"for profile {profile.name!r}; assuming a custom/pre-reduced checkpoint"
        )
    # Unknown config arrays whose length equals the layer count are almost
    # certainly per-layer and would silently describe dropped layers.
    for key, value in layer_config.items():
        if (
            isinstance(value, list)
            and len(value) == source_layers
            and key not in PER_LAYER_ARRAY_KEYS
            and key not in profile.layer_count_keys
        ):
            message = (
                f"config key {key!r} is a list of length num_hidden_layers but is not a known per-layer "
                "array; it was left unchanged"
            )
            if not truncate_unknown_arrays:
                raise ReductionError(
                    message + "; re-run with --truncate-unknown-arrays after confirming it is per-layer"
                )
            warnings.append(message + " (truncated by --truncate-unknown-arrays)")

    actions: list[TensorAction] = []
    unknown: list[str] = []
    for name in sorted(source.weight_map):
        try:
            actions.append(_classify(name, profile.naming, source_layers, keep, mtp_layers, profile.keep_mtp))
        except UnknownTensorError:
            if not allow_extra_tensors:
                raise
            unknown.append(name)
            actions.append(TensorAction(name, "keep", name))
    if unknown:
        warnings.append(f"{len(unknown)} tensors matched no known pattern and were copied verbatim: {unknown}")

    new_config = json.loads(json.dumps(config))  # deep copy; source config never mutated
    target = new_config if profile.config_layout == "flat" else new_config["text_config"]
    truncated = truncate_layer_config(profile, layer_config, keep, source_layers)
    if truncate_unknown_arrays:
        for key, value in layer_config.items():
            if isinstance(value, list) and len(value) == source_layers and key not in PER_LAYER_ARRAY_KEYS:
                truncated[key] = value[:keep]
    target.clear()
    target.update(truncated)
    if not profile.keep_mtp and mtp_layers:
        target["num_nextn_predict_layers"] = 0
    _plan_config_quant(new_config, source_layers, keep, mtp_layers)
    new_quant_description = _plan_quant_description(source, source_layers, keep, mtp_layers)

    plan = Plan(
        profile=profile,
        keep_layers=keep,
        source_layers=source_layers,
        mtp_layers=mtp_layers,
        actions=actions,
        new_config=new_config,
        new_quant_description=new_quant_description,
        warnings=warnings,
    )
    _check_completeness(plan, source.weight_map)
    return plan


def _check_paths(src_dir: str, dst_dir: str) -> tuple[Path, Path]:
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()
    if src == dst:
        raise SafetyError("output directory equals the source directory; refusing to overwrite the input")
    if dst in src.parents:
        raise SafetyError(f"output {dst} is a parent of the source {src}; refusing an unsafe output path")
    if src in dst.parents:
        raise SafetyError(f"output {dst} is inside the source {src}; refusing to nest output in the input")
    if dst.exists():
        raise SafetyError(f"output directory {dst} already exists; refusing to overwrite it")
    if not dst.parent.is_dir():
        raise SafetyError(f"parent of output directory {dst} does not exist")
    return src, dst


def execute_plan(
    plan: Plan,
    source: SourceCheckpoint,
    dst_dir: str,
    *,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict:
    """Write the reduced checkpoint and manifest; returns the manifest dict."""
    src, dst = _check_paths(source.src_dir, dst_dir)
    staging = dst.with_name(f"{dst.name}.glm-reduced-staging-{uuid.uuid4().hex[:12]}")
    staging.mkdir()
    try:
        out_tensors = []
        for action in plan.kept:
            shard_name = source.weight_map[action.src_name]
            shard = source.shards[shard_name]
            info = shard.tensors[action.src_name]
            out_tensors.append(
                OutTensor(
                    name=action.dst_name or action.src_name,
                    dtype=info.dtype,
                    shape=info.shape,
                    nbytes=info.nbytes,
                    src=shard,
                    src_info=info,
                )
            )
        shards_plan = plan_shards(out_tensors, max_shard_bytes)
        weight_map: dict[str, str] = {}
        total_size = 0
        tensor_digests: dict[str, str] = {}
        output_files: list[dict] = []
        for shard_idx, shard_tensors in enumerate(shards_plan):
            shard_name = shard_filename(shard_idx, len(shards_plan))
            shard_path = staging / shard_name
            shard_bytes, digests = write_shard(str(shard_path), shard_tensors)
            total_size += shard_bytes
            tensor_digests.update(digests)
            for tensor in shard_tensors:
                weight_map[tensor.name] = shard_name
            output_files.append({"name": shard_name, "bytes": shard_bytes, "sha256": sha256_of_file(str(shard_path))})
        if len(shards_plan) > 1:
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            (staging / INDEX_FILENAMES[0]).write_text(json.dumps(index, indent=2), encoding="utf-8")
            index_path = staging / INDEX_FILENAMES[0]
            output_files.append(
                {
                    "name": INDEX_FILENAMES[0],
                    "bytes": index_path.stat().st_size,
                    "sha256": sha256_of_file(str(index_path)),
                }
            )

        (staging / "config.json").write_text(json.dumps(plan.new_config, indent=2), encoding="utf-8")
        output_files.append(
            {
                "name": "config.json",
                "bytes": (staging / "config.json").stat().st_size,
                "sha256": sha256_of_file(str(staging / "config.json")),
            }
        )
        if plan.new_quant_description is not None:
            (staging / QUANT_DESCRIPTION_FILENAME).write_text(
                json.dumps(plan.new_quant_description, indent=2), encoding="utf-8"
            )
            output_files.append(
                {
                    "name": QUANT_DESCRIPTION_FILENAME,
                    "bytes": (staging / QUANT_DESCRIPTION_FILENAME).stat().st_size,
                    "sha256": sha256_of_file(str(staging / QUANT_DESCRIPTION_FILENAME)),
                }
            )
        for rel in source.aux_files:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_resolve_aux(src / rel, src), target)
            output_files.append({"name": rel, "bytes": target.stat().st_size, "sha256": sha256_of_file(str(target))})

        manifest = _build_manifest(plan, source, output_files, total_size, tensor_digests, max_shard_bytes)
        (staging / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.rename(staging, dst)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _build_manifest(
    plan: Plan,
    source: SourceCheckpoint,
    output_files: list[dict],
    total_size: int,
    tensor_digests: dict[str, str],
    max_shard_bytes: int,
) -> dict:
    layer_pairs = sorted({(a.src_layer, a.dst_layer) for a in plan.kept if a.src_layer is not None})
    return {
        "schema": MANIFEST_SCHEMA,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": source.src_dir,
            "config_sha256": source.config_sha256,
            "index_sha256": source.index_sha256,
            "quant_description_sha256": source.quant_description_sha256,
            "weight_files": [
                {"name": name, "bytes": os.path.getsize(source.shards[name].path)} for name in sorted(source.shards)
            ],
            "architectures": source.config.get("architectures"),
            "num_hidden_layers": plan.source_layers,
            "selective": source.selective,
        },
        "reduction": {
            "profile": plan.profile.name,
            "keep_layers": plan.keep_layers,
            "keep_mtp": plan.profile.keep_mtp,
            "mtp_layers": plan.mtp_layers,
            "include_vision": plan.profile.include_vision,
            "layer_mapping": [{"src": s, "dst": d} for s, d in layer_pairs],
            "dropped_decoder_layers": sorted(
                {a.src_layer for a in plan.dropped if a.action == "drop-layer" and a.src_layer is not None}
            ),
            "max_shard_bytes": max_shard_bytes,
        },
        "quantization": {
            "config_quant_method": (plan.new_config.get("quantization_config") or {}).get("quant_method"),
            "has_quant_description": plan.new_quant_description is not None,
        },
        "output": {
            "total_size": total_size,
            "tensor_count": len(plan.kept),
            "dropped_tensor_count": len(plan.dropped),
            "files": output_files,
            "tensor_sha256": tensor_digests,
        },
        "warnings": plan.warnings,
    }


def manifest_checkpoint_id(out_dir: str) -> str:
    """Stable identity of a checkpoint produced by this tool: manifest digest."""
    manifest_path = Path(out_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ManifestError(f"{out_dir!r} has no {MANIFEST_FILENAME}; pass an explicit checkpoint identity instead")
    return sha256_of_file(str(manifest_path))


def verify_reduced(out_dir: str) -> dict:
    """Re-check a reduced checkpoint against its manifest; returns a report."""
    out = Path(out_dir)
    manifest_path = out / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ManifestError(f"{out_dir!r} has no {MANIFEST_FILENAME}; nothing to verify against")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"unsupported manifest schema {manifest.get('schema')!r}")
    problems: list[str] = []

    recorded = {entry["name"]: entry for entry in manifest["output"]["files"]}
    for name, entry in recorded.items():
        path = out / name
        if not path.is_file():
            problems.append(f"missing output file {name!r}")
            continue
        if sha256_of_file(str(path)) != entry["sha256"]:
            problems.append(f"checksum mismatch for {name!r}")
    extra = {
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p.name != MANIFEST_FILENAME and str(p.relative_to(out).as_posix()) not in recorded
    }
    if extra:
        problems.append(f"unexpected files in output: {sorted(extra)}")

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    keep = manifest["reduction"]["keep_layers"]
    layer_config = config.get("text_config", config)
    layer_counts = [
        value for key in ("num_hidden_layers", "num_layers") if (value := layer_config.get(key)) is not None
    ]
    if not layer_counts or any(value != keep for value in layer_counts):
        problems.append(f"config layer counts {layer_counts} disagree with manifest keep_layers={keep}")

    weight_files = [name for name in recorded if name.endswith(".safetensors") and "/" not in name]
    tensor_names: list[str] = []
    total_size = 0
    for name in weight_files:
        parsed = read_safetensors(str(out / name))
        tensor_names.extend(parsed.tensors)
        total_size += sum(t.nbytes for t in parsed.tensors.values())
    if len(weight_files) > 1:
        index = json.loads((out / INDEX_FILENAMES[0]).read_text(encoding="utf-8"))
        if index.get("metadata", {}).get("total_size") != total_size:
            problems.append("index total_size disagrees with actual tensor bytes")
        if set(index.get("weight_map", {})) != set(tensor_names):
            problems.append("index weight_map disagrees with shard contents")
    expected_tensors = set(manifest["output"]["tensor_sha256"])
    if set(tensor_names) != expected_tensors:
        problems.append("tensor inventory disagrees with the manifest")

    # Decoder-layer completeness: every retained layer must have tensors.
    layer_root_re = re.compile(r"layers\.(\d+)\.")
    covered = {int(m.group(1)) for name in tensor_names if (m := layer_root_re.search(name))}
    missing_layers = [i for i in range(keep) if i not in covered]
    if missing_layers:
        problems.append(f"output is missing all tensors for decoder layers {missing_layers}")
    mtp_layers = manifest["reduction"].get("mtp_layers", 0)
    if manifest["reduction"].get("keep_mtp") and mtp_layers:
        missing_mtp = [keep + i for i in range(mtp_layers) if keep + i not in covered]
        if missing_mtp:
            problems.append(f"output is missing remapped MTP layers {missing_mtp}")

    naming_roots = [name for name in tensor_names]
    has_embed = any(
        name.endswith("embed_tokens.weight") or name.endswith("word_embeddings.weight") for name in naming_roots
    )
    has_norm = any(
        re.search(r"(^|\.)model\.norm\.weight$", name)
        or name.endswith("language_model.norm.weight")
        or name.endswith("final_layernorm.weight")
        for name in naming_roots
    )
    has_lm_head = any(
        name.startswith("lm_head.") or name.endswith(".lm_head.weight") or name.endswith("output_layer.weight")
        for name in naming_roots
    )
    tied = bool(layer_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    if not has_embed:
        problems.append("output is missing the embedding tensor")
    if not has_norm:
        problems.append("output is missing the final norm tensor")
    if not has_lm_head and not tied:
        problems.append("output has no lm_head although tie_word_embeddings is false")

    return {
        "output_dir": str(out),
        "profile": manifest["reduction"]["profile"],
        "keep_layers": keep,
        "tensor_count": len(tensor_names),
        "tie_word_embeddings": tied,
        "ok": not problems,
        "problems": problems,
    }


def required_shards(config_path: str, index_path: str, profile: ReductionProfile, keep_layers: int) -> dict:
    """Selective-download support: given only the small config.json and index
    files of a public checkpoint, list exactly which shard files a reduction
    needs (plus their sizes from the index metadata when available)."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise UnsupportedFormatError(f"{index_path} has no usable weight_map")
    layer_config = _layer_config_of(profile, config)
    source_layers = read_layer_count(profile, layer_config)
    profile.validate_keep_layers(keep_layers, source_layers)
    problems = validate_layer_arrays(profile, layer_config, keep_layers, source_layers)
    if problems:
        raise ReductionError("config rejected:\n- " + "\n- ".join(problems))
    mtp_layers = int(layer_config.get("num_nextn_predict_layers") or 0)
    needed = set()
    for name, shard_name in weight_map.items():
        action = _classify(name, profile.naming, source_layers, keep_layers, mtp_layers, profile.keep_mtp)
        if action.action in ("keep", "remap"):
            needed.add(shard_name)
    return {
        "shards": sorted(needed),
        "shard_count": len(needed),
        "total_shards": len(set(weight_map.values())),
        "keep_layers": keep_layers,
        "source_layers": source_layers,
    }
