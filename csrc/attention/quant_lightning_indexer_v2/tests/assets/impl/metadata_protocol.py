#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private five-operator bridge for TTK manual-data prepare and replay.

The protocol is intentionally environment- and file-based.  It does not import
TTK or depend on a runner/context object.  TTK continues to own its case files;
the assets only keep a compact sidecar outside each case directory and replace
an existing metadata input atomically after a zero-placeholder replay.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import regex as re
import torch

ENV_NAME = "MANUAL_DATA_DIRS"
PROTOCOL_VERSION = 1
SIDECAR_DIRECTORY = ".ttk_asset_metadata"
_SAFE_CASE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_OPERATOR = re.compile(r"[A-Za-z0-9_.-]+")
_FORMATS = ("bin", "npy", "pt")


def manual_data_roots():
    """Resolve the optional private roots from one shell environment value."""
    value = os.getenv(ENV_NAME)
    if not value:
        return ()
    roots = []
    seen = set()
    for item in value.split(os.pathsep):
        if not item.strip():
            continue
        root = Path(item).expanduser().resolve()
        if root not in seen:
            roots.append(root)
            seen.add(root)
    return tuple(roots)


def case_directory_name(testcase_name):
    """Mirror TTK's stable case-directory rule without importing TTK."""
    name = str(testcase_name)
    safe = _SAFE_CASE_NAME.sub("_", name).strip("._") or "case"
    if safe == name and len(safe) <= 120:
        return safe
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:96]}-{digest}"


def validate_operator(operator):
    if _SAFE_OPERATOR.fullmatch(str(operator)) is None:
        raise ValueError(f"invalid metadata protocol operator: {operator!r}")


def clone_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, copy=True))
    if isinstance(value, dict):
        return {name: clone_to_cpu(item) for name, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported metadata sidecar value: {type(value).__name__}")


def build_sidecar_path(root, operator, testcase_name):
    validate_operator(operator)
    return root / SIDECAR_DIRECTORY / str(operator) / f"{case_directory_name(testcase_name)}.pt"


def save_metadata_inputs(operator, testcase_name, metadata_input):
    """Atomically save exact CPU metadata arguments during input preparation."""
    roots = manual_data_roots()
    if not roots or testcase_name is None:
        return None
    if not isinstance(metadata_input, dict):
        raise ValueError(f"{operator} pytest data lacks metadata_input")

    path = build_sidecar_path(roots[0], operator, testcase_name)
    if path.parent.is_symlink():
        raise ValueError(f"metadata sidecar directory must not be a symlink: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PROTOCOL_VERSION,
        "operator": str(operator),
        "testcase_name": str(testcase_name),
        "metadata_input": clone_to_cpu(metadata_input),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_metadata_inputs(operator, testcase_name):
    """Load the first matching sidecar, or return None for fallback derivation."""
    if testcase_name is None:
        return None
    for root in manual_data_roots():
        path = build_sidecar_path(root, operator, testcase_name)
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"metadata sidecar must be a regular file: {path}")
        payload = load_pt(path)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != PROTOCOL_VERSION
            or payload.get("operator") != str(operator)
            or payload.get("testcase_name") != str(testcase_name)
            or not isinstance(payload.get("metadata_input"), dict)
        ):
            raise ValueError(f"incompatible metadata sidecar: {path}")
        return payload["metadata_input"]
    return None


def metadata_is_materialized(metadata):
    """Treat a nonzero metadata slot as authoritative in every execution mode."""
    if metadata is None:
        return False
    if torch.is_tensor(metadata):
        return bool(torch.count_nonzero(metadata).item())
    return bool(np.count_nonzero(np.asarray(metadata)))


def metadata_to_array(metadata):
    if torch.is_tensor(metadata):
        return metadata.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(np.asarray(metadata))


def find_metadata_file(root, testcase_name, metadata_index):
    case_dir = root / case_directory_name(testcase_name)
    if not case_dir.exists():
        return None
    if case_dir.is_symlink() or not case_dir.is_dir():
        raise ValueError(f"manual-data testcase path must be a regular directory: {case_dir}")
    for file_format in _FORMATS:
        matches = tuple(case_dir.glob(f"input_{int(metadata_index)}_*.{file_format}"))
        if len(matches) > 1:
            raise RuntimeError(f"expected one metadata input[{metadata_index}], found {len(matches)} in {case_dir}")
        if matches:
            path = matches[0]
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"metadata input must be a regular file: {path}")
            return path
    return None


def write_array(path, array):
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        if path.suffix == ".bin":
            array.tofile(temporary)
        elif path.suffix == ".npy":
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
        elif path.suffix == ".pt":
            torch.save(torch.from_numpy(np.array(array, copy=True)), temporary)
        else:
            raise ValueError(f"unsupported metadata input format: {path.suffix}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rewrite_metadata_input(operator, testcase_name, metadata_index, metadata):
    """Replace the existing zero placeholder after successful metadata execution."""
    validate_operator(operator)
    if testcase_name is None:
        return None
    for root in manual_data_roots():
        path = find_metadata_file(root, testcase_name, metadata_index)
        if path is None:
            continue
        write_array(path, metadata_to_array(metadata))
        return path
    return None
