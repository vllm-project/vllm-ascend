# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup diagnostics for the installed Engram runtime, without device sync."""

import inspect
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch
from vllm.logger import logger


def _package_info(name: str) -> dict[str, str]:
    module = sys.modules.get(name)
    try:
        version = metadata.version(name.replace("_", "-"))
    except metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", "unknown"))
    return {"version": version, "file": str(getattr(module, "__file__", "not-loaded"))}


def _loaded_libraries() -> list[str]:
    # Linux maps reflects the libraries actually loaded, including custom OPP
    # libraries. It is more useful than inferring an ACLNN version from a wheel.
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return ["unavailable: /proc/self/maps is not readable"]
    return sorted(
        {
            fields[-1]
            for line in lines
            if (fields := line.split())
            and fields[-1].startswith("/")
            and ".so" in fields[-1]
            and any(
                name in fields[-1].lower() for name in ("opapi", "optiling", "ascendcl", "engram", "elastic_buffer")
            )
        }
    )


def log_engram_runtime(layer_id: int | None, rank: int, *, stage: str, buffer=None) -> None:
    """Best-effort provenance; missing metadata never changes inference behavior."""
    try:
        details = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _package_info("torch"),
            "torch_npu": _package_info("torch_npu"),
            "cann_ops_transformer": _package_info("cann_ops_transformer"),
            "cann": str(getattr(torch.version, "cann", "unknown")),
            "opp_aclnn_version": "not exposed; verify loaded libraries and startup probe",
            "loaded_libraries": _loaded_libraries(),
        }
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            index = npu.current_device()
            details["device"] = {"index": index, "name": npu.get_device_name(index)}
            torch_npu = sys.modules.get("torch_npu")
            if torch_npu is not None:
                details["device"]["soc_version"] = torch_npu.npu.get_soc_version()
                details["cann"] = str(getattr(getattr(torch_npu, "version", None), "cann", details["cann"]))
        if buffer is not None:
            details["write_signature"] = str(inspect.signature(type(buffer).engram_write))
            runtime = getattr(buffer, "_runtime", None)
            extension = sys.modules.get(type(runtime).__module__)
            details["extension_file"] = str(getattr(extension, "__file__", "unknown"))
        logger.debug(
            "FOR-ENGRAM runtime: layer=%s tp_rank=%d stage=%s details=%s",
            layer_id,
            rank,
            stage,
            json.dumps(details, default=str, sort_keys=True),
        )
    except Exception as exc:
        logger.warning(
            "FOR-ENGRAM runtime diagnostics unavailable: layer=%s tp_rank=%d stage=%s reason=%s",
            layer_id,
            rank,
            stage,
            exc,
        )
