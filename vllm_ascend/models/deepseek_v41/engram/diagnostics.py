# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup diagnostics for the installed Engram runtime, without device sync."""

import inspect
import json
import logging
import platform
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

import torch
from vllm.logger import logger


def _package_info(name: str) -> dict[str, str]:
    module = sys.modules.get(name)
    try:
        version = metadata.version(name.replace("_", "-"))
    except Exception:
        version = str(getattr(module, "__version__", "unknown"))
    return {"version": version, "file": str(getattr(module, "__file__", "not-loaded"))}


def _loaded_libraries() -> list[str]:
    # Linux maps reflects the libraries actually loaded, including custom OPP
    # libraries. It is more useful than inferring an ACLNN version from a wheel.
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return ["unavailable: /proc/self/maps is not readable"]
    paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[5]
        if (
            path.startswith("/")
            and ".so" in path
            and any(name in path.lower() for name in ("opapi", "optiling", "ascendcl", "engram", "elastic_buffer"))
        ):
            paths.add(path)
    return sorted(paths)


def _tensor_info(tensor: object) -> dict:
    """Read descriptors only: no tensor values, device copies, or stream waits."""
    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__}
    try:
        details = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "contiguous": tensor.is_contiguous(),
        }
        if tensor.device.type == "npu":
            get_format = getattr(sys.modules.get("torch_npu"), "get_npu_format", None)
            if get_format is not None:
                try:
                    details["npu_format"] = get_format(tensor)
                except Exception as exc:
                    details["npu_format_error"] = str(exc)
        return details
    except Exception as exc:
        return {"metadata_error": str(exc)}


def _schemas() -> dict[str, str]:
    result = {}
    for name in ("engram_fetch", "engram_fetch_wait"):
        try:
            result[name] = str(getattr(torch.ops.cann_ops_transformer, name).default._schema)
        except Exception as exc:
            result[name] = f"unavailable: {exc}"
    return result


def log_engram_runtime(
    layer_id: int | None,
    rank: int,
    *,
    stage: str,
    buffer=None,
    failure: bool = False,
    tensors: Mapping[str, object] | None = None,
    parameters: Mapping[str, object] | None = None,
) -> None:
    """Best-effort provenance; missing metadata never changes inference behavior."""
    level = logging.ERROR if failure else logging.DEBUG
    if not logger.isEnabledFor(level):
        return
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
            "schemas": _schemas(),
        }
        if failure:
            details["native_call"] = {
                "stage": "GetWorkspaceSize/execute not distinguished by the Python wrapper",
                "function_providers": "not determined by library mappings; requires native dladdr",
                "workspace_size_and_executor": "requires native call instrumentation",
            }
        if tensors is not None:
            details["tensors"] = {name: _tensor_info(tensor) for name, tensor in tensors.items()}
        if parameters is not None:
            details["parameters"] = dict(parameters)
        npu = getattr(torch, "npu", None)
        try:
            if npu is not None and npu.is_available():
                index = npu.current_device()
                details["device"] = {"index": index, "name": npu.get_device_name(index)}
                torch_npu = sys.modules.get("torch_npu")
                if torch_npu is not None:
                    details["device"]["soc_version"] = torch_npu.npu.get_soc_version()
                    details["cann"] = str(getattr(getattr(torch_npu, "version", None), "cann", details["cann"]))
        except Exception as exc:
            details["device_metadata_error"] = str(exc)
        if buffer is not None:
            try:
                details["write_signature"] = str(inspect.signature(type(buffer).engram_write))
            except Exception as exc:
                details["write_signature_error"] = str(exc)
            try:
                runtime = getattr(buffer, "_runtime", None)
                extension = sys.modules.get(type(runtime).__module__)
                details["extension_file"] = str(getattr(extension, "__file__", "unknown"))
            except Exception as exc:
                details["extension_metadata_error"] = str(exc)
        logger.log(
            level,
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
