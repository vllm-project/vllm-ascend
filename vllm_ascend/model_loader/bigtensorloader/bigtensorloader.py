# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import mmap
import os
import threading
import time
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend import envs

from .bigtensordefaultloader import BigTensorDefaultLoader


@register_model_loader("bigtensorloader")
class BigTensorLoader(BigTensorDefaultLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    # dtype string <-> torch.dtype mapping
    _DTYPE_MAP: dict[str, torch.dtype] = {
        "F32": torch.float32,
        "F64": torch.float64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
        "F8_E8M0": torch.float8_e8m0fnu,  # MXFP8 block-scale dtype
    }

    _DTYPE_KEY: dict[torch.dtype, str] = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e5m2: "F8_E5M2",
        torch.float8_e8m0fnu: "F8_E8M0",
    }

    @staticmethod
    def _fingerprint(vllm_config: VllmConfig, model_config: ModelConfig) -> dict:
        """Config fingerprint to reject snapshot/model config mismatch."""
        pc = vllm_config.parallel_config
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        weight_nz_mode = additional_config.get("weight_nz_mode", 1)
        return {
            "model": model_config.model,
            "dtype": str(model_config.dtype).replace("torch.", ""),
            "quantization": model_config.quantization,
            "tp": pc.tensor_parallel_size,
            "pp": pc.pipeline_parallel_size,
            "dp": getattr(pc, "data_parallel_size", 1),
            "ep": pc.enable_expert_parallel,
            "world_size": torch.distributed.get_world_size(),
            "weight_nz_mode": weight_nz_mode,
        }

    def _resolve_param(
        self, name: str, param_dict: dict, name_to_module: dict, target_device: torch.device, dtype_str: str
    ):
        """Resolve or dynamically register a parameter for a snapshot entry."""
        param = param_dict.get(name)
        if param is not None:
            return param
        module_name, _, local_weight_name = name.rpartition(".")
        if not local_weight_name:
            raise RuntimeError(
                f"snapshot resolve: snapshot entry {name!r} has no module prefix "
                f"(expected 'module.weight' form); the snapshot is stale or "
                f"corrupt -- re-run convert."
            )
        layer = name_to_module.get(module_name)
        if layer is None:
            return None
        dtype = self._DTYPE_MAP.get(dtype_str, torch.float32)
        if hasattr(layer, local_weight_name):
            return getattr(layer, local_weight_name)
        # Attribute is missing -> register a placeholder.
        # requires_grad=False: non-float dtypes crash on default True.
        placeholder = nn.Parameter(torch.empty(0, dtype=dtype, device=target_device), requires_grad=False)
        layer.register_parameter(local_weight_name, placeholder)
        logger.debug("snapshot resolve: registered placeholder param %r on module %r", name, module_name)
        return placeholder

    def _snapshot_paths(self) -> tuple:
        rank = torch.distributed.get_rank()
        path = envs.VLLM_ASCEND_CHECKPOINT_PATH
        if not path:
            raise RuntimeError(
                "VLLM_ASCEND_BIGTENSOR_FAST_RESTORE=1 requires VLLM_ASCEND_CHECKPOINT_PATH "
                "to be set to a writable directory where the post-process "
                "snapshot is stored (e.g. VLLM_ASCEND_CHECKPOINT_PATH="
                "/path/to/snapshot_dir). Unset VLLM_ASCEND_BIGTENSOR_FAST_RESTORE to "
                "use the normal fresh-load path."
            )
        if os.path.exists(path) and not os.path.isdir(path):
            raise RuntimeError(
                f"VLLM_ASCEND_CHECKPOINT_PATH={path!r} is not a directory; "
                f"bigtensorloader needs a writable directory to store "
                f"snapshot files. Point it to a directory instead."
            )
        path = path.rstrip("/")
        return f"{path}/{rank}.json", f"{path}/{rank}.snapshot"

    def _has_snapshot(self) -> bool:
        if not envs.VLLM_ASCEND_CHECKPOINT_PATH:
            return False
        manifest, blob = self._snapshot_paths()
        return os.path.exists(manifest) and os.path.exists(blob)

    @staticmethod
    def _is_packed_nz(t: torch.Tensor) -> bool:
        """Check if t is a packed low-bit NZ weight (int8 NZ viewed as int32)."""
        return t.device.type == "npu" and t.dtype == torch.int32 and int(torch_npu.get_npu_format(t)) == 29

    def _append_tensor(self, t: torch.Tensor, blob_parts: list, offset: int) -> tuple[dict, int]:
        """Serialize one tensor into blob_parts (16B-aligned), return entry + new offset."""
        t = t.detach()
        fmt = int(torch_npu.get_npu_format(t)) if t.device.type == "npu" else 0
        save_dtype = save_shape = None
        d2h = t
        if self._is_packed_nz(t):
            save_dtype = "int8"
            save_shape = list(t.view(torch.int8).shape)
            d2h = t.view(torch.int8)
        raw = d2h.to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        pad = (-offset) % 16  # 16B-align every tensor start (elem_size <= 8)
        if pad:
            blob_parts.append(b"\x00" * pad)
            offset += pad
        entry = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "format": fmt,
            "offset": offset,
            "nbytes": len(raw),
        }
        if save_dtype is not None:
            entry["save_dtype"] = save_dtype
            entry["save_shape"] = save_shape
        blob_parts.append(raw)
        return entry, offset + len(raw)

    def save_snapshot(
        self,
        model: nn.Module,
        fingerprint: dict | None = None,
        removed_params: list | None = None,
        scalar_attrs: dict | None = None,
    ):
        """Serialize post-process weights to blob_parts + manifest.

        D2H runs synchronously (before graph capture); disk write is async.
        Captures params/buffers, tensor attrs, obj_attr tensors, null attrs,
        removed params, and scalar side effects. Model-agnostic: integrity
        is enforced by blob sha256 + descriptor self-consistency + fingerprint.
        """
        import hashlib

        rank = torch.distributed.get_rank()
        manifest_path, blob_path = self._snapshot_paths()
        entries: dict[str, dict] = {}
        blob_parts: list[bytes] = []
        offset = 0

        # (1) params/buffers in state_dict
        for name, t in model.state_dict().items():
            entry, offset = self._append_tensor(t, blob_parts, offset)
            entry["kind"] = "param"
            entries[name] = entry

        # (2) Plain tensor attrs in module __dict__ (invisible to state_dict)
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in list(mod.__dict__.items()):
                if isinstance(v, (list, tuple, dict)):
                    if attr.startswith("_"):
                        continue
                    vals = v.values() if isinstance(v, dict) else v
                    n_tensors = sum(1 for x in vals if isinstance(x, torch.Tensor) and x.numel() > 0)
                    if n_tensors:
                        logger.warning(
                            "snapshot save: %s.%s is a container holding %d tensor(s); "
                            "containers are NOT captured -- verify restore coverage",
                            mod_name or "<root>",
                            attr,
                            n_tensors,
                        )
                    continue
                if not isinstance(v, torch.Tensor):
                    continue
                if v.numel() == 0:
                    continue  # runtime placeholder, no data to persist
                key = f"{mod_name}.{attr}" if mod_name else attr
                if key in entries:
                    continue
                entry, offset = self._append_tensor(v, blob_parts, offset)
                entry.update({"kind": "attr", "device": v.device.type})
                entries[key] = entry

        # (3) Tensors inside non-Module attribute objects (e.g. MLA's impl)
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in list(mod.__dict__.items()):
                if isinstance(v, (torch.nn.Module, torch.Tensor, type)) or callable(v) or not hasattr(v, "__dict__"):
                    continue
                for sub, sv in list(vars(v).items()):
                    if isinstance(sv, (list, tuple, dict)):
                        vals = sv.values() if isinstance(sv, dict) else sv
                        n_tensors = sum(1 for x in vals if isinstance(x, torch.Tensor))
                        if n_tensors:
                            logger.warning(
                                "snapshot save: carrier %s.%s holds container attr '%s' "
                                "with %d tensor(s); containers are NOT captured",
                                mod_name or "<root>",
                                attr,
                                sub,
                                n_tensors,
                            )
                        continue
                    if not isinstance(sv, torch.Tensor):
                        continue
                    if sv.numel() == 0:
                        continue
                    entry, offset = self._append_tensor(sv, blob_parts, offset)
                    key = f"{mod_name}.{attr}.{sub}" if mod_name else f"{attr}.{sub}"
                    entry.update(
                        {
                            "kind": "obj_attr",
                            "carrier": f"{mod_name}.{attr}" if mod_name else attr,
                            "attr": sub,
                            "device": sv.device.type,
                        }
                    )
                    entries[key] = entry

        # (4) Attrs nulled by process (recorded for restore to replay)
        null_attrs = []
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in mod.__dict__.items():
                if v is None:
                    null_attrs.append(f"{mod_name}.{attr}" if mod_name else attr)

        h = hashlib.sha256()
        for raw in blob_parts:
            h.update(raw)
        manifest = {
            "schema_version": 2,
            "format_note": "Model-agnostic: integrity by sha256 + descriptor + fingerprint",
            "world_rank": rank,
            "world_size": torch.distributed.get_world_size(),
            "fingerprint": fingerprint,
            "sha256": h.hexdigest(),
            "tensors": entries,
            "null_attrs": null_attrs,
            "removed_params": removed_params or [],
            "scalar_attrs": scalar_attrs or {},
        }
        logger.info(
            "save_snapshot rank=%d tensors=%d bytes=%d sha256=%s (serialized, awaiting disk write)",
            rank,
            len(entries),
            offset,
            manifest["sha256"][:16],
        )
        return blob_parts, manifest

    def _write_snapshot_async(self, blob_parts: list, manifest: dict):
        """Background disk writer: blob + manifest, atomic (tmp+rename)."""
        try:
            manifest_path, blob_path = self._snapshot_paths()
            tmp_blob = blob_path + ".tmp"
            os.makedirs(os.path.dirname(blob_path) or ".", exist_ok=True)
            for stale in (tmp_blob, manifest_path + ".tmp"):
                if os.path.exists(stale):
                    logger.warning(
                        "snapshot snapshot: removing stale tmp file %s left by an interrupted convert", stale
                    )
                    os.remove(stale)
            with open(tmp_blob, "wb") as f:
                for raw in blob_parts:
                    f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_blob, blob_path)
            with open(manifest_path + ".tmp", "w") as f:
                json.dump(manifest, f)
            os.rename(manifest_path + ".tmp", manifest_path)
            logger.info(
                "save_snapshot disk write complete rank=%d tensors=%d sha256=%s",
                manifest["world_rank"],
                len(manifest["tensors"]),
                manifest["sha256"][:16],
            )
        except Exception:
            logger.exception(
                "snapshot background snapshot disk write FAILED; "
                "no snapshot written, next startup falls back to "
                "fresh load+process. "
                "Tip: check disk space (df -h), write permissions, "
                "and inode limits on the checkpoint path."
            )

    def _save_snapshot_async(
        self,
        model: nn.Module,
        fingerprint: dict | None = None,
        removed_params: list | None = None,
        scalar_attrs: dict | None = None,
    ):
        """Sync wrapper for tests: save_snapshot + _write_snapshot_async."""
        try:
            blob_parts, manifest = self.save_snapshot(model, fingerprint, removed_params, scalar_attrs)
            self._write_snapshot_async(blob_parts, manifest)
        except Exception:
            logger.exception(
                "snapshot background save FAILED; "
                "no snapshot written, next startup falls back to fresh load+process"
            )

    @staticmethod
    def _capture_scalar_state(model: nn.Module) -> dict[str, Any]:
        """Capture scalar attrs for pre/post diff during restore."""
        state: dict[str, Any] = {}
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in mod.__dict__.items():
                if attr.startswith("_"):
                    continue
                if isinstance(v, (int, float, str, bool)):
                    key = f"{mod_name}.{attr}" if mod_name else attr
                    state[key] = v
        return state

    @staticmethod
    def _check_fingerprint(saved_fp: dict | None, fingerprint: dict | None, manifest_path: str) -> None:
        """Refuse to restore a snapshot converted under a different config
        (model/dtype/quant/parallel layout)."""
        if saved_fp is None:
            logger.warning(
                "snapshot restore: snapshot predates config fingerprinting; skipping (model/parallel layout) validation"
            )
            return
        if fingerprint is None:
            return
        mismatched = {
            k: (saved_fp.get(k), fingerprint.get(k))
            for k in set(saved_fp) | set(fingerprint)
            if saved_fp.get(k) != fingerprint.get(k)
        }
        if mismatched:
            raise RuntimeError(
                f"snapshot restore: snapshot fingerprint mismatch {mismatched}; "
                f"the snapshot at {os.path.dirname(manifest_path)} was converted "
                f"under a different config -- use a fresh CHECKPOINT_PATH and "
                f"re-run convert"
            )

    @staticmethod
    def _plan_bulk(tensors: dict[str, dict], blob_len: int) -> tuple[bool, int, int, int]:
        """Decide bulk H2D chunk size. Returns (bulk, chunk_size, dev_bytes, nz_bytes)."""
        if not envs.VLLM_ASCEND_BIGTENSOR_BULK_H2D:
            return False, 0, 0, 0
        dev_entries = [e for e in tensors.values() if e.get("device") != "cpu"]
        dev_bytes = sum(e["nbytes"] for e in dev_entries)
        nz_bytes = sum(e["nbytes"] for e in dev_entries if e["format"] == 29)
        env_chunk_raw = os.getenv("VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB")
        if env_chunk_raw is not None:
            try:
                chunk_size = envs.VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB * 1024 * 1024
            except ValueError:
                raise RuntimeError(
                    f"VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB must be an integer (MB), got {env_chunk_raw!r}"
                )
        else:
            free_b, _ = torch.npu.mem_get_info()
            margin = max(4 * 1024**3, free_b // 10) + nz_bytes // 4
            budget = free_b - dev_bytes - margin
            if budget < 1024**3:
                logger.warning(
                    "snapshot bulk: dynamic chunk budget %.2fGB "
                    "(free %.2fGB - weights %.2fGB - margin %.2fGB) "
                    "< 1GB, falling back to per-tensor restore",
                    budget / 1024**3,
                    free_b / 1024**3,
                    dev_bytes / 1024**3,
                    margin / 1024**3,
                )
                return False, 0, dev_bytes, nz_bytes
            chunk_size = min(budget, blob_len)
        logger.info(
            "snapshot bulk: weights=%.2fGB(n_nz=%.2fGB) chunk=%.2fGB",
            dev_bytes / 1024**3,
            nz_bytes / 1024**3,
            chunk_size / 1024**3,
        )
        return True, chunk_size, dev_bytes, nz_bytes

    def _resolve_target(self, name: str, e: dict, param_dict: dict, name_to_module: dict, target_device) -> tuple:
        """Resolve manifest entry to (param, module, attr_name)."""
        kind = e.get("kind")
        if kind not in ("attr", "obj_attr"):
            dtype = getattr(torch, e["dtype"], None)
            dtype_key = self._DTYPE_KEY.get(dtype) if dtype is not None else None
            if dtype_key is None:
                raise RuntimeError(f"snapshot restore: dtype {e['dtype']} not in _DTYPE_MAP for {name}")
            param = self._resolve_param(name, param_dict, name_to_module, target_device, dtype_key)
            if param is None:
                return None, None, None  # stale, coverage check will report
            return param, None, None
        if kind == "attr":
            mod_name, _, attr_name = name.rpartition(".")
            module = name_to_module.get(mod_name)
            if module is None:
                raise RuntimeError(f"snapshot restore: module {mod_name} for attr {name} not found")
            return None, module, attr_name
        # kind == "obj_attr": resolve carrier by longest module prefix
        carrier_segs = e["carrier"].split(".")
        obj = None
        for i in range(len(carrier_segs), 0, -1):
            cand = ".".join(carrier_segs[:i])
            if cand in name_to_module:
                obj = name_to_module[cand]
                for s in carrier_segs[i:]:
                    obj = getattr(obj, s, None)
                    if obj is None:
                        break
                break
        if obj is None:
            raise RuntimeError(f"snapshot restore: carrier {e['carrier']} for {name} not found")
        return None, obj, e["attr"]

    @staticmethod
    def _check_coverage(
        model: nn.Module, tensors: dict, null_attrs: set[str], name_to_module: dict, removed_params: set[str]
    ) -> None:
        """Verify all model params are in snapshot, and replay null/deleted attrs."""
        model_keys = set(model.state_dict().keys())
        missing = [n for n in model_keys if n not in tensors]
        exempt = null_attrs | removed_params
        unexpected = [n for n in missing if n not in exempt]
        if unexpected:
            raise RuntimeError(
                f"snapshot restore: {len(unexpected)} model params not in snapshot, "
                f"first few: {unexpected[:5]} (re-run convert)"
            )
        # reverse: snapshot "param" entries not in model (structure drift)
        stale = [n for n, e in tensors.items() if n not in model_keys and e.get("kind") == "param" and n not in exempt]
        if stale:
            raise RuntimeError(
                f"snapshot restore: {len(stale)} snapshot params not found in "
                f"model, first few: {stale[:5]}. The snapshot was generated "
                f"for a different model structure -- re-run convert."
            )
        for n in missing:
            mod_name, _, attr_name = n.rpartition(".")
            module = name_to_module.get(mod_name)
            if module is None:
                continue
            if n in removed_params and hasattr(module, attr_name):
                delattr(module, attr_name)
            elif n in null_attrs:
                setattr(module, attr_name, None)

    @staticmethod
    def _verify_blob(mm: "mmap.mmap", manifest: dict, manifest_path: str) -> float:
        """Pre-restore integrity gate. Returns elapsed time.

        Modes (VLLM_ASCEND_BIGTENSOR_VERIFY):
        - size (default): per-entry bounds, sub-ms
        - sha256: bounds + whole-blob hash, ~tens of seconds on 31GB
        - none: skip
        """
        verify_mode = envs.VLLM_ASCEND_BIGTENSOR_VERIFY
        if verify_mode == "none":
            logger.info("snapshot restore: blob verification disabled (VLLM_ASCEND_BIGTENSOR_VERIFY=none)")
            return 0.0

        blob_len = len(mm)
        t0 = time.perf_counter()

        # (1) per-entry bounds validation
        n_entries = 0
        for name, e in manifest["tensors"].items():
            if e["nbytes"] == 0:
                continue
            n_entries += 1
            end = e["offset"] + e["nbytes"]
            if e["offset"] < 0 or end > blob_len:
                raise RuntimeError(
                    f"snapshot restore: snapshot blob is truncated or corrupt -- "
                    f"tensor '{name}' spans bytes [{e['offset']}, {end}) but "
                    f"the blob is only {blob_len} bytes. Delete the snapshot "
                    f"at {os.path.dirname(manifest_path)} and re-run convert."
                )

        if verify_mode == "size":
            logger.info(
                "snapshot restore: blob bounds verified (%d tensors, %.3fs)", n_entries, time.perf_counter() - t0
            )
            return time.perf_counter() - t0

        if verify_mode != "sha256":
            raise RuntimeError(
                f"snapshot restore: invalid VLLM_ASCEND_BIGTENSOR_VERIFY={verify_mode!r}; "
                "must be one of: none, size, sha256"
            )

        # (2) whole-blob sha256
        saved_sha = manifest.get("sha256")
        if not saved_sha:
            logger.warning(
                "snapshot restore: VLLM_ASCEND_BIGTENSOR_VERIFY=sha256 but "
                "manifest has no sha256 field (pre-checksum snapshot); "
                "skipping sha256"
            )
            return time.perf_counter() - t0
        import hashlib

        h = hashlib.sha256()
        chunk_size = 128 * 1024 * 1024  # 128 MiB
        for pos in range(0, blob_len, chunk_size):
            h.update(mm[pos : pos + chunk_size])
        actual_sha = h.hexdigest()
        if actual_sha != saved_sha:
            raise RuntimeError(
                f"snapshot restore: blob sha256 mismatch -- manifest records "
                f"{saved_sha[:16]}... but actual is {actual_sha[:16]}...; "
                f"the snapshot at {os.path.dirname(manifest_path)} is corrupt "
                f"or was only partially written. Delete it and re-run convert "
                f"rather than restoring corrupt weights."
            )
        logger.info(
            "snapshot restore: blob integrity verified (sha256=%s, %.2fs)", actual_sha[:16], time.perf_counter() - t0
        )
        return time.perf_counter() - t0

    def restore_snapshot(self, model: nn.Module, target_device, fingerprint: dict | None = None) -> None:
        """Restore post-process weights from snapshot, skip process entirely.

        Each tensor is cloned into standalone offset-0 storage (aclnn rejects
        nonzero-offset views). Bulk H2D (VLLM_ASCEND_BIGTENSOR_BULK_H2D=1)
        chunks the blob to keep peak memory at weights + one chunk.
        """
        manifest_path, blob_path = self._snapshot_paths()
        with open(manifest_path) as f:
            manifest = json.load(f)
        rank = torch.distributed.get_rank()
        self._check_fingerprint(manifest.get("fingerprint"), fingerprint, manifest_path)

        with open(blob_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            verify_time = self._verify_blob(mm, manifest, manifest_path)
            h2d_start = time.perf_counter()
            bulk, chunk_size, _, _ = self._plan_bulk(manifest["tensors"], len(mm))
            # Bulk requires offset-ordered consumption
            tensor_items = (
                sorted(manifest["tensors"].items(), key=lambda kv: kv[1]["offset"])
                if bulk
                else manifest["tensors"].items()
            )
            param_dict = dict(model.named_parameters(remove_duplicate=False))
            param_dict.update(dict(model.named_buffers(remove_duplicate=False)))
            name_to_module = dict(model.named_modules(remove_duplicate=False))
            dev_chunk = None  # current device-side bulk buffer
            chunk_base = 0  # blob offset where dev_chunk starts
            chunk_end = 0  # blob offset where dev_chunk ends
            bulk_h2d = 0.0
            n_chunks = 0
            n_nd = n_nz = 0

            for name, e in tensor_items:
                dtype = getattr(torch, e["dtype"], None)
                if dtype is None:
                    raise RuntimeError(f"snapshot restore: unsupported dtype {e['dtype']} for {name}")
                shape = tuple(e["shape"])
                elem_size = torch.empty((), dtype=dtype).element_size()
                if e["offset"] % elem_size != 0:
                    raise RuntimeError(f"snapshot restore: unaligned offset for {name}")
                # Descriptor self-consistency: nbytes must match shape*elem_size
                if e["nbytes"] != 0:
                    if "save_dtype" in e:
                        sd = getattr(torch, e["save_dtype"])
                        ss = tuple(e["save_shape"])
                        n_elems = 1
                        for d in ss:
                            n_elems *= int(d)
                        expected = n_elems * torch.empty((), dtype=sd).element_size()
                    else:
                        n_elems = 1
                        for d in shape:
                            n_elems *= int(d)
                        expected = n_elems * elem_size
                    if e["nbytes"] != expected:
                        raise RuntimeError(
                            f"snapshot restore: descriptor inconsistency for "
                            f"{name}: nbytes={e['nbytes']} but "
                            f"shape {shape} dtype {e['dtype']} implies "
                            f"{expected} bytes. Manifest may be corrupt "
                            f"or tampered -- re-run convert."
                        )
                param, module, attr_name = self._resolve_target(name, e, param_dict, name_to_module, target_device)
                kind = e.get("kind")
                if param is None and module is None and attr_name is None:
                    continue  # stale, coverage check reports

                with torch.no_grad():
                    # Packed NZ: build int8 tensor, view back to int32 after NZ cast
                    save_dtype = getattr(torch, e.get("save_dtype", e["dtype"]))
                    save_shape = tuple(e.get("save_shape", e["shape"]))
                    if e["nbytes"] == 0:
                        # Empty placeholder: reconstruct from manifest
                        tensor = torch.empty(save_shape, dtype=save_dtype)
                        if e.get("device") != "cpu":
                            tensor = tensor.to(target_device)
                            if e["format"] == 29:
                                tensor = torch_npu.npu_format_cast(tensor, 29)
                                n_nz += 1
                            else:
                                n_nd += 1
                    elif bulk and e.get("device") != "cpu":
                        if dev_chunk is None or e["offset"] + e["nbytes"] > chunk_end:
                            dev_chunk = None  # release before realloc
                            chunk_base = e["offset"]
                            # chunk covers at least the current tensor
                            chunk_end = min(max(chunk_base + chunk_size, e["offset"] + e["nbytes"]), len(mm))
                            t0 = time.perf_counter()
                            src = torch.frombuffer(
                                mm, dtype=torch.uint8, count=chunk_end - chunk_base, offset=chunk_base
                            )
                            dev_chunk = torch.empty(chunk_end - chunk_base, dtype=torch.uint8, device=target_device)
                            dev_chunk.copy_(src)
                            bulk_h2d += time.perf_counter() - t0
                            n_chunks += 1
                        view = (
                            dev_chunk.narrow(0, e["offset"] - chunk_base, e["nbytes"]).view(save_dtype).view(save_shape)
                        )
                        if e["format"] == 29:
                            tensor = torch_npu.npu_format_cast(view.clone(), 29)
                            n_nz += 1
                        else:
                            tensor = view.clone()  # offset-0 standalone
                            n_nd += 1
                    else:
                        cpu_bytes = torch.frombuffer(mm, dtype=torch.uint8, count=e["nbytes"], offset=e["offset"])
                        tensor = cpu_bytes.view(save_dtype).view(save_shape)
                        if e.get("device") == "cpu":
                            tensor = tensor.clone()
                        else:
                            tensor = tensor.to(target_device)
                            if e["format"] == 29:
                                tensor = torch_npu.npu_format_cast(tensor, 29)
                                n_nz += 1
                            else:
                                n_nd += 1
                    if "save_dtype" in e:
                        # Packed: view back to logical int32
                        tensor = tensor.view(dtype).contiguous()
                        # Logical anchor: shape/nbytes must match manifest
                        if tuple(tensor.shape) != shape or tensor.numel() * elem_size != e["nbytes"]:
                            raise RuntimeError(
                                f"snapshot restore: packed logical mismatch "
                                f"for {name}: decoded "
                                f"{tuple(tensor.shape)}/{tensor.dtype} "
                                f"!= manifest {shape}/{e['dtype']} "
                                f"(nbytes {e['nbytes']}). "
                                f"Re-run convert."
                            )
                    if kind in ("attr", "obj_attr"):
                        # Drift detection: attr shape/dtype set by code, not process
                        current = getattr(module, attr_name, None)
                        if isinstance(current, torch.Tensor) and (
                            current.shape != tensor.shape or current.dtype != tensor.dtype
                        ):
                            raise RuntimeError(
                                f"snapshot restore: shape/dtype mismatch for "
                                f"{name}: model has "
                                f"{tuple(current.shape)}/{current.dtype},"
                                f" snapshot has "
                                f"{tuple(tensor.shape)}/{tensor.dtype}."
                                f" Re-run convert."
                            )
                        setattr(module, attr_name, tensor)
                    elif "save_dtype" in e or dtype != param.dtype:
                        # Packed/dtype-changing: use .data (set_ rejects dtype change)
                        param.data = tensor
                    else:
                        # Standard: set_ preserves NZ metadata; .data for shape-changing
                        if param.numel() != 0 and tensor.shape != param.shape:
                            param.data = tensor
                        else:
                            param.set_(tensor)

            # Release last chunk, replay scalar side effects
            dev_chunk = None
            for name, value in manifest.get("scalar_attrs", {}).items():
                mod_name, _, attr_name = name.rpartition(".")
                module = name_to_module.get(mod_name)
                if module is not None:
                    setattr(module, attr_name, value)
            self._check_coverage(
                model,
                manifest["tensors"],
                set(manifest.get("null_attrs", ())),
                name_to_module,
                set(manifest.get("removed_params", ())),
            )
            h2d_time = time.perf_counter() - h2d_start
        logger.info(
            "restore_snapshot rank=%d mode=%s n_nd=%d n_nz=%d chunks=%d bulk_h2d=%.2fs verify=%.2fs h2d=%.2fs",
            rank,
            "bulk" if bulk else "per-tensor",
            n_nd,
            n_nz,
            n_chunks,
            bulk_h2d,
            verify_time,
            h2d_time,
        )

    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "") -> nn.Module:
        """Load model. FAST_RESTORE=1: snapshot present -> restore-only; absent -> load+process+async save."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else load_config.device
        target_device = torch.device(load_device)

        fast_restore_enabled = envs.VLLM_ASCEND_BIGTENSOR_FAST_RESTORE
        if fast_restore_enabled:
            fingerprint = self._fingerprint(vllm_config, model_config)
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                if self._has_snapshot():
                    self.restore_snapshot(model, target_device, fingerprint)
                    return model.eval()
                self.load_weights(model, model_config)
                pre_params = set(model.state_dict().keys())
                pre_scalar = self._capture_scalar_state(model)
                process_weights_after_loading(model, model_config, target_device)
                post_params = set(model.state_dict().keys())
                post_scalar = self._capture_scalar_state(model)
                removed_params = list(pre_params - post_params)
                scalar_attrs = {k: v for k, v in post_scalar.items() if pre_scalar.get(k) != v}
                # D2H sync (before graph capture); disk write async
                blob_parts, manifest = self.save_snapshot(model, fingerprint, removed_params, scalar_attrs)
                threading.Thread(
                    target=self._write_snapshot_async,
                    args=(blob_parts, manifest),
                    name="snapshot-write",
                    daemon=True,
                ).start()
                logger.info("snapshot snapshot save dispatched to background thread; startup continues without waiting")
            return model.eval()

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)

            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
