# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import hashlib
import json
import mmap
import os
import threading
import time
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.config import LoadConfig, ModelConfig, VllmConfig, get_current_vllm_config_or_none
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend import envs
from vllm_ascend.model_loader.weight_state import iter_module_weight_states

from .bigtensordefaultloader import BigTensorDefaultLoader


@register_model_loader("bigtensorloader")
class BigTensorLoader(BigTensorDefaultLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    # Manifest schema this loader writes and restores. v3 adds: container
    # capture (module/attr/path chains + container defs), alias dedup
    # (alias_of) so tied weights persist once, and per-tensor spot hashes
    # for small tensors. Older snapshots are rejected and regenerated.
    _SCHEMA_VERSION = 3

    # Tensors up to this size get a per-tensor sha256 in the manifest, checked
    # on restore even in "size" verify mode. Small tensors are exactly the
    # scale/bias weights whose silent corruption is hardest to notice.
    _SPOT_HASH_MAX_BYTES = 4 * 1024 * 1024

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
        if len(raw) <= self._SPOT_HASH_MAX_BYTES:
            # Content check for small tensors even in "size" verify mode;
            # cheap here because raw is already a contiguous host copy.
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
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
        Captures params/buffers, the full non-state_dict weight state (plain
        tensor attrs, containers, carrier objects -- via the shared
        weight_state walker also used by netloader/rfork), alias dedup for
        tied weights, null attrs, removed params, and scalar side effects.
        Model-agnostic: integrity is enforced by blob sha256 + per-tensor
        spot hashes + descriptor self-consistency + fingerprint.
        """
        rank = torch.distributed.get_rank()
        manifest_path, blob_path = self._snapshot_paths()
        entries: dict[str, dict] = {}
        blob_parts: list[bytes] = []
        offset = 0
        # (data_ptr, dtype, shape, format) -> name of the primary entry.
        # Tied weights / re-referenced tensors are stored once and installed
        # as aliases at restore, so tying survives and the blob stays small.
        alias_primaries: dict[tuple, str] = {}
        n_aliases = 0

        def dedup_key(t: torch.Tensor) -> tuple:
            fmt = int(torch_npu.get_npu_format(t)) if t.device.type == "npu" else 0
            return (t.data_ptr(), t.dtype, tuple(t.shape), fmt)

        # (1) params/buffers in state_dict
        for name, t in model.state_dict().items():
            key = dedup_key(t)
            primary = alias_primaries.get(key)
            if primary is not None:
                entries[name] = {
                    "kind": "param",
                    "alias_of": primary,
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                    "format": key[3],
                    "nbytes": 0,
                }
                n_aliases += 1
                continue
            entry, offset = self._append_tensor(t, blob_parts, offset)
            entry["kind"] = "param"
            entries[name] = entry
            alias_primaries[key] = name

        # (2) full non-state_dict weight state via the shared walker:
        # plain tensor attrs, list/tuple/dict containers (recorded as
        # module+attr+path chains plus container defs), and carrier-object
        # tensors (e.g. MLA impl W_UV / W_UK_T). Same enumeration policy as
        # netloader's field-tested P2P transfer, incl. cycle protection and
        # skipping private runtime attrs.
        containers: dict[str, dict] = {}
        for rec in iter_module_weight_states(model, deep_scan_all_objects=True, container_defs=containers):
            t = rec.tensor
            if t.numel() == 0:
                continue  # runtime placeholder, no data to persist
            if rec.name in entries:
                continue  # same slot reached twice (shared graph); already stored
            key = dedup_key(t)
            primary = alias_primaries.get(key)
            if primary is not None:
                entries[rec.name] = {
                    "kind": "attr",
                    "alias_of": primary,
                    "module": rec.module,
                    "attr": rec.attr,
                    "path": [dict(step) for step in rec.path],
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                    "format": key[3],
                    "nbytes": 0,
                    "device": t.device.type,
                }
                n_aliases += 1
                continue
            entry, offset = self._append_tensor(t, blob_parts, offset)
            entry.update(
                {
                    "kind": "attr",
                    "module": rec.module,
                    "attr": rec.attr,
                    "path": [dict(step) for step in rec.path],
                    "device": t.device.type,
                }
            )
            entries[rec.name] = entry
            alias_primaries[key] = rec.name

        # (3) Attrs nulled by process (recorded for restore to replay)
        null_attrs = []
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in mod.__dict__.items():
                if v is None:
                    null_attrs.append(f"{mod_name}.{attr}" if mod_name else attr)

        h = hashlib.sha256()
        for raw in blob_parts:
            h.update(raw)
        manifest = {
            "schema_version": self._SCHEMA_VERSION,
            "format_note": "Model-agnostic: integrity by sha256 + spot hashes + descriptor + fingerprint",
            "world_rank": rank,
            "world_size": torch.distributed.get_world_size(),
            "fingerprint": fingerprint,
            "sha256": h.hexdigest(),
            "tensors": entries,
            "containers": containers,
            "null_attrs": null_attrs,
            "removed_params": removed_params or [],
            "scalar_attrs": scalar_attrs or {},
        }
        n_spot = sum(1 for e in entries.values() if "sha256" in e)
        logger.info(
            "save_snapshot rank=%d tensors=%d (aliases=%d spot_hashed=%d) "
            "containers=%d bytes=%d sha256=%s (serialized, awaiting disk write)",
            rank,
            len(entries),
            n_aliases,
            n_spot,
            len(containers),
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
                "snapshot background save FAILED; no snapshot written, next startup falls back to fresh load+process"
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
            # free_b already excludes model params (allocated by
            # initialize_model before restore); only keep a safety margin.
            margin = max(4 * 1024**3, free_b // 10) + nz_bytes // 4
            budget = free_b - margin
            if budget < 1024**3:
                logger.warning(
                    "snapshot bulk: dynamic chunk budget %.2fGB "
                    "(free %.2fGB - margin %.2fGB) "
                    "< 1GB, falling back to per-tensor restore",
                    budget / 1024**3,
                    free_b / 1024**3,
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
        """Resolve a manifest 'param' entry to its model parameter (or None if stale)."""
        dtype = getattr(torch, e["dtype"], None)
        dtype_key = self._DTYPE_KEY.get(dtype) if dtype is not None else None
        if dtype_key is None:
            raise RuntimeError(f"snapshot restore: dtype {e['dtype']} not in _DTYPE_MAP for {name}")
        param = self._resolve_param(name, param_dict, name_to_module, target_device, dtype_key)
        if param is None:
            return None  # stale, coverage check will report
        return param

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
    def _container_def_key(module_name: str, attr: str, chain: tuple[dict, ...]) -> str:
        """Key into manifest["containers"] for the container at ``chain``.

        Must mirror weight_state.iter_module_weight_states: root attr as first
        step, module prefix scoping."""
        return f"{module_name}|{json.dumps(({'a': attr}, *chain))}"

    def _step_into(
        self,
        name: str,
        module_name: str,
        attr: str,
        parent: Any,
        step: dict,
        next_step: dict,
        chain: tuple[dict, ...],
        container_defs: dict[str, dict],
        created: list,
    ) -> Any:
        """Descend one path step, creating the child container when missing.

        ``chain`` is the steps taken so far *excluding* ``step``; the child's
        own def key includes ``step``. The declared container type comes from
        the recorded def (a dict child must not be reborn as a list);
        ``next_step`` only backfills when no def was recorded.
        """
        if "a" in step:
            child = getattr(parent, step["a"], None)
            if child is None:
                raise RuntimeError(
                    f"snapshot restore: object attribute {step['a']!r} on the way "
                    f"to {name} is missing on the model; the snapshot is stale "
                    f"or the carrier is created during process -- re-run convert."
                )
            return child
        child_key = self._container_def_key(module_name, attr, (*chain, step))
        d = container_defs.get(child_key)
        child_type = d.get("type") if d else ("dict" if "k" in next_step else "list")
        is_tuple = child_type == "tuple"
        length = d.get("len", 0) if d else 0

        def _new_child(container_index: int) -> Any:
            if child_type == "dict":
                return {}
            return [None] * max(length, container_index + 1)

        if "k" in step:
            key = step["k"]
            if not isinstance(parent, dict):
                raise RuntimeError(
                    f"snapshot restore: slot for {name} is a {type(parent).__name__}, "
                    f"manifest expects dict; re-run convert."
                )
            if key in parent:
                return parent[key]
            child: Any = _new_child(0)
            parent[key] = child
            created.append((parent, ("k", key), is_tuple, child))
            return child
        index = step["i"]
        if isinstance(parent, dict):
            if index in parent:
                return parent[index]
            child = _new_child(0)
            parent[index] = child
            created.append((parent, ("k", index), is_tuple, child))
            return child
        if not isinstance(parent, list):
            raise RuntimeError(
                f"snapshot restore: slot for {name} is a {type(parent).__name__}, "
                f"manifest expects list; re-run convert."
            )
        while len(parent) <= index:
            parent.append(None)
        slot = parent[index]
        if slot is None:
            slot = _new_child(index)
            parent[index] = slot
            created.append((parent, ("i", index), is_tuple, slot))
        return slot

    def _install_attr_tensor(
        self,
        name: str,
        e: dict,
        tensor: torch.Tensor,
        name_to_module: dict,
        container_defs: dict[str, dict],
        created: list,
    ) -> None:
        """Install a restored tensor into the model via its access chain.

        Plain attr (empty path) -> setattr. Containers/carriers -> walk the
        tagged chain from the walker's save, creating missing containers from
        the recorded defs. Existing slots are drift-checked (shape/dtype)."""
        module = name_to_module.get(e.get("module", ""))
        if module is None:
            raise RuntimeError(
                f"snapshot restore: module {e.get('module')!r} for {name} not found; "
                f"the snapshot is stale -- re-run convert."
            )
        attr = e["attr"]
        path = e.get("path") or []

        if not path:
            current = getattr(module, attr, None)
            if isinstance(current, torch.Tensor) and (current.shape != tensor.shape or current.dtype != tensor.dtype):
                raise RuntimeError(
                    f"snapshot restore: shape/dtype mismatch for {name}: model has "
                    f"{tuple(current.shape)}/{current.dtype}, snapshot has "
                    f"{tuple(tensor.shape)}/{tensor.dtype}. Re-run convert."
                )
            setattr(module, attr, tensor)
            return

        parent = getattr(module, attr, None)
        if parent is None:
            # Root container attr (e.g. module.caches = [..]) missing on the
            # freshly initialized model: recreate it from the recorded def.
            first = path[0]
            if not ("i" in first or "k" in first):
                raise RuntimeError(
                    f"snapshot restore: carrier attr {attr!r} for {name} is missing "
                    f"on the model; the snapshot is stale -- re-run convert."
                )
            d = container_defs.get(self._container_def_key(e.get("module", ""), attr, ()))
            root_type = d.get("type") if d else ("dict" if "k" in first else "list")
            is_tuple = root_type == "tuple"
            length = d.get("len", 0) if d else 0
            parent: Any = {} if root_type == "dict" else [None] * max(length, 1)
            setattr(module, attr, parent)
            created.append((module, ("a", attr), is_tuple, parent))
        for depth, step in enumerate(path[:-1]):
            parent = self._step_into(
                name,
                e.get("module", ""),
                attr,
                parent,
                step,
                path[depth + 1],
                tuple(path[:depth]),
                container_defs,
                created,
            )

        last = path[-1]
        if "a" in last:
            sub = last["a"]
            current = getattr(parent, sub, None)
            if isinstance(current, torch.Tensor) and (current.shape != tensor.shape or current.dtype != tensor.dtype):
                raise RuntimeError(
                    f"snapshot restore: shape/dtype mismatch for {name}: model has "
                    f"{tuple(current.shape)}/{current.dtype}, snapshot has "
                    f"{tuple(tensor.shape)}/{tensor.dtype}. Re-run convert."
                )
            setattr(parent, sub, tensor)
        elif "k" in last:
            key = last["k"]
            if not isinstance(parent, dict):
                raise RuntimeError(
                    f"snapshot restore: slot for {name} is a {type(parent).__name__}, "
                    f"manifest expects dict; re-run convert."
                )
            current = parent.get(key)
            if isinstance(current, torch.Tensor) and (current.shape != tensor.shape or current.dtype != tensor.dtype):
                raise RuntimeError(
                    f"snapshot restore: shape/dtype mismatch for {name}: model has "
                    f"{tuple(current.shape)}/{current.dtype}, snapshot has "
                    f"{tuple(tensor.shape)}/{tensor.dtype}. Re-run convert."
                )
            parent[key] = tensor
        else:
            index = last["i"]
            if isinstance(parent, tuple):
                # Tuples are immutable: rebuild the container and rebind it.
                # Only supported when the tuple is the root attr itself;
                # nested immutable containers cannot be mutated in place.
                if len(path) != 1:
                    raise RuntimeError(
                        f"snapshot restore: {name} lives inside a nested immutable tuple; re-run convert."
                    )
                parent = tuple(tensor if i == index else v for i, v in enumerate(parent))
                setattr(module, attr, parent)
                return
            if isinstance(parent, dict):
                current = parent.get(index)
            elif isinstance(parent, list):
                while len(parent) <= index:
                    parent.append(None)
                current = parent[index]
            else:
                raise RuntimeError(
                    f"snapshot restore: slot for {name} is a {type(parent).__name__}, "
                    f"manifest expects list; re-run convert."
                )
            if isinstance(current, torch.Tensor) and (current.shape != tensor.shape or current.dtype != tensor.dtype):
                raise RuntimeError(
                    f"snapshot restore: shape/dtype mismatch for {name}: model has "
                    f"{tuple(current.shape)}/{current.dtype}, snapshot has "
                    f"{tuple(tensor.shape)}/{tensor.dtype}. Re-run convert."
                )
            parent[index] = tensor

    @staticmethod
    def _seal_created_containers(created: list) -> None:
        """Convert containers we recreated to tuples where the save recorded tuples.

        Sealed in reverse creation order: nested containers are created before
        their parents, so sealing children first lets the parent's ``tuple(child)``
        pick up already-sealed (tuple) children instead of re-wrapping the
        unsealed list and orphaning the inner seal.
        """
        for parent, how, is_tuple, child in reversed(created):
            if not is_tuple:
                continue
            sealed = tuple(child)
            if how[0] == "a":
                setattr(parent, how[1], sealed)
            else:
                parent[how[1]] = sealed

    def _delete_snapshot(self) -> None:
        """Remove a stale/corrupt snapshot so the fallback path regenerates it."""
        try:
            manifest_path, blob_path = self._snapshot_paths()
        except RuntimeError:
            return  # checkpoint path unset/unusable; nothing to delete
        for path in (manifest_path, blob_path, manifest_path + ".tmp", blob_path + ".tmp"):
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info("snapshot restore: deleted stale snapshot file %s", path)
                except OSError:
                    logger.warning("snapshot restore: failed to delete stale snapshot file %s", path)

    @staticmethod
    def _iter_static_contexts(vllm_config: VllmConfig):
        """Yield unique (compilation_config, forward_context, moe_layers) triples.

        Covers both the passed config and the thread-current vllm config: layer
        registration goes through the current config context, which in
        same-process multi-model setups (e.g. speculative-decoding drafts) is
        not necessarily the config passed to load_model.
        """
        candidates = [vllm_config]
        current_vllm_config = get_current_vllm_config_or_none()
        if current_vllm_config is not None:
            candidates.append(current_vllm_config)
        seen_compilation: set[int] = set()
        for config in candidates:
            compilation_config = getattr(config, "compilation_config", None)
            if compilation_config is None or id(compilation_config) in seen_compilation:
                continue
            seen_compilation.add(id(compilation_config))
            forward_context = getattr(compilation_config, "static_forward_context", None)
            moe_layers = getattr(compilation_config, "static_all_moe_layers", None)
            if not isinstance(forward_context, dict):
                continue
            yield compilation_config, forward_context, moe_layers

    @classmethod
    def _snapshot_forward_context(cls, vllm_config: VllmConfig) -> dict:
        """Baseline keys/MoE lists before initialize_model, for precise fallback cleanup."""
        baseline: dict = {"ctx_keys": {}, "moe_layers": {}}
        for _, forward_context, moe_layers in cls._iter_static_contexts(vllm_config):
            if id(forward_context) not in baseline["ctx_keys"]:
                try:
                    baseline["ctx_keys"][id(forward_context)] = set(forward_context.keys())
                except TypeError:
                    baseline["ctx_keys"][id(forward_context)] = set()
            if isinstance(moe_layers, list) and id(moe_layers) not in baseline["moe_layers"]:
                baseline["moe_layers"][id(moe_layers)] = (moe_layers, list(moe_layers))
        return baseline

    @classmethod
    def _reset_forward_context(
        cls,
        vllm_config: VllmConfig,
        failed_model: nn.Module | None = None,
        baseline: dict | None = None,
    ) -> None:
        """Drop the abandoned model's registrations so a second
        initialize_model(vllm_config) does not hit 'Duplicate layer name'.

        Attention/MoE/MLA layers register themselves by prefix into
        ``static_forward_context`` (and MoE appends to ``static_all_moe_layers``).
        The abandoned model instance is discarded on fallback, so only the
        entries it added must go; the fresh model re-registers itself as it is
        built. Removal is precise (new keys + keys owned by the failed model's
        modules; MoE lists rolled back to the pre-load baseline) so that, in a
        same-process multi-model setup, a failed restore of one model does not
        wipe the live registrations of another. Mirrors the netloader cleanup.
        """
        if baseline is None:
            baseline = cls._snapshot_forward_context(vllm_config)
        stale_module_ids = {id(m) for m in failed_model.modules()} if failed_model is not None else set()
        for _, forward_context, moe_layers in cls._iter_static_contexts(vllm_config):
            keep_keys = baseline["ctx_keys"].get(id(forward_context), set())
            stale_keys = [
                key
                for key, module in list(forward_context.items())
                if key not in keep_keys or id(module) in stale_module_ids
            ]
            for key in stale_keys:
                del forward_context[key]
            if isinstance(moe_layers, list):
                rollback = baseline["moe_layers"].get(id(moe_layers))
                if rollback is not None:
                    moe_layers[:] = rollback[1]

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
            # Spot content check for small tensors (scales/biases): bounds
            # checks cannot detect in-place tampering of a same-size tensor,
            # but a corrupted norm weight silently poisons every output.
            n_spot = 0
            for name, e in manifest["tensors"].items():
                if "sha256" not in e or e["nbytes"] == 0:
                    continue
                digest = hashlib.sha256(mm[e["offset"] : e["offset"] + e["nbytes"]]).hexdigest()
                if digest != e["sha256"]:
                    raise RuntimeError(
                        f"snapshot restore: spot hash mismatch for '{name}' -- "
                        f"manifest records {e['sha256'][:16]}... but the blob holds "
                        f"{digest[:16]}...; the snapshot at "
                        f"{os.path.dirname(manifest_path)} is corrupt or was "
                        f"tampered with. Re-run convert."
                    )
                n_spot += 1
            logger.info(
                "snapshot restore: blob bounds verified (%d tensors, %d spot-hashed, %.3fs)",
                n_entries,
                n_spot,
                time.perf_counter() - t0,
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
        nonzero-offset views). Aliased tensors (tied weights, re-referenced
        derived tensors) are installed once and re-shared, so tying survives.
        Bulk H2D (VLLM_ASCEND_BIGTENSOR_BULK_H2D=1) chunks the blob to keep
        peak memory at weights + one chunk.

        Raises on any integrity/staleness problem; the caller (load_model)
        catches, deletes the snapshot, and falls back to a fresh load.
        """
        manifest_path, blob_path = self._snapshot_paths()
        with open(manifest_path) as f:
            manifest = json.load(f)
        rank = torch.distributed.get_rank()
        schema = manifest.get("schema_version", 0)
        if schema != self._SCHEMA_VERSION:
            raise RuntimeError(
                f"snapshot restore: unsupported manifest schema_version={schema} "
                f"(expected {self._SCHEMA_VERSION}). The snapshot at "
                f"{os.path.dirname(manifest_path)} predates container capture "
                f"and alias dedup; it will be deleted and regenerated from a "
                f"fresh load."
            )
        self._check_fingerprint(manifest.get("fingerprint"), fingerprint, manifest_path)

        container_defs = manifest.get("containers", {})
        with open(blob_path, "rb") as bf, mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            verify_time = self._verify_blob(mm, manifest, manifest_path)
            h2d_start = time.perf_counter()
            bulk, chunk_size, _, _ = self._plan_bulk(manifest["tensors"], len(mm))
            # Alias entries carry no blob bytes; consume blobs in offset order
            # for the real entries, then install aliases from the registry.
            real_items = [(n, e) for n, e in manifest["tensors"].items() if "alias_of" not in e]
            tensor_items = sorted(real_items, key=lambda kv: kv[1]["offset"]) if bulk else real_items
            param_dict = dict(model.named_parameters(remove_duplicate=False))
            param_dict.update(dict(model.named_buffers(remove_duplicate=False)))
            name_to_module = dict(model.named_modules(remove_duplicate=False))
            registry: dict[str, torch.Tensor] = {}  # name -> restored standalone tensor
            created_containers: list = []  # containers recreated by install, sealed to tuples at the end
            dev_chunk = None  # current device-side bulk buffer
            chunk_base = 0  # blob offset where dev_chunk starts
            chunk_end = 0  # blob offset where dev_chunk ends
            bulk_h2d = 0.0
            n_chunks = 0
            n_nd = n_nz = 0

            for name, e in tensor_items:
                kind = e.get("kind")
                if kind == "param":
                    param = self._resolve_target(name, e, param_dict, name_to_module, target_device)
                    if param is None:
                        continue  # stale, coverage check will report
                else:
                    param = None
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
                    registry[name] = tensor
                    if param is not None:
                        if "save_dtype" in e or dtype != param.dtype:
                            # Packed/dtype-changing: use .data (set_ rejects dtype change)
                            param.data = tensor
                        else:
                            # Standard: set_ preserves NZ metadata; .data for shape-changing
                            if param.numel() != 0 and tensor.shape != param.shape:
                                param.data = tensor
                            else:
                                param.set_(tensor)
                    else:
                        self._install_attr_tensor(name, e, tensor, name_to_module, container_defs, created_containers)

            # Alias pass: re-share primaries so tied weights stay tied. The
            # registry was filled in the complete tensor pass above, so
            # manifest ordering between primaries and aliases is irrelevant.
            n_aliases = 0
            for name, e in manifest["tensors"].items():
                alias_of = e.get("alias_of")
                if alias_of is None:
                    continue
                primary = registry.get(alias_of)
                if primary is None:
                    raise RuntimeError(
                        f"snapshot restore: alias {name!r} references {alias_of!r} "
                        f"which was not restored -- the manifest is corrupt; "
                        f"re-run convert."
                    )
                with torch.no_grad():
                    if e.get("kind") == "param":
                        param = self._resolve_target(name, e, param_dict, name_to_module, target_device)
                        if param is None:
                            continue  # stale alias target, coverage check reports
                        if tuple(param.shape) != tuple(primary.shape) or param.dtype != primary.dtype:
                            raise RuntimeError(
                                f"snapshot restore: alias {name}: model has "
                                f"{tuple(param.shape)}/{param.dtype} but primary "
                                f"{alias_of} restored as {tuple(primary.shape)}/"
                                f"{primary.dtype}. Re-run convert."
                            )
                        param.set_(primary)
                    else:
                        self._install_attr_tensor(name, e, primary, name_to_module, container_defs, created_containers)
                    n_aliases += 1

            # Containers we recreated were built as lists (tuples are
            # immutable); seal them back to tuples where save recorded them.
            self._seal_created_containers(created_containers)

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
            "restore_snapshot rank=%d mode=%s n_nd=%d n_nz=%d aliases=%d chunks=%d "
            "bulk_h2d=%.2fs verify=%.2fs h2d=%.2fs",
            rank,
            "bulk" if bulk else "per-tensor",
            n_nd,
            n_nz,
            n_aliases,
            n_chunks,
            bulk_h2d,
            verify_time,
            h2d_time,
        )

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        """Load model. FAST_RESTORE=1: snapshot present -> restore-only; absent -> load+process+async save.

        Restore failures never kill startup: the stale snapshot is deleted
        and loading falls back to the fresh path (which re-saves a new,
        correct snapshot). Mirrors netloader's revert_to_default philosophy --
        degrade to the known-good path instead of serving broken weights.
        """
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else load_config.device
        target_device = torch.device(load_device)

        fast_restore_enabled = envs.VLLM_ASCEND_BIGTENSOR_FAST_RESTORE
        fingerprint = self._fingerprint(vllm_config, model_config) if fast_restore_enabled else None

        if fast_restore_enabled and self._has_snapshot():
            model = None
            context_baseline = self._snapshot_forward_context(vllm_config)
            try:
                with set_default_torch_dtype(model_config.dtype), target_device:
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                self.restore_snapshot(model, target_device, fingerprint)
                return model.eval()
            except Exception:
                logger.exception(
                    "bigtensorloader: snapshot restore failed; deleting the "
                    "stale snapshot and falling back to a fresh load + process"
                )
                self._delete_snapshot()
                # The abandoned model already registered its layers by prefix
                # into the compilation config; drop only those (new/owned)
                # registrations so the fresh initialize_model below does not
                # hit 'Duplicate layer name' -- and a co-resident model's live
                # registrations survive.
                self._reset_forward_context(vllm_config, model, context_baseline)
                model = None
                gc.collect()
                if target_device.type == "npu":
                    torch.npu.empty_cache()
                # fall through to the fresh path below

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)

            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            if fast_restore_enabled:
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
            else:
                process_weights_after_loading(model, model_config, target_device)

        return model.eval()
