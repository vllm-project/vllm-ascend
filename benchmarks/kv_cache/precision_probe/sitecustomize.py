# SPDX-License-Identifier: Apache-2.0

"""Temporary process-local probe for the Jenga KV-cache accuracy diagnosis.

Add this directory to ``PYTHONPATH`` and set ``JENGA_PRECISION_PROBE_DIR`` to
capture the first two decode steps.  The probe monkey-patches only the running
Python process; production vLLM-Ascend source is not modified.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any

PROBE_DIR = os.getenv("JENGA_PRECISION_PROBE_DIR")
PROBE_LABEL = os.getenv("JENGA_PRECISION_PROBE_LABEL", "unknown")
ADDRESS_ORDER = os.getenv("JENGA_PRECISION_PROBE_ADDRESS_ORDER")


def _tensor_signature(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().contiguous()
    raw = value.view(torch.uint8).cpu().numpy().tobytes()
    flat = value.float().reshape(-1)
    sample = flat[: min(8, flat.numel())].cpu().tolist()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sample": sample,
    }


def _output_signatures(value: Any) -> list[dict[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        return [_tensor_signature(value)]
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_output_signatures(item))
        return result
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_output_signatures(value[key]))
        return result
    return []


if PROBE_DIR:
    _output_path = Path(PROBE_DIR) / f"{PROBE_LABEL}.jsonl"
    _output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_lock = threading.Lock()

    def _write(record: dict[str, Any]) -> None:
        record = {"label": PROBE_LABEL, **record}
        with _write_lock, _output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _install_layer_hooks(runner: Any) -> None:
        if getattr(runner, "_jenga_probe_hooks_installed", False):
            return
        installed = []
        for name, module in runner.model.named_modules():
            is_decoder_layer = module.__class__.__name__ == "Qwen3_5DecoderLayer"
            is_layer_zero_boundary = name in {
                "language_model.model.layers.0.input_layernorm",
                "language_model.model.layers.0.linear_attn",
                "language_model.model.layers.0.post_attention_layernorm",
            }
            if not is_decoder_layer and not is_layer_zero_boundary:
                continue

            def hook(_module, _inputs, output, *, layer_name=name):
                step = getattr(runner, "_jenga_probe_step", -1)
                if step not in (1, 2):
                    return
                _write(
                    {
                        "event": "layer_output",
                        "step": step,
                        "layer": layer_name,
                        "module_class": _module.__class__.__name__,
                        "outputs": _output_signatures(output),
                    }
                )

            installed.append(module.register_forward_hook(hook))
        runner._jenga_probe_hooks_installed = True
        runner._jenga_probe_hook_handles = installed

        # ``rearrange_mixed_qkv`` is called immediately after causal-conv and
        # before recurrent GDN.  Capturing its non-None input separates the two
        # stateful kernels without changing the production module source.
        for name, module in runner.model.named_modules():
            if name != "language_model.model.layers.0.linear_attn":
                continue
            original_rearrange = module.rearrange_mixed_qkv

            def rearrange_mixed_qkv(this, mixed_qkv, *, _original=original_rearrange):
                step = getattr(runner, "_jenga_probe_step", -1)
                if step in (1, 2) and mixed_qkv is not None:
                    call_counts = getattr(runner, "_jenga_probe_rearrange_counts", {})
                    call_index = call_counts.get(step, 0)
                    call_counts[step] = call_index + 1
                    runner._jenga_probe_rearrange_counts = call_counts
                    _write(
                        {
                            "event": "post_causal_conv",
                            "step": step,
                            "call_index": call_index,
                            "mixed_qkv": _tensor_signature(mixed_qkv),
                        }
                    )
                return _original(mixed_qkv)

            module.rearrange_mixed_qkv = types.MethodType(rearrange_mixed_qkv, module)
            break
        layer_zero_modules = [
            {"name": name, "class": module.__class__.__name__}
            for name, module in runner.model.named_modules()
            if name.startswith("language_model.model.layers.0")
        ]
        _write(
            {
                "event": "hooks_installed",
                "count": len(installed),
                "layer_zero_modules": layer_zero_modules,
            }
        )

    def _dump_gdn_metadata(runner: Any, step: int) -> None:
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        metadata_by_layer = context.attn_metadata
        if not isinstance(metadata_by_layer, dict):
            return
        records = []
        state_index = None
        for name, metadata in metadata_by_layer.items():
            if "layers.0.linear_attn" not in name:
                continue

            def values(value: Any) -> list[int] | None:
                if value is None:
                    return None
                return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]

            decode_metadata = getattr(metadata, "non_spec_decode_metadata", None)
            causal_conv1d = getattr(decode_metadata, "causal_conv1d", None)
            state_indices = getattr(metadata, "non_spec_state_indices_tensor", None)
            state_index_values = values(state_indices)
            if state_index_values:
                state_index = state_index_values[0]
            records.append(
                {
                    "layer": name,
                    "non_spec_state_indices": state_index_values,
                    "conv_cache_indices": values(getattr(causal_conv1d, "cache_indices", None)),
                    "num_decodes": int(getattr(metadata, "num_decodes", -1)),
                    "num_prefills": int(getattr(metadata, "num_prefills", -1)),
                }
            )
        cache_tensors = []
        if state_index is not None:
            for name, module in runner.model.named_modules():
                if name != "language_model.model.layers.0.linear_attn":
                    continue
                for cache in getattr(module, "kv_cache", ()):
                    cache_tensors.append(
                        {
                            "shape": list(cache.shape),
                            "stride": list(cache.stride()),
                            "storage_offset": int(cache.storage_offset()),
                            "selected_state": _tensor_signature(cache[state_index]),
                        }
                    )
        _write(
            {
                "event": "gdn_metadata",
                "step": step,
                "records": records,
                "cache_tensors": cache_tensors,
            }
        )

    def _dump_cache_before_forward(runner: Any, step: int) -> None:
        from vllm_ascend.core.typed_kv_cache import (
            get_typed_kv_cache_plan,
            make_block_byte_view,
        )

        plan = get_typed_kv_cache_plan(runner.kv_cache_config)
        if plan is None:
            return
        raw_tensors = getattr(runner, "_typed_kv_cache_raw_tensors", ())
        groups = []
        for group_id, block_table in enumerate(runner.input_batch.block_table.block_tables):
            count = int(block_table.num_blocks_per_row[0])
            logical_ids = [int(item) for item in block_table.block_table.np[0, :count].tolist()]
            physical_ids = [int(item) for item in block_table.get_device_tensor(1)[0, :count].cpu().tolist()]
            raw_hashes = []
            for raw_tensor in raw_tensors:
                digest = hashlib.sha256()
                for logical_id in logical_ids:
                    page = make_block_byte_view(raw_tensor, plan, group_id, logical_id)
                    digest.update(page.cpu().numpy().tobytes())
                raw_hashes.append(digest.hexdigest())
            groups.append(
                {
                    "group_id": group_id,
                    "logical_ids": logical_ids,
                    "physical_ids": physical_ids,
                    "slot_mapping": [int(item) for item in block_table.slot_mapping.gpu[:1].cpu().tolist()],
                    "logical_page_hashes_by_raw_tensor": raw_hashes,
                }
            )
        _write({"event": "cache_before_forward", "step": step, "groups": groups})

    _patch_lock = threading.Lock()
    _patched = False
    _address_pool_patched = False

    def _try_patch_address_pool() -> None:
        global _address_pool_patched
        if not ADDRESS_ORDER or _address_pool_patched:
            return
        module = sys.modules.get("vllm_ascend.core.typed_kv_cache")
        pool_class = getattr(module, "TypedAddressPool", None)
        if pool_class is None:
            return
        if ADDRESS_ORDER != "small_low_large_high":
            raise ValueError(f"unsupported diagnostic address order: {ADDRESS_ORDER}")

        def candidate_ids(self, group_id):
            spec = self.plan.spec(group_id)
            smallest_page = min(item.page_size_bytes for item in self.plan.specs)
            ids = list(range(1, self.plan.num_blocks(group_id)))
            return ids if spec.page_size_bytes == smallest_page else list(reversed(ids))

        pool_class._candidate_ids = candidate_ids
        _address_pool_patched = True

    def _try_patch() -> None:
        global _patched
        if _patched:
            return
        module = sys.modules.get("vllm_ascend.worker.model_runner_v1")
        runner_class = getattr(module, "NPUModelRunner", None)
        if runner_class is None:
            return
        with _patch_lock:
            if _patched:
                return
            original_execute_model = runner_class.execute_model
            original_model_forward = runner_class._model_forward

            def execute_model(self, scheduler_output, *args, **kwargs):
                step = getattr(self, "_jenga_probe_next_step", 0)
                self._jenga_probe_step = step
                self._jenga_probe_next_step = step + 1
                _install_layer_hooks(self)
                return original_execute_model(self, scheduler_output, *args, **kwargs)

            def model_forward(self, *args, **kwargs):
                step = getattr(self, "_jenga_probe_step", -1)
                if step in (1, 2):
                    _dump_cache_before_forward(self, step)
                    _dump_gdn_metadata(self, step)
                return original_model_forward(self, *args, **kwargs)

            runner_class.execute_model = execute_model
            runner_class._model_forward = model_forward
            _patched = True

    _original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        result = _original_import(name, globals, locals, fromlist, level)
        _try_patch_address_pool()
        _try_patch()
        return result

    builtins.__import__ = _import
    _try_patch_address_pool()
    _try_patch()
