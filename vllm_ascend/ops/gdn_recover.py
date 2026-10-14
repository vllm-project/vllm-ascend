# SPDX-License-Identifier: Apache-2.0
"""Owned, eager GDN recovery lifecycle between target forward and sampling."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
from vllm_ascend.ops.triton.mamba.gdn_recover import (
    gdn_commit_conv_history,
    gdn_recover_commit,
)

_MAX_SPEC_TOKENS = 5
_QWEN_GDN_TYPES = frozenset(("qwen3_5", "qwen3_5_text"))

if TYPE_CHECKING:
    from vllm_ascend.ops.triton.mamba.gdn_recover_batch import GDNRecoverBatchLayout


def validate_gdn_recover_config(config: VllmConfig) -> None:
    """Reject combinations whose state/commit ownership is not implemented."""
    model, cache, parallel = config.model_config, config.cache_config, config.parallel_config
    spec = config.speculative_config
    if get_current_hardware_profile().device_adaptor_family is not DeviceAdaptorFamily.STANDARD:
        raise ValueError("enable_gdn_recover currently requires an A2/A3 NPU")
    if model.dtype != torch.bfloat16 or cache.mamba_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("enable_gdn_recover requires BF16 model and conv cache")
    if spec is None or not 1 <= spec.num_speculative_tokens <= _MAX_SPEC_TOKENS:
        raise ValueError("enable_gdn_recover requires MTP with 1..5 speculative tokens")
    if spec.method not in ("mtp", "qwen3_5_mtp") or not spec.enforce_eager:
        raise ValueError("enable_gdn_recover requires an eager native MTP drafter")
    if model.hf_text_config.model_type not in _QWEN_GDN_TYPES:
        raise ValueError("enable_gdn_recover currently supports dense Qwen3.5-family GDN targets")
    if config.use_v2_model_runner or not model.enforce_eager:
        raise ValueError("enable_gdn_recover requires eager Model Runner V1")
    if cache.enable_prefix_caching or cache.mamba_cache_mode != "none":
        raise ValueError("enable_gdn_recover requires prefix caching off and mamba_cache_mode=none")
    if cache.mamba_ssm_cache_dtype != "float32":
        raise ValueError("enable_gdn_recover requires mamba_ssm_cache_dtype=float32")
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, field) != 1:
            raise ValueError(f"enable_gdn_recover requires {field}=1")
    if config.kv_transfer_config is not None or getattr(model, "enable_sleep_mode", False):
        raise ValueError("enable_gdn_recover does not support KV transfer or sleep mode")


@dataclass
class GDNRecoverEntry:
    prefix: str
    cache: tuple[torch.Tensor, ...]
    indices: torch.Tensor
    offsets: torch.Tensor
    request_rows: torch.Tensor


class GDNRecoverContext:
    """Runner-owned pending work; never retained in a process-wide registry.

    Metadata is snapshotted once per KV group by its builder. Sampling uses
    that forward's row mapping even if a proposer prepares other metadata.
    The completion event orders the next forward and block reuse after both
    SSM and conv commits. Initialization clears layouts before cache rebinding.
    """

    def __init__(self) -> None:
        self.active = False
        self.pending: list[GDNRecoverEntry] = []
        self._prefixes: set[str] = set()
        self._completion: Any = None
        self._layout: GDNRecoverBatchLayout | None = None
        self._layout_key: tuple | None = None

    def begin(self) -> None:
        if self.pending:
            raise RuntimeError("Previous GDN verify was not committed before the next forward")
        if self._completion is not None:
            torch.npu.current_stream().wait_event(self._completion)
        self.active = True

    def reset(self) -> None:
        if self.pending:
            raise RuntimeError("Cannot rebind GDN caches with an outstanding verify")
        if self._completion is not None:
            self._completion.synchronize()
        self.active = False
        self._layout = None
        self._layout_key = None
        self._completion = None

    def enqueue(self, prefix: str, cache: tuple[torch.Tensor, ...], metadata) -> None:
        if not self.active:
            return
        if prefix in self._prefixes:
            raise RuntimeError(f"GDN layer verified twice before sampling: {prefix}")
        indices, offsets, rows = (
            metadata.recover_state_indices,
            metadata.recover_query_offsets,
            metadata.recover_request_rows,
        )
        if indices is None or offsets is None or rows is None:
            raise RuntimeError("Missing GDN recovery metadata for speculative rows")
        self.pending.append(GDNRecoverEntry(prefix, tuple(cache), indices, offsets, rows))
        self._prefixes.add(prefix)

    @torch.no_grad()
    def commit(self, sampled_token_ids: torch.Tensor) -> None:
        if not self.pending:
            if self._completion is not None:
                torch.npu.current_stream().wait_event(self._completion)
            self.active = False
            return
        # Accepted drafts + sampled bonus count equals the processed mandatory
        # input + accepted draft count. The bonus itself is not replayed.
        counts = (sampled_token_ids != -1).sum(dim=1, dtype=torch.int32)
        selected_counts = {}
        entries = []
        for entry in self.pending:
            key = entry.request_rows.data_ptr()
            if key not in selected_counts:
                selected_counts[key] = counts.index_select(0, entry.request_rows)
            entries.append((entry.cache, entry.indices, entry.offsets, selected_counts[key]))
        # The fused kernel is optional for layouts. Correctness falls back to
        # the same public per-layer primitives for heterogeneous allocations.
        from vllm_ascend.ops.triton.mamba.gdn_recover_batch import GDNRecoverBatchLayout

        # Cache rebinding calls reset(); field identities are stable between
        # reinitializations, so geometry/storage checks stay on the cold path.
        layout_key = tuple(tuple(id(t) for t in entry.cache) for entry in self.pending)
        if layout_key != self._layout_key:
            self._layout = GDNRecoverBatchLayout.try_create([e[0] for e in entries])
            self._layout_key = layout_key
        if self._layout is not None:
            self._layout.commit(entries)
        else:
            for cache, indices, offsets, prefixes in entries:
                gdn_recover_commit(cache[1], cache[2], indices, offsets, prefixes)
                gdn_commit_conv_history(cache[0], cache[3], indices, offsets, prefixes)
        if self._completion is None:
            self._completion = torch.npu.Event()
        self._completion.record()
        self.pending.clear()
        self._prefixes.clear()
        self.active = False
