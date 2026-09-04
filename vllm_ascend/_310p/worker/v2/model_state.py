# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""MRV2 model state for Ascend 310P (dense/VL + hybrid/GDN)."""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend._310p.ops.rotary_embedding import prepare_mrope_cos_sin_slices_from_runner
from vllm_ascend._310p.worker.v2.rope import Ascend310PRopeState, get_310p_rope_state
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

from .rejection_sampler import RejectionSampler310V2
from .sampler import Ascend310PSampler


class _Ascend310PModelStateMixin:
    """310P RoPE / FULL-graph seq_lens helpers shared by dense and hybrid.

    Attribute annotations below are provided at runtime by AscendModelState /
    DefaultModelState (or set in the concrete subclass ``__init__``). Declared
    here so mypy can type-check the mixin in isolation.
    """

    model_config: Any
    model: nn.Module
    max_num_reqs: int
    max_num_tokens: int
    max_model_len: int
    device: torch.device
    rope_state: Any
    mm_pruner: Any
    _capture_seq_lens_by_ptr: dict[int, torch.Tensor]

    def _replace_310p_rope_state(self, encoder_cache: EncoderCache | None) -> None:
        del encoder_cache  # EVS / mm_pruner unsupported on 310P (see review notes).
        self.rope_state = get_310p_rope_state(
            self.model_config,
            self.model,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            self.device,
        )
        # Clear any parent-created EVS pruner: 310P MRv1 has no EVS path, and
        # Ascend310PRopeState lacks the read/update_prefill_positions EVS needs.
        self.mm_pruner = None

    def _record_capture_seq_lens(self, seq_lens: torch.Tensor) -> None:
        """Record the largest captured view for each physical buffer."""
        data_ptr = seq_lens.data_ptr()
        recorded = self._capture_seq_lens_by_ptr.get(data_ptr)
        if recorded is None or seq_lens.numel() > recorded.numel():
            self._capture_seq_lens_by_ptr[data_ptr] = seq_lens

    def _refresh_capture_seq_lens(self, runtime_seq_lens: torch.Tensor) -> None:
        """Copy runtime lengths to buffers read by FULL graph replay."""
        for capture_seq_lens in self._capture_seq_lens_by_ptr.values():
            num_seq_lens = min(capture_seq_lens.numel(), runtime_seq_lens.numel())
            capture_seq_lens[:num_seq_lens].copy_(runtime_seq_lens[:num_seq_lens], non_blocking=True)
            if num_seq_lens < capture_seq_lens.numel():
                capture_seq_lens[num_seq_lens:].zero_()

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if for_capture:
            self._record_capture_seq_lens(input_batch.seq_lens)
        elif cudagraph_mode == CUDAGraphMode.FULL:
            # Updating only input_batch.seq_lens is insufficient when replay is
            # bound to a different capture-time address.
            self._refresh_capture_seq_lens(input_batch.seq_lens)

        # Mixin sits before AscendModelState / AscendMambaHybridModelState in MRO.
        return super().prepare_attn(  # type: ignore[misc]
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states):
        if self.rope_state is None:
            return super().prepare_inputs(input_batch, req_states)  # type: ignore[misc]

        assert isinstance(self.rope_state, Ascend310PRopeState)
        # Upstream RopeState.prepare_positions uses Triton; 310P builds positions
        # on CPU from staged prefill tables, then H2D copies.
        self.rope_state.prepare_positions_cpu(
            input_batch.idx_mapping_np,
            input_batch.query_start_loc_np,
            req_states.prefill_len.np,
            req_states.num_computed_tokens_np,
            input_batch.num_tokens_after_padding,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        if self.model_config.uses_mrope:
            prepare_mrope_cos_sin_slices_from_runner(self, positions)
        return {"positions": positions}

    def custom_sampler(self, sampler):
        del sampler
        base_sampler = Ascend310PSampler(self.max_num_reqs, self.device)
        vllm_config = getattr(self, "vllm_config", None)
        spec_config = None if vllm_config is None else vllm_config.speculative_config
        if spec_config is None:
            return base_sampler, None
        if spec_config.method != "mtp":
            raise NotImplementedError(f"310P MRv2 only supports MTP speculative decoding, got {spec_config.method!r}.")
        return base_sampler, RejectionSampler310V2(base_sampler, spec_config, self.device)


class Ascend310PModelState(_Ascend310PModelStateMixin, AscendModelState):
    """Model state with Triton-free 310P sampler / MRoPE and encoder support."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Initialize the full Ascend/DefaultModelState contract first so
        # attributes such as ``prompt_embeds_state`` / ``encoder_runner`` exist,
        # then swap RoPE to the Triton-free 310P implementation.
        # follow-imports=skip hides DefaultModelState.__init__; ignore call-arg.
        AscendModelState.__init__(  # type: ignore[call-arg]
            self, vllm_config, model, encoder_cache, device
        )
        # ACLGraph replays the tensor addresses bound during capture. Keep every
        # captured seq_lens buffer so its contents can be refreshed before replay.
        self._capture_seq_lens_by_ptr = {}
        self._replace_310p_rope_state(encoder_cache)


class Ascend310PMambaHybridModelState(_Ascend310PModelStateMixin, AscendMambaHybridModelState):
    """310P hybrid/GDN state: keep Ascend hybrid contract, swap Triton RoPE."""

    def add_request(self, req_index: int, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        if not self._align_mode:
            return
        # Seed running-state block from resumed prefix length. Use the Mamba
        # page size (``mamba_block_size`` when APC+align is enabled) so the
        # index matches ``mamba_get_block_table_tensor`` / ``MAMBA_BLOCK_SIZE``.
        block_size = self.cache_config.mamba_block_size or self.cache_config.block_size
        self._mamba_state_idx_gpu[req_index].fill_((new_req_data.num_computed_tokens - 1) // block_size)

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Initialize the complete upstream/Ascend hybrid contract first (e.g.
        # ``_align_mode`` / mamba metadata), then replace Triton RoPE.
        AscendMambaHybridModelState.__init__(  # type: ignore[call-arg]
            self, vllm_config, model, encoder_cache, device
        )
        self._capture_seq_lens_by_ptr = {}
        self._replace_310p_rope_state(encoder_cache)
        self._align_postprocess_block_tables: tuple[torch.Tensor, ...] | None = None
        self._align_postprocess_kv_cache_config: KVCacheConfig | None = None

    def preprocess_state(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Triton-free align preprocess + precopy for 310P prefix caching.

        Upstream MRv2 uses ``preprocess_mamba_align_fused_kernel[grid]`` which is
        unavailable without Triton. Mirror the same semantics with torch ops and
        tensor-view copies (same approach as MRv1 310P mamba utils fallback).
        """
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return

        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        block_size = int(mamba_spec.block_size)
        idx_mapping = input_batch.idx_mapping[:num_reqs].to(dtype=torch.long)
        valid = idx_mapping >= 0
        if not bool(valid.any().item()):
            return
        req_indices = idx_mapping.masked_select(valid)

        state_idx = self._mamba_state_idx_gpu.index_select(0, req_indices)
        num_accepted = self.num_accepted_tokens_gpu.index_select(0, req_indices)
        src_off = torch.maximum(num_accepted - 1, torch.zeros_like(num_accepted))
        self._mamba_src_col_gpu.index_copy_(0, req_indices, state_idx)
        self._mamba_src_off_gpu.index_copy_(0, req_indices, src_off.to(self._mamba_src_off_gpu.dtype))

        num_computed = num_computed_tokens.index_select(0, req_indices)
        query_start_loc = input_batch.query_start_loc[: num_reqs + 1]
        # query_start_loc is in batch order; align with valid batch rows.
        batch_rows = torch.arange(num_reqs, device=idx_mapping.device, dtype=torch.long)
        batch_rows = batch_rows.masked_select(valid)
        query_lens = query_start_loc.index_select(0, batch_rows + 1) - query_start_loc.index_select(0, batch_rows)
        computed_after = num_computed + query_lens.to(num_computed.dtype)
        new_state_idx = (computed_after + block_size - 1) // block_size - 1
        new_state_idx = new_state_idx.to(dtype=self._mamba_state_idx_gpu.dtype)
        self._mamba_state_idx_gpu.index_copy_(0, req_indices, new_state_idx)

        should_reset = (state_idx >= 0) & (state_idx != new_state_idx)
        reset_indices = req_indices.masked_select(should_reset)
        if reset_indices.numel() > 0:
            DeviceOperator.index_fill(self.num_accepted_tokens_gpu, 0, reset_indices, 1)

        self._align_postprocess_block_tables = block_tables
        self._align_postprocess_kv_cache_config = kv_cache_config

        self._precopy_mamba_align_torch(
            input_batch=input_batch,
            block_tables=block_tables,
            kv_cache_config=kv_cache_config,
            mamba_group_ids=mamba_group_ids,
            num_reqs=num_reqs,
        )

    def _precopy_mamba_align_torch(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        num_reqs: int,
    ) -> None:
        """Copy mamba state across block boundaries without Triton."""
        from vllm_ascend.patch.worker.patch_mamba_utils import _tensor_view_from_data_ptr

        forward_context = self.vllm_config.compilation_config.static_forward_context
        copy_funcs = self.model.get_mamba_state_copy_func()
        for batch_i in range(num_reqs):
            req_idx = int(input_batch.idx_mapping[batch_i].item())
            if req_idx < 0:
                continue
            src_col = int(self._mamba_src_col_gpu[req_idx].item())
            dst_col = int(self._mamba_state_idx_gpu[req_idx].item())
            if src_col < 0 or src_col == dst_col:
                continue
            token_bias = int(self._mamba_src_off_gpu[req_idx].item())
            for group_id in mamba_group_ids:
                block_ids = block_tables[group_id][batch_i].detach().to("cpu").tolist()
                # Drop padded / unused slots.
                while block_ids and block_ids[-1] < 0:
                    block_ids.pop()
                if not block_ids or src_col >= len(block_ids) or dst_col >= len(block_ids):
                    continue
                dest_block_id = block_ids[dst_col]
                layer_names = kv_cache_config.kv_cache_groups[group_id].layer_names
                for layer_name in layer_names:
                    attention = forward_context[layer_name]
                    kv_caches = attention.kv_cache
                    for state, state_copy_func in zip(kv_caches, copy_funcs):
                        copy_spec = state_copy_func(state, block_ids, src_col, token_bias + 1)
                        src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                        dst_state = _tensor_view_from_data_ptr(
                            state, state[dest_block_id].data_ptr(), copy_spec.num_elements
                        )
                        dst_state.copy_(src_state.clone())

    def _postprocess_mamba_align_v2_torch(
        self,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_reqs: int,
    ) -> None:
        """MRv2 align postprocess without Triton (``PRECOMPUTED_NEW_COMPUTED`` path).

        Mirrors ``MambaSpecDecodeGPUContext.run_fused_postprocess_align`` and the
        MRv1 310P CPU fallback in ``patch_mamba_utils``.
        """
        from vllm_ascend.patch.worker.patch_mamba_utils import _tensor_view_from_data_ptr

        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        block_size = int(mamba_spec.block_size)
        forward_context = self.vllm_config.compilation_config.static_forward_context
        copy_funcs = self.model.get_mamba_state_copy_func()

        for batch_i in range(num_reqs):
            req_idx = int(idx_mapping[batch_i].item())
            if req_idx < 0:
                continue
            num_accepted = int(self.num_accepted_tokens_gpu[req_idx].item())
            src_block_idx = int(self._mamba_state_idx_gpu[req_idx].item())
            new_num_computed = int(num_computed_tokens[req_idx].item())
            num_tokens_running_state = new_num_computed - num_accepted + 1
            aligned_new_computed = new_num_computed // block_size * block_size
            if aligned_new_computed < num_tokens_running_state:
                continue

            accept_token_bias = aligned_new_computed - num_tokens_running_state
            dest_block_idx = aligned_new_computed // block_size - 1
            if src_block_idx == dest_block_idx:
                self.num_accepted_tokens_gpu[req_idx] = 1
                if accept_token_bias == 0:
                    continue

            for group_id in mamba_group_ids:
                block_ids = block_tables[group_id][batch_i].detach().to("cpu").tolist()
                while block_ids and block_ids[-1] < 0:
                    block_ids.pop()
                if not block_ids or src_block_idx >= len(block_ids) or dest_block_idx >= len(block_ids):
                    continue
                dest_block_id = block_ids[dest_block_idx]
                layer_names = kv_cache_config.kv_cache_groups[group_id].layer_names
                for layer_name in layer_names:
                    attention = forward_context[layer_name]
                    kv_caches = attention.kv_cache
                    for state, state_copy_func in zip(kv_caches, copy_funcs):
                        copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)
                        src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                        dst_state = _tensor_view_from_data_ptr(
                            state, state[dest_block_id].data_ptr(), copy_spec.num_elements
                        )
                        dst_state.copy_(src_state.clone())

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # Upstream uses Triton scatter kernels. On 310P the decorated kernel is
        # unusable; keep the op Triton-free via NPU indexing. Filter padding
        # ``-1`` indices: ``index_fill_`` treats ``-1`` as the last slot.
        valid = idx_mapping >= 0
        valid_indices = idx_mapping.masked_select(valid).to(dtype=torch.long)
        if valid_indices.numel() == 0:
            return

        if isinstance(num_sampled, int):
            DeviceOperator.index_fill(self.num_accepted_tokens_gpu, 0, valid_indices, max(num_sampled, 1))
        else:
            accepted = torch.clamp(num_sampled.masked_select(valid), min=1).to(self.num_accepted_tokens_gpu.dtype)
            self.num_accepted_tokens_gpu.index_copy_(0, valid_indices, accepted)

        if (
            not getattr(self, "_align_mode", False)
            or num_computed_tokens is None
            or self._align_postprocess_block_tables is None
            or self._align_postprocess_kv_cache_config is None
        ):
            return

        num_reqs = idx_mapping.shape[0]
        self._postprocess_mamba_align_v2_torch(
            idx_mapping=idx_mapping,
            num_computed_tokens=num_computed_tokens,
            block_tables=self._align_postprocess_block_tables,
            kv_cache_config=self._align_postprocess_kv_cache_config,
            num_reqs=num_reqs,
        )
