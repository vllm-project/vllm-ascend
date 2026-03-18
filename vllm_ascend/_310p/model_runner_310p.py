#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

_NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN = 1
PAGED_ATTENTION_SPLIT_BLOCK_SIZE_310P = 64
PAGED_ATTENTION_HEAD_BLOCK_PRODUCT_LIMIT_310P = 128 * 128


def get_310p_attention_kernel_block_sizes(
    *,
    block_size: int,
    head_size: int,
    backend_block_sizes: list[int] | None,
    use_hybrid_blocks: bool,
) -> list[int]:
    """Select logical kernel block sizes for 310P paged attention."""
    candidate_sizes: list[int] = []
    if (
        head_size * block_size > PAGED_ATTENTION_HEAD_BLOCK_PRODUCT_LIMIT_310P
        and block_size % PAGED_ATTENTION_SPLIT_BLOCK_SIZE_310P == 0
    ):
        candidate_sizes.append(PAGED_ATTENTION_SPLIT_BLOCK_SIZE_310P)
    if use_hybrid_blocks and backend_block_sizes:
        candidate_sizes.extend(backend_block_sizes)
    candidate_sizes.append(block_size)

    kernel_block_sizes: list[int] = []
    for candidate_size in candidate_sizes:
        if candidate_size <= 0 or block_size % candidate_size != 0:
            continue
        if candidate_size not in kernel_block_sizes:
            kernel_block_sizes.append(candidate_size)

    return kernel_block_sizes or [block_size]


class NPUModelRunner310(NPUModelRunner):
    # Inherited from parent runner; annotated here to satisfy strict type checks.
    uniform_decode_query_len: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acl_format = ACL_FORMAT_FRACTAL_NZ
        if self.speculative_config is not None and self.speculative_config.method == "ngram":
            # 310P ngram requires decode-only graph shapes to be built with q_len=1.
            # Keep dispatcher's internal query_len in sync to avoid key-init assert.
            self.cudagraph_dispatcher.uniform_decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN

    @contextmanager
    def temporary_modify_uniform_decode_query_len(self):
        # This is only needed for the 310P ngram path where dispatcher uses q_len=1
        # while runner's default uniform_decode_query_len remains 1 + num_spec_tokens.
        # TODO: remove this temporary override after upstream supports independent
        # decode capture query_len for backend-specific paths.
        if self.speculative_config is None or self.speculative_config.method != "ngram":
            yield
            return

        original_uniform_decode_query_len = self.uniform_decode_query_len
        self.uniform_decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN
        try:
            yield
        finally:
            self.uniform_decode_query_len = original_uniform_decode_query_len

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ):
        if self.attn_state in (AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit):
            force_eager = True

        if force_uniform_decode is None and self.attn_state == AscendAttentionState.DecodeOnly:
            decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN
            if (
                max_num_scheduled_tokens == decode_query_len
                and num_tokens == max_num_scheduled_tokens * num_reqs
                and np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
            ):
                force_uniform_decode = True

        return super()._determine_batch_execution_and_padding(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=use_cascade_attn,
            allow_microbatching=allow_microbatching,
            force_eager=force_eager,
            force_uniform_decode=force_uniform_decode,
            force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs,
        )

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        # Keep this aligned with the dispatcher because batch_desc.num_reqs is
        # generated by dispatcher._create_padded_batch_descriptor().
        uniform_decode_query_len = self.cudagraph_dispatcher.uniform_decode_query_len

        if num_tokens_padded == num_reqs_padded * uniform_decode_query_len:
            assert num_reqs <= num_reqs_padded
            last_loc = self.query_start_loc.np[num_reqs]
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * uniform_decode_query_len + last_loc
            )
        else:
            assert num_reqs == num_reqs_padded
            self.query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        self.query_start_loc.copy_to_gpu()
        return num_reqs_padded

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode=None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ):
        temporary_context = self.temporary_modify_uniform_decode_query_len() if uniform_decode else nullcontext()
        with temporary_context:
            return super()._dummy_run(
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                force_attention=force_attention,
                uniform_decode=uniform_decode,
                is_profile=is_profile,
                create_mixed_batch=create_mixed_batch,
                allow_microbatching=allow_microbatching,
                skip_eplb=skip_eplb,
                remove_lora=remove_lora,
                is_graph_capturing=is_graph_capturing,
                num_active_loras=num_active_loras,
                profile_seq_lens=profile_seq_lens,
            )

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends,
        kv_cache_groups,
    ) -> None:
        with self.temporary_modify_uniform_decode_query_len():
            super()._check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)

    def _get_attention_kernel_block_sizes(
        self,
        kv_cache_spec: AttentionSpec,
        attn_backend=None,
    ) -> list[int]:
        backend_block_sizes = None
        if attn_backend is not None and hasattr(attn_backend, "get_supported_kernel_block_sizes"):
            backend_block_sizes = attn_backend.get_supported_kernel_block_sizes()
        kernel_block_sizes = get_310p_attention_kernel_block_sizes(
            block_size=kv_cache_spec.block_size,
            head_size=kv_cache_spec.head_size,
            backend_block_sizes=backend_block_sizes,
            use_hybrid_blocks=self.use_hybrid_blocks,
        )
        if kernel_block_sizes[0] != kv_cache_spec.block_size:
            logger.info_once(
                "310P paged attention uses split kernel blocks: physical_block_size=%d, "
                "kernel_block_size=%d, head_size=%d",
                kv_cache_spec.block_size,
                kernel_block_sizes[0],
                kv_cache_spec.head_size,
            )
        return kernel_block_sizes

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, torch.Tensor]:
        """Initialize the memory buffer for KV cache."""
        if self.vllm_config.kv_transfer_config is not None:
            raise ValueError("KV cache transfer is not supported for 310P.")
        if self.use_sparse:
            raise ValueError("Deepseek Sparse Attention is not supported for 310P.")
        if self.model_config.use_mla:
            raise ValueError("MLAAttention is not supported for 310P.")

        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)

        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches)
        return kv_caches

    def _allocate_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, torch.Tensor | int | None]:
        """
        Initialize KV cache raw buffers/sizes before reshape.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor | int | None] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for layer_name in kv_cache_tensor.shared_by:
                if "linear_attn" in layer_name and layer_name not in kv_cache_raw_tensors:
                    # Keep mamba cache as raw bytes, then reinterpret by spec dtypes
                    # during reshape. This matches MambaSpec byte-level contract.
                    tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        if "linear_attn" in layer_name_inner:
                            kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors:
                    # Attention cache allocates K/V separately in reshape stage.
                    kv_tensor_size = int(kv_cache_tensor.size // 2)
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            kv_cache_raw_tensors[layer_name_inner] = kv_tensor_size

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor | int | None],
    ) -> dict[str, torch.Tensor]:
        """Allocate KV cache tensors to backend-required shapes."""
        kv_caches: dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_tensor_size = kv_cache_raw_tensors[layer_name]
                    assert isinstance(kv_tensor_size, int)
                    sum_page_size_bytes = kv_tensor_size * 2
                    assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    kernel_block_sizes = self._get_attention_kernel_block_sizes(kv_cache_spec, attn_backend)
                    block_size = kernel_block_sizes[0]
                    if block_size != kv_cache_spec.block_size:
                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                        )
                    else:
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks,
                            kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                        )

                    dtype = kv_cache_spec.dtype
                    k_shape = kv_cache_shape[1:]
                    v_shape = k_shape
                    k_cache = torch_npu.empty_with_format(
                        size=k_shape,
                        dtype=dtype,
                        device=self.device,
                        acl_format=self._acl_format,
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=v_shape,
                        dtype=dtype,
                        device=self.device,
                        acl_format=self._acl_format,
                    )
                    kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert isinstance(raw_tensor, torch.Tensor)
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        target_shape = (num_blocks, *shape)
                        target_idx += math.prod(target_shape) * get_dtype_size(dtype)
                        tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_caches.keys()), "Some layers are not correctly initialized"

        return kv_caches

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            if isinstance(kv_cache_spec, AttentionSpec):
                try:
                    attn_groups = self.attn_groups[kv_cache_group_id]
                except IndexError:
                    attn_groups = None
                backend = attn_groups[0].backend if attn_groups else None
                kernel_block_sizes.append(self._get_attention_kernel_block_sizes(kv_cache_spec, backend))
            else:
                kernel_block_sizes.append([0])

        if block_sizes != [self.cache_config.block_size] or kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "
                "for more details."
            )
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config
                    else 0
                ),
                kernel_block_sizes=kernel_block_sizes,
            )

    # Override this function because of tensor.copy_(other) accuracy issue.
    # TODO: This override will be removed after tensor.copy_(other) accuracy issue is resolved.
    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch."""
        if self.input_batch.prev_sampled_token_ids is None:
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(range(flattened_index - draft_len + 1, flattened_index + 1))
                start = prev_index * self.num_spec_tokens
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_common_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_common_tokens < total_without_spec:
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens == 0:
            return
        if indices_match and max_flattened_index == (num_common_tokens - 1):
            indices = torch.arange(num_common_tokens, device=self.input_ids.gpu.device)
            source = self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0]
            self.input_ids.gpu.index_copy_(0, indices, source)
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_common_tokens] = True
            return

        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0],
        )
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _update_hybrid_attention_mamba_layout(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Keep attention KV cache contiguous on 310P.

        Interleaving attention+mamba storage makes key_cache/value_cache
        non-contiguous and leads to paged-attention setup failures.
        """
        logger.warning(
            "Skip _update_hybrid_attention_mamba_layout on 310P: "
            "keep attention KV cache contiguous for paged-attention."
        )
        return
