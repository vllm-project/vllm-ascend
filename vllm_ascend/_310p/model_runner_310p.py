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

import numpy as np
import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUModelRunner310(NPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acl_format = ACL_FORMAT_FRACTAL_NZ

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # 310P limitation: KV transfer is not supported.
        if self.vllm_config.kv_transfer_config is not None:
            raise ValueError("KV cache transfer is not supported for 310P.")
        if self.use_sparse:
            raise ValueError("Deepseek Sparse Attention is not supported for 310P.")
        if self.model_config.use_mla:
            raise ValueError("MLAAttention is not supported for 310P.")
        # Initialize the memory size for KV cache
        kv_cache_size = self._calculate_kv_cache_tensors_size(kv_cache_config)
        # Allocate and reshape KV cache Tensors
        kv_caches = self._allocate_kv_cache_and_reshape_tensors(kv_cache_config, kv_cache_size)
        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches)
        return kv_caches

    def _calculate_kv_cache_tensors_size(self, kv_cache_config: KVCacheConfig) -> dict[str, int]:
        """
        Initializes the KV cache size. The buffer needs to be reshaped to the desired shape before being used by
        the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, int]: A map between layer names to their
            corresponding memory buffer size.
        """
        # init kv cache tensors
        kv_cache_sizes: dict[str, int] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # TODO: REFACTOR ME to sharing hybrid cache
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if "linear_attn" in layer_name and layer_name not in kv_cache_sizes:
                    # for mamba linear attention
                    kv_cache_size = kv_cache_tensor.size
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "linear_attn" in layer_name_inner:
                            kv_cache_sizes[layer_name_inner] = kv_cache_size
                elif "attn" in layer_name and layer_name not in kv_cache_sizes:
                    kv_tensor_split_factor = 2
                    kv_tensor_size = int(kv_cache_tensor.size // kv_tensor_split_factor)
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            kv_cache_sizes[layer_name_inner] = kv_tensor_size

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_sizes.keys()), "Some layers are not correctly initialized"

        return kv_cache_sizes

    def _allocate_kv_cache_and_reshape_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_sizes: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """
        Allocate the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_sizes: The KV cache size of each layer
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_tensor_size = kv_cache_sizes[layer_name]
                    assert kv_tensor_size is not None
                    sum_page_size_bytes = kv_tensor_size * 2
                    assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    if hasattr(attn_backend, "get_supported_kernel_block_sizes") and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_kernel_block_sizes()[0]

                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                        )
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size
                        )
                    dtype = kv_cache_spec.dtype
                    k_shape = kv_cache_shape[1:]
                    v_shape = k_shape
                    k_cache = torch_npu.empty_with_format(
                        size=k_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=v_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    tensor_size = kv_cache_sizes[layer_name]
                    dtype = kv_cache_spec.dtype
                    tensor_element_size = torch.tensor([], dtype=dtype).element_size()
                    raw_tensor = torch.zeros(tensor_size // tensor_element_size, dtype=dtype, device=self.device)
                    assert tensor_size is not None
                    assert tensor_size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = tensor_size // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)

                        target_idx += torch.prod(torch.tensor(target_shape)).item()
                        tensor = raw_tensor[start_idx:target_idx].view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    # Override this function because of tensor.copy_(other) accuracy issue.
    # TODO: This override will be removed after tensor.copy_(other) accuracy issue is resolved.
    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the NPU from prev_sampled_token_ids.
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
        num_commmon_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_commmon_tokens < total_without_spec:
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # NOTE: Override the copy_ function here
            indices = torch.arange(num_commmon_tokens, device=self.input_ids.gpu.device)
            source = self.input_batch.prev_sampled_token_ids[:num_commmon_tokens, 0]
            self.input_ids.gpu.index_copy_(0, indices, source)
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_commmon_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
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
        # Scatter the draft tokens after the sampled tokens are scattered.
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
