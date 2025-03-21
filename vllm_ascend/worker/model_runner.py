#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/model_runner.py
# Copyright 2023 The vLLM team.
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
#

import dataclasses
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Type, TypeVar, Union)

import torch
import torch.distributed
import torch.nn as nn
import torch_npu
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import VllmConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.distributed import get_kv_transfer_group, get_pp_group
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata, SamplingMetadataCache
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import (DeviceMemoryProfiler, PyObjectCache, flatten_2d_lists,
                        is_pin_memory_available)
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

TModelInputForNPU = TypeVar('TModelInputForNPU', bound="ModelInputForNPU")


@dataclass(frozen=True)
class ModelInputForNPU(ModelRunnerInputBase):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    token_types: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None
    finished_requests_ids: Optional[List[str]] = None
    virtual_engine: int = 0
    async_callback: Optional[Callable] = None
    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None
    scheduler_outputs: Optional[SchedulerOutputs] = None
    previous_hidden_states: Optional[torch.Tensor] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForNPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForNPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)

    # Exclude `async_callback` to be able to pickle this object
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["async_callback"]
        return state

    # TODO: What happens when we depickle this object?
    # How can we update this callback to properly pass it to the engine?
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.update({'async_callback': None})


@dataclass(frozen=True)
class ModelInputForNPUWithSamplingMetadata(ModelInputForNPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None
    # Used for speculative decoding. We do not broadcast it because it is only
    # used by the driver worker.
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForNPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForNPUBuilder(ModelRunnerInputBuilderBase[ModelInputForNPU]):
    """Build ModelInputForNPU from SequenceGroupMetadata."""

    # Note: ideally we would be using a dataclass(kw_only=True)
    # here, so that this can be subclassed easily,
    # but kw_only is not supported in python<3.10.
    class InterDataForSeqGroup:
        """Intermediate data for the current sequence group."""

        def simple_reinit(self):
            self.input_tokens[0].clear()  # type: ignore
            self.input_positions[0].clear()  # type: ignore
            self.token_types[0].clear()  # type: ignore
            self.mrope_input_positions = None  # type: ignore
            self.seq_lens[0] = 0  # type: ignore
            self.orig_seq_lens[0] = 0  # type: ignore
            self.query_lens[0] = 0  # type: ignore
            self.context_lens[0] = 0  # type: ignore
            self.curr_sliding_window_blocks[0] = 0  # type: ignore

        def __init__(
            self,
            *,
            # From sequence group metadata.
            request_id: str,
            seq_ids: List[int],
            is_prompt: bool,
            block_tables: Optional[Dict[int, List[int]]],
            computed_block_nums: List[int],
            n_seqs: int = 0,

            # Input tokens and positions.
            input_tokens: Optional[List[List[int]]] = None,
            input_positions: Optional[List[List[int]]] = None,
            token_types: Optional[List[List[int]]] = None,
            mrope_input_positions: Optional[List[List[List[int]]]] = None,

            # The sequence length (may be capped to the sliding window).
            seq_lens: Optional[List[int]] = None,
            # The original sequence length (before applying sliding window).
            # This is used to compute slot mapping.
            orig_seq_lens: Optional[List[int]] = None,
            # The query length.
            query_lens: Optional[List[int]] = None,
            # The number of tokens that are already computed.
            context_lens: Optional[List[int]] = None,
            # The current sliding window block.
            curr_sliding_window_blocks: Optional[List[int]] = None,

            # Multi-modal inputs.
            multi_modal_kwargs: Optional[MultiModalKwargs] = None,
            multi_modal_placeholder_maps: Optional[Dict[
                str, MultiModalPlaceholderMap]] = None,

            # Whether the prefix cache is hit (prefill only).
            prefix_cache_hit: bool = False,
            reinit: bool = False,
            reinit_use_defaults: bool = False,
            encoder_seq_len: int = 0,
        ):
            if reinit:
                assert len(self.seq_ids) == len(seq_ids)  # type: ignore
                for i, seq_id in enumerate(seq_ids):
                    self.seq_ids[i] = seq_id  # type: ignore
            else:
                self.seq_ids = seq_ids

            self.request_id = request_id
            self.is_prompt = is_prompt
            self.block_tables = block_tables
            self.computed_block_nums = computed_block_nums
            self.n_seqs = n_seqs
            self.encoder_seq_len = encoder_seq_len

            if reinit:
                if len(self.seq_ids) == 1 and reinit_use_defaults:
                    self.simple_reinit()
                else:
                    if input_tokens:
                        self.input_tokens = input_tokens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.input_tokens[seq_id].clear()

                    if input_positions:
                        self.input_positions = input_positions
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.input_positions[seq_id].clear()

                    if token_types:
                        self.token_types = token_types
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.token_types[seq_id].clear()

                    self.mrope_input_positions = None

                    if seq_lens:
                        self.seq_lens = seq_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.seq_lens[seq_id] = 0

                    if orig_seq_lens:
                        self.orig_seq_lens = orig_seq_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.orig_seq_lens[seq_id] = 0

                    if query_lens:
                        self.query_lens = query_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.query_lens[seq_id] = 0

                    if context_lens:
                        self.context_lens = context_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.context_lens[seq_id] = 0

                    if curr_sliding_window_blocks:
                        self.curr_sliding_window_blocks = \
                            curr_sliding_window_blocks
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.curr_sliding_window_blocks[seq_id] = 0

            else:
                self.input_tokens = input_tokens or []
                self.input_positions = input_positions or []
                self.token_types = token_types or []
                self.mrope_input_positions = mrope_input_positions or None
                self.seq_lens = seq_lens or []
                self.orig_seq_lens = orig_seq_lens or []
                self.query_lens = query_lens or []
                self.context_lens = context_lens or []
                self.curr_sliding_window_blocks = \
                    curr_sliding_window_blocks or []

            self.multi_modal_kwargs = multi_modal_kwargs
            self.multi_modal_placeholder_maps = multi_modal_placeholder_maps
            self.prefix_cache_hit = prefix_cache_hit

            self.n_seqs = len(self.seq_ids)

            if not reinit:
                self.__post_init__()

        def __post_init__(self):
            self.n_seqs = len(self.seq_ids)

            self.input_tokens = [[] for _ in range(self.n_seqs)]
            self.input_positions = [[] for _ in range(self.n_seqs)]
            self.token_types = [[] for _ in range(self.n_seqs)]
            self.mrope_input_positions = None
            self.seq_lens = [0] * self.n_seqs
            self.orig_seq_lens = [0] * self.n_seqs
            self.query_lens = [0] * self.n_seqs
            self.context_lens = [0] * self.n_seqs
            self.curr_sliding_window_blocks = [0] * self.n_seqs

    def __init__(self,
                 runner,
                 finished_requests_ids: Optional[List[str]] = None):
        super().__init__()
        # Compute functions for each sequence in a sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_compute_fns = [
            self._compute_lens,
            self._compute_for_prefix_cache_hit,
            self._compute_for_sliding_window,
        ]
        # Compute functions for each sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_group_compute_fns = [
            self._compute_multi_modal_input,
        ]

        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.scheduler_config = self.runner.scheduler_config
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.multi_modal_input_mapper = self.runner.multi_modal_input_mapper
        self.finished_requests_ids = finished_requests_ids
        self.decode_only = True
        self.is_encoder_decoder = self.runner.model_config.is_encoder_decoder

        # Attention metadata inputs.
        self.attn_metadata_builder = self.attn_backend.make_metadata_builder(
            weakref.proxy(self))

        # Engine/Model configurations.
        self.chunked_prefill_enabled = (
            self.scheduler_config is not None
            and self.scheduler_config.chunked_prefill_enabled)
        if self.sliding_window is not None:
            self.sliding_window_blocks = (
                self.sliding_window + self.block_size - 1) // self.block_size
            self.block_aligned_sliding_window = \
                self.sliding_window_blocks * self.block_size

    def prepare(self,
                finished_requests_ids: Optional[List[str]] = None) -> None:
        self.finished_requests_ids = finished_requests_ids

        # if the current batch is decode-only.
        # will be set to False if there is any non-decode request.
        self.decode_only = True

        # Intermediate data (data in CPU before going to NPU) for
        # the current sequence group.
        self.inter_data_list: List[
            ModelInputForNPUBuilder.InterDataForSeqGroup] = []

        self.attn_metadata_builder.prepare()

    def gen_inter_data_builder(self, num_seqs: int):
        return lambda: ModelInputForNPUBuilder.InterDataForSeqGroup(
            request_id="",
            seq_ids=[0] * num_seqs,
            is_prompt=True,
            block_tables=None,
            computed_block_nums=[])

    def init_cached_inter_data(self, *args, **kwargs):
        assert len(args) == 0
        assert "seq_ids" in kwargs
        seq_ids = kwargs["seq_ids"]
        num_seqs = len(seq_ids)

        # The inter-data cache is per model_runner
        inter_data_cache = self.runner.inter_data_cache
        if num_seqs not in inter_data_cache:
            inter_data_cache[num_seqs] = PyObjectCache(
                self.gen_inter_data_builder(num_seqs))

        obj = inter_data_cache[num_seqs].get_object()
        obj.__init__(*args, **kwargs)
        return obj

    def reset_cached_inter_data(self):
        for cache in self.runner.inter_data_cache.values():
            cache.reset()

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        """Add a sequence group to the builder."""
        seq_ids = seq_group_metadata.seq_data.keys()
        n_seqs = len(seq_ids)
        is_prompt = seq_group_metadata.is_prompt

        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False

        encoder_seq_len = 0

        if self.is_encoder_decoder:
            encoder_seq_len = seq_group_metadata.encoder_seq_data.get_len()

        inter_data = self.init_cached_inter_data(
            request_id=seq_group_metadata.request_id,
            seq_ids=seq_ids,
            is_prompt=is_prompt,
            block_tables=seq_group_metadata.block_tables,
            computed_block_nums=seq_group_metadata.computed_block_nums,
            reinit=True,
            reinit_use_defaults=True,
            encoder_seq_len=encoder_seq_len)

        self.inter_data_list.append(inter_data)

        for seq_idx in range(n_seqs):
            for per_seq_fn in self.per_seq_compute_fns:
                per_seq_fn(inter_data, seq_idx, seq_group_metadata)
        for per_seq_group_fn in self.per_seq_group_compute_fns:
            per_seq_group_fn(inter_data, seq_group_metadata)

    def build(self) -> ModelInputForNPU:
        """Finalize the builder intermediate data and
        create on-device tensors.
        """
        # Combine and flatten intermediate data.
        input_tokens = [
            flatten_2d_lists(inter_data.input_tokens)
            for inter_data in self.inter_data_list
        ]
        if not input_tokens:
            # This may happen when all prefill requests hit
            # prefix caching and there is no decode request.
            return self.model_input_cls()

        mrope_input_positions: Optional[List[List[int]]] = None
        if any(inter_data.mrope_input_positions is not None
               for inter_data in self.inter_data_list):
            mrope_input_positions = [[] for _ in range(3)]

            for idx in range(3):
                for inter_data in self.inter_data_list:
                    msections = inter_data.mrope_input_positions
                    if msections is None:
                        for _seq_input_positions in inter_data.input_positions:
                            mrope_input_positions[idx].extend(
                                _seq_input_positions)
                    else:
                        for _seq_mrope_input_positions in msections:
                            mrope_input_positions[idx].extend(
                                _seq_mrope_input_positions[idx])
            input_positions = None
        else:
            input_positions = [
                flatten_2d_lists(inter_data.input_positions)
                for inter_data in self.inter_data_list
            ]

        seq_lens = []
        max_decode_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len,
                                         max(inter_data.seq_lens))
        query_lens = flatten_2d_lists(
            [inter_data.query_lens for inter_data in self.inter_data_list])
        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        request_ids_to_seq_ids = {
            data.request_id: data.seq_ids
            for data in self.inter_data_list
        }

        input_tokens_tensor = torch.tensor(flatten_2d_lists(input_tokens),
                                           dtype=torch.long,
                                           device=self.runner.device)
        if mrope_input_positions is not None:
            input_positions_tensor = torch.tensor(mrope_input_positions,
                                                  dtype=torch.long,
                                                  device=self.runner.device)
        else:
            input_positions_tensor = torch.tensor(
                flatten_2d_lists(input_positions),
                dtype=torch.long,
                device=self.runner.device)

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(seq_lens, query_lens)

        # Multi-modal data.
        multi_modal_kwargs_list = [
            data.multi_modal_kwargs for data in self.inter_data_list
            if data.multi_modal_kwargs is not None
        ]
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            multi_modal_kwargs=multi_modal_kwargs,
            request_ids_to_seq_ids=request_ids_to_seq_ids,
            finished_requests_ids=self.finished_requests_ids)

    def _compute_lens(self, inter_data: InterDataForSeqGroup, seq_idx: int,
                      seq_group_metadata: SequenceGroupMetadata):
        """Compute context length, sequence length and tokens
        for the given sequence data.
        """
        seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
        token_chunk_size = seq_group_metadata.token_chunk_size

        # Compute context length (the number of tokens that are
        # already computed) and sequence length (total number of tokens).

        seq_len = seq_data.get_len()
        if inter_data.is_prompt:
            context_len = seq_data.get_num_computed_tokens()
            seq_len = min(seq_len, context_len + token_chunk_size)
        elif self.runner.scheduler_config.is_multi_step or \
            self.is_encoder_decoder:
            context_len = seq_len - 1
        else:
            context_len = seq_data.get_num_computed_tokens()

        # Compute tokens.
        tokens = seq_data.get_token_ids()[context_len:seq_len]
        token_types = seq_group_metadata.token_type_ids

        inter_data.seq_lens[seq_idx] = seq_len
        inter_data.orig_seq_lens[seq_idx] = seq_len
        inter_data.context_lens[seq_idx] = context_len
        inter_data.input_tokens[seq_idx].extend(tokens)
        inter_data.input_positions[seq_idx].extend(range(context_len, seq_len))
        inter_data.token_types[seq_idx].extend(
            token_types if token_types else [])
        inter_data.query_lens[seq_idx] = seq_len - context_len

        if seq_data.mrope_position_delta is not None:
            if inter_data.mrope_input_positions is None:
                inter_data.mrope_input_positions = [None] * inter_data.n_seqs

            inter_data.mrope_input_positions[
                seq_idx] = MRotaryEmbedding.get_next_input_positions(
                    seq_data.mrope_position_delta,
                    context_len,
                    seq_len,
                )

    def _compute_for_prefix_cache_hit(
            self, inter_data: InterDataForSeqGroup, seq_idx: int,
            seq_group_metadata: SequenceGroupMetadata):
        """Check if hit prefix cache (i.e., some blocks are already computed).
        If hit, update input tokens and positions to only compute the
        remaining blocks.
        """
        computed_block_nums = inter_data.computed_block_nums

        # Note that prefix caching does not support sliding window.
        prefix_cache_hit = (computed_block_nums is not None
                            and len(computed_block_nums) > 0
                            and self.sliding_window is None
                            and inter_data.is_prompt)
        inter_data.prefix_cache_hit = prefix_cache_hit

        if not prefix_cache_hit:
            return

        assert computed_block_nums is not None
        # The cache hit prompt tokens in this sequence. Note that
        # this may be larger than the sequence length if chunked
        # prefill is enabled.
        prefix_cache_len = len(computed_block_nums) * self.block_size
        seq_group_metadata.seq_data[inter_data.seq_ids[
            seq_idx]].update_num_cached_tokens(prefix_cache_len)

        # The number of so far computed prompt tokens in this sequence.
        context_len = inter_data.context_lens[seq_idx]
        # The total number of prompt tokens in this sequence.
        # When chunked prefill is enabled, this is the token number of
        # computed chunks + current chunk.
        seq_len = inter_data.seq_lens[seq_idx]
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            pass
        elif context_len < prefix_cache_len < seq_len:
            # Partial hit. Compute the missing part.
            uncomputed_start = prefix_cache_len - context_len
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[
                seq_idx][uncomputed_start:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][uncomputed_start:]
            inter_data.token_types[seq_idx] = inter_data.token_types[seq_idx][
                uncomputed_start:]
            context_len = prefix_cache_len

            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[
                seq_idx] = inter_data.seq_lens[seq_idx] - context_len
        elif seq_len <= prefix_cache_len:
            # Full hit. Only compute the last token to avoid
            # erroneous behavior. FIXME: Ideally we should directly
            # mark all tokens as computed in the scheduler and do not
            # schedule this sequence, so this case should not happen.
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[
                seq_idx][-1:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][-1:]
            inter_data.token_types[seq_idx] = inter_data.token_types[seq_idx][
                -1:]
            inter_data.query_lens[seq_idx] = 1
            inter_data.context_lens[seq_idx] = inter_data.seq_lens[seq_idx] - 1

    def _compute_for_sliding_window(self, inter_data: InterDataForSeqGroup,
                                    seq_idx: int,
                                    seq_group_metadata: SequenceGroupMetadata):
        """Update seq_len and curr_sliding_window_block for the given
        sequence data (only required by decoding) if sliding window is enabled.
        """
        curr_sliding_window_block = 0
        sliding_seq_len = inter_data.seq_lens[seq_idx]
        if not inter_data.is_prompt and self.sliding_window is not None:
            # TODO(sang): This is a hack to make sliding window work with
            # paged attn. We can remove it if we make paged attn kernel
            # to properly handle slinding window attn.
            curr_sliding_window_block = self.sliding_window_blocks
            # number of elements in last block
            suff_len = inter_data.seq_lens[seq_idx] % self.block_size
            sliding_seq_len = min(inter_data.seq_lens[seq_idx],
                                  self.block_aligned_sliding_window + suff_len)
            if suff_len > 0:
                curr_sliding_window_block += 1

        inter_data.curr_sliding_window_blocks[
            seq_idx] = curr_sliding_window_block
        inter_data.seq_lens[seq_idx] = sliding_seq_len

    def _compute_multi_modal_input(self, inter_data: InterDataForSeqGroup,
                                   seq_group_metadata: SequenceGroupMetadata):
        """If multi-modal data is given, add it to the input."""
        # NOTE: mm_data only includes the subset of multi-modal items that
        # intersect with the current prefill positions.
        positions = inter_data.input_positions[0]
        mm_data, placeholder_maps = MultiModalPlaceholderMap.from_seq_group(
            seq_group_metadata,
            range(positions[0], positions[0] + len(positions)))
        if not mm_data:
            return

        if self.runner.mm_registry.has_processor(self.runner.model_config):
            mm_kwargs = mm_data
        else:
            mm_kwargs = self.multi_modal_input_mapper(
                mm_data,
                seq_group_metadata.mm_processor_kwargs,
            )

        inter_data.multi_modal_kwargs = mm_kwargs
        inter_data.multi_modal_placeholder_maps = placeholder_maps

        # special processing for mrope position deltas.
        if self.runner.model_config.uses_mrope:
            image_grid_thw = mm_kwargs.get("image_grid_thw", None)
            video_grid_thw = mm_kwargs.get("video_grid_thw", None)
            assert image_grid_thw is not None or video_grid_thw is not None, (
                "mrope embedding type requires multi-modal input mapper "
                "returns 'image_grid_thw' or 'video_grid_thw'.")
            second_per_grid_ts = mm_kwargs.get("second_per_grid_ts", None)

            hf_config = self.runner.model_config.hf_config

            inter_data.mrope_input_positions = [None] * inter_data.n_seqs
            for seq_idx in range(inter_data.n_seqs):
                seq_data = seq_group_metadata.seq_data[
                    inter_data.seq_ids[seq_idx]]
                token_ids = seq_data.get_token_ids()

                mrope_input_positions, mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions(
                        token_ids,
                        hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        context_len=inter_data.context_lens[seq_idx],
                        seq_len=inter_data.seq_lens[seq_idx],
                    )

                seq_data.mrope_position_delta = mrope_position_delta
                inter_data.mrope_input_positions[
                    seq_idx] = mrope_input_positions


class NPUModelRunnerBase(ModelRunnerBase[TModelInputForNPU]):
    """
    Helper class for shared methods between NPU model runners.
    """
    _model_input_cls: Type[TModelInputForNPU]
    _builder_cls: Type[ModelInputForNPUBuilder]

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config

        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.max_batchsize_to_capture = \
            self.vllm_config.compilation_config.max_capture_size

        self.has_inner_state = model_config.has_inner_state

        self.in_profile_run = False

        # Attention-free but stateful models like Mamba need a placeholder attn
        # backend, as the attention metadata is needed to manage internal state.
        # However we must bypass attention selection altogether for some models
        # used for speculative decoding to avoid a divide-by-zero in
        # model_config.get_head_size()
        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        needs_attn_backend = (num_attn_heads != 0
                              or self.model_config.is_attention_free)

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        ) if needs_attn_backend else None
        if self.attn_backend:
            self.attn_state = self.attn_backend.get_state_cls()(
                weakref.proxy(self))
        else:
            self.attn_state = CommonAttentionState(weakref.proxy(self))

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry \
            .create_input_mapper(model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization
        self.model: nn.Module  # Set after load_model

        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        # Used to cache python objects
        self.inter_data_cache: Dict[int, PyObjectCache] = {}

        # Using the PythonizationCache in Pipeline-Parallel clobbers the
        # SequenceGroupToSample object. In Pipeline-Parallel, we have
        # more than 1 Scheduler, resulting in a potential back-to-back
        # prepare_model_inputs() call. This clobbers the cached
        # SequenceGroupToSample objects, as we reset the cache during
        # every prepare_model_inputs() call.
        self.sampling_metadata_cache: SamplingMetadataCache = \
              SamplingMetadataCache() \
                if self.parallel_config.pipeline_parallel_size == 1 else None

    def get_model(self) -> nn.Module:
        return self.model

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:
            self.model = get_model(vllm_config=self.vllm_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from vllm.model_executor.model_loader.loader import TensorizerLoader
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> TModelInputForNPU:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.
        """
        builder = self._builder_cls(weakref.proxy(self), finished_requests_ids)
        builder.prepare(finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)

        builder.reset_cached_inter_data()

        return builder.build()  # type: ignore

    @contextmanager
    def set_in_profile_run(self):
        self.in_profile_run = True
        try:
            yield
        finally:
            self.in_profile_run = False

    @torch.inference_mode()
    def profile_run(self) -> None:
        with self.set_in_profile_run():
            # Enable top-k sampling to reflect the accurate memory usage.
            sampling_params = \
                SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
            max_num_batched_tokens = \
                self.scheduler_config.max_num_batched_tokens
            max_num_seqs = self.scheduler_config.max_num_seqs

            # Profile memory usage with max_num_sequences sequences and the
            # total number of tokens equal to max_num_batched_tokens.
            seqs: List[SequenceGroupMetadata] = []
            # Additional GPU memory may be needed for multi-modal encoding,
            # which needs to be accounted for when calculating the GPU blocks
            # for vLLM blocker manager.
            # To exercise the worst scenario for GPU memory consumption,
            # the number of seqs (batch_size) is chosen to maximize the number
            # of images processed.

            max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
                self.model_config)
            if max_mm_tokens > 0:
                max_num_seqs_orig = max_num_seqs
                max_num_seqs = min(max_num_seqs,
                                   max_num_batched_tokens // max_mm_tokens)
                if max_num_seqs < 1:
                    expr = (f"min({max_num_seqs_orig}, "
                            f"{max_num_batched_tokens} // {max_mm_tokens})")
                    logger.warning(
                        "Computed max_num_seqs (%s) to be less than 1. "
                        "Setting it to the minimum value of 1.", expr)
                    max_num_seqs = 1

            batch_size = 0
            for group_id in range(max_num_seqs):
                seq_len = (max_num_batched_tokens // max_num_seqs +
                           (group_id < max_num_batched_tokens % max_num_seqs))
                batch_size += seq_len

                dummy_data = self.input_registry \
                    .dummy_data_for_profiling(self.model_config,
                                            seq_len,
                                            self.mm_registry)

                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: dummy_data.seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                    lora_request=None,
                    multi_modal_data=dummy_data.multi_modal_data,
                    multi_modal_placeholders=dummy_data.
                    multi_modal_placeholders,
                )
                seqs.append(seq)

            # Run the model with the dummy inputs.
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            # use an empty tensor instead of `None`` to force Dynamo to pass
            # it by reference, rather by specializing on the value ``None``.
            # the `dtype` argument does not matter, and we use `float32` as
            # a placeholder (it has wide hardware support).
            # it is important to create tensors inside the loop, rather than
            # multiplying the list, to avoid Dynamo from treating them as
            # tensor aliasing.
            kv_caches = [
                torch.tensor([], dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]
            finished_requests_ids = [seq.request_id for seq in seqs]
            model_input = self.prepare_model_input(
                seqs, finished_requests_ids=finished_requests_ids)
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = \
                    self.model.make_empty_intermediate_tensors(
                    batch_size=batch_size,
                    dtype=self.model_config.dtype,
                    device=self.device)

            self.execute_model(model_input, kv_caches, intermediate_tensors)
            torch_npu.npu.synchronize()
            return

    def remove_all_loras(self):
        raise RuntimeError("LoRA is not supported on NPU now.")

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        raise RuntimeError("LoRA is not supported on NPU now.")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise RuntimeError("LoRA is not supported on NPU now.")

    def remove_lora(self, lora_id: int) -> bool:
        raise RuntimeError("LoRA is not supported on NPU now.")

    def pin_lora(self, lora_id: int) -> bool:
        raise RuntimeError("LoRA is not supported on NPU now.")

    def list_loras(self) -> Set[int]:
        raise RuntimeError("LoRA is not supported on NPU now.")

    def remove_all_prompt_adapters(self):
        raise RuntimeError("PromptAdapter is not supported on NPU now.")

    def set_active_prompt_adapters(
            self, prompt_adapter_requests: Set[PromptAdapterRequest],
            prompt_adapter_mapping: PromptAdapterMapping) -> None:
        raise RuntimeError("PromptAdapter is not supported on NPU now.")

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        raise RuntimeError("PromptAdapter is not supported on NPU now.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise RuntimeError("PromptAdapter is not supported on NPU now.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise RuntimeError("PromptAdapter is not supported on NPU now.")

    def list_prompt_adapters(self) -> Set[int]:
        raise RuntimeError("PromptAdapter is not supported on NPU now.")

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class NPUModelRunner(NPUModelRunnerBase[ModelInputForNPUWithSamplingMetadata]):
    """
    NPU model runner with sampling step.
    """
    _model_input_cls: Type[ModelInputForNPUWithSamplingMetadata] = (
        ModelInputForNPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForNPUBuilder] = ModelInputForNPUBuilder

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForNPUWithSamplingMetadata:
        model_input = \
            ModelInputForNPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForNPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.
        The API assumes seq_group_metadata_list is sorted by prefill -> decode.
        The result tensors and data structure also batches input in prefill
        -> decode order. For example,
        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.
        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        if get_pp_group().is_last_rank:
            # Sampling metadata is only required for the final pp group
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list,
                model_input.seq_lens,
                model_input.query_lens,
                self.device,
                self.pin_memory,
                generators,
                self.sampling_metadata_cache,
                # TODO (cmq): enable this after supported in vllm
                # pad_for_invariant_seq_len=True,
            )
            # Get hash value of request id list to perform sampling param cache in sampler.
            request_ids = model_input.request_ids_to_seq_ids.keys(  # type: ignore
            )  # type: ignore
            request_ids_hash = hash("".join(request_ids))
            sampling_metadata.request_ids_hash = request_ids_hash  # type: ignore
        else:
            sampling_metadata = None
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForNPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        self.attn_state.begin_forward(model_input)

        assert model_input.attn_metadata is not None
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        model_executable = self.model

        # Receive KV cache in distributed KV cache transfer setting
        # In disagg prefill setting, it will also recv hidden states and bypass
        # model forwarding
        # In KV cache database setting, it will change the model input so that
        # we can skip prefilling on tokens that successfully received KV caches
        # NOTE: The receive operation is blocking
        bypass_model_exec = False
        if self.need_recv_kv(model_input, kv_caches):
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    model_executable,
                    model_input,
                    kv_caches=kv_caches
                )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}

        previous_hidden_states = kwargs.get("previous_hidden_states")
        model_kwargs = {}
        if previous_hidden_states is not None:
            model_kwargs["previous_hidden_states"] = previous_hidden_states

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch_npu.npu.Event(enable_timing=True)
            model_forward_end = torch_npu.npu.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config, virtual_engine):
                if model_input.attn_metadata is not None:
                    model_input.attn_metadata.input_positions = model_input.input_positions
                hidden_or_intermediate_states = model_executable(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                                 device=self.device),
                    **seqlen_agnostic_kwargs,
                    **model_kwargs)

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (self.is_driver_worker
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states,
                                   IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time))
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time
                and output is not None):
            model_forward_end.synchronize()
            model_forward_time = model_forward_start.elapsed_time(
                model_forward_end)
            orig_model_forward_time = 0.0
            if intermediate_tensors is not None:
                orig_model_forward_time = intermediate_tensors.tensors.get(
                    "model_forward_time", torch.tensor(0.0)).item()
            # If there are multiple workers, we are still tracking the latency
            # from the start time of the driver worker to the end time of the
            # driver worker. The model forward time will then end up covering
            # the communication time as well.
            output.model_forward_time = (orig_model_forward_time +
                                         model_forward_time)

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
                output.prefill_hidden_states = hidden_or_intermediate_states
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]

    def need_recv_kv(self, model_input, kv_caches) -> bool:
        """Check if we need to receive kv-cache from the other worker.
        We need to receive KV when
            1. current vLLM instance is KV cache consumer/decode vLLM instance
            2. this batch is not a profiling run
            3. this batch is a prefill run
            
        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """

        if self.vllm_config.kv_transfer_config is None:
            return False

        prefill_meta = model_input.attn_metadata.prefill_metadata

        # check if the current run is profiling
        is_profile_run = (kv_caches[0].numel() == 0)
        # check if the current run is prefill
        is_prefill_run = prefill_meta is not None

        return self.vllm_config.kv_transfer_config.is_kv_consumer and (
            not is_profile_run) and is_prefill_run

    def need_send_kv(self, model_input, kv_caches) -> bool:
        """Check if we need to send kv-cache to the other worker.
        We need to send KV when
            1. current vLLM instance is KV cache producer/prefill vLLM instance
            2. this batch is not a profiling run
            3. this batch is a prefill run
            
        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """

        if self.vllm_config.kv_transfer_config is None:
            return False

        prefill_meta = model_input.attn_metadata.prefill_metadata

        # check if the current run is profiling
        is_profile_run = (kv_caches[0].numel() == 0)
        # check if the current run is prefill
        is_prefill_run = prefill_meta is not None

        return self.vllm_config.kv_transfer_config.is_kv_producer and (
            not is_profile_run) and is_prefill_run
