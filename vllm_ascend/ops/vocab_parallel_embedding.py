#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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


import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parameter import Parameter
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import divide
from vllm.distributed.parallel_state import get_pcp_group, get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    pad_vocab_size,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import (
    GroupCoordinator,
    get_embed_tp_group,
    get_lmhead_tp_group,
    get_replicated_group,
)
from vllm_ascend.utils import embedding_tp_enable, get_potential_max_tokens, lmhead_tp_enable


class AscendVocabParallelEmbedding(VocabParallelEmbedding):
    """
    Register VocabParallelEmbedding as a custom op for Ascend.
    AscendVocabParallelEmbedding support different communication parallel groups
    Added the feature of lmheadTP in pure dp scenario
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ):
        nn.Module.__init__(self)
        self.forward_type = None
        self.disable_tp = disable_tp

        # A disable_tp layer is pinned to the world_size=1 ReplicatedGroup:
        # every rank holds the full table, tp_size==1 makes shard_indices
        # cover the full vocab, and forward / logits skip all TP
        # communication. The DSpark Markov head reaches this through the
        # upstream interface — vllm's DSparkMarkovHead constructs the markov
        # lm_head with disable_tp=True (vllm#49731; its markov_w1 is a plain
        # nn.Embedding) — so Ascend needs no prefix heuristic of its own.
        # disable_tp must be matched before the lmhead prefix: the markov
        # prefix ("layers.N.markov_head.markov_w2") also contains "head" and
        # would otherwise be routed to the lmhead_tp group. The
        # ReplicatedGroup is a pure stand-in (no hcclCommInitRootInfoConfig)
        # exposing the attributes read below, so tp_size / tp_rank always
        # derive from the group — no None special case.
        if disable_tp:
            self.comm_group = get_replicated_group()
        elif lmhead_tp_enable() and "head" in prefix:
            self.comm_group = get_lmhead_tp_group()
        elif embedding_tp_enable() and "embed_tokens" in prefix:
            self.comm_group = get_embed_tp_group()
            self.forward_type = "embed_tp"
        else:
            self.comm_group = get_tp_group()
            vllm_config = get_current_vllm_config_or_none()
            if (
                self.comm_group.world_size == 1
                and vllm_config is not None
                and vllm_config.parallel_config.prefill_context_parallel_size > 1
                and vllm_config.model_config is not None
                and not embedding_tp_enable()
                and not lmhead_tp_enable()
            ):
                # Reuse vocab-row sharding and the checkpoint loader with PCP
                # ranks. Unlike TP, PCP ranks own different token sequences,
                # so lookup needs token all-gather and output reduce-scatter.
                # Both tables use the same layout, including tied weights.
                if get_ascend_config().enable_reduce_sample:
                    raise ValueError("PCP embedding/LM-head sharding does not support enable_reduce_sample.")
                self.comm_group = get_pcp_group()
                self.forward_type = "lmhead_pcp" if isinstance(self, ParallelLMHead) else "embed_pcp"
                # The existing embedding-TP capacity only covers decode.
                # PCP also runs long prefills through this collective path.
                self._pcp_embed_capacity = max(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    vllm_config.compilation_config.max_cudagraph_capture_size or 0,
                )

        self.tp_size = self.comm_group.world_size
        self.tp_rank = self.comm_group.rank_in_group

        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(self.org_vocab_size, self.padding_size)
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            self.tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim
        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(type(quant_method))
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        # Divide the weight matrix along the vocaburaly dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(self.num_embeddings_padded, self.tp_size)
        assert self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

        self.update_param_tp_status()

    def _mask_input_for_vocab_range(
        self,
        input_: torch.Tensor,
        org_vocab_start_index: int,
        org_vocab_end_index: int,
        num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # torch.compile will fuse all of the pointwise ops below
        # into a single kernel, making it very fast
        org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
        # Adapt: avoid create added_vocab_mask when added_vocab_start_index == added_vocab_end_index.
        if added_vocab_start_index == added_vocab_end_index:
            valid_offset = org_vocab_start_index * org_vocab_mask
            vocab_mask = org_vocab_mask
        else:
            added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
            added_offset = (
                added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
            )
            valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)
            vocab_mask = org_vocab_mask | added_vocab_mask
        # Adapt end.
        input_ = vocab_mask * (input_ - valid_offset)
        return input_, ~vocab_mask

    def forward(self, input_):
        if self.forward_type == "embed_pcp":
            return self._forward_embed_pcp(input_)
        if self.forward_type == "embed_tp":
            return self._forward_embed_tp(input_)
        return self._forward_origin(input_)

    def _forward_embed_pcp(self, input_):
        replicated = None
        if is_forward_context_available():
            replicated = getattr(get_forward_context().attn_metadata, "pcp_inputs_replicated", None)
        if replicated is True:
            return self._forward_replicated_pcp(input_)
        # MRV2 PCP pads every rank to the same current batch length. Unknown
        # contexts (e.g. profiling) retain the fixed-capacity fallback.
        return self._forward_partitioned_inputs(
            input_, self._pcp_embed_capacity, active_tokens=input_.shape[0] if replicated is False else None
        )

    def _forward_replicated_pcp(self, input_):
        num_tokens = input_.shape[0]
        if num_tokens > self._pcp_embed_capacity:
            raise ValueError("PCP embedding input exceeds static capacity.")
        if not hasattr(self, "_pcp_decode_out"):
            self._pcp_decode_out = torch.empty(
                (self._pcp_embed_capacity, self.embedding_dim), dtype=self.params_dtype, device=input_.device
            )
        masked_input, input_mask = self._mask_input_for_vocab_range(
            input_,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        output = self._pcp_decode_out[:num_tokens]
        output.copy_(self.quant_method.embedding(self, masked_input.long()))
        output.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Same IDs and row order on every rank: no token gather is necessary.
        if num_tokens:
            dist.all_reduce(output, group=self.comm_group.device_group)
        return output

    def _forward_embed_tp(self, input_):
        return self._forward_partitioned_inputs(input_, get_potential_max_tokens())

    def _forward_partitioned_inputs(self, input_, capacity, active_tokens=None):
        assert self.comm_group is not None
        num_tokens = input_.shape[0]

        # All ranks use the same static capacity, including empty local inputs.
        if num_tokens > capacity:
            raise ValueError(
                f"{self.forward_type} static capacity {capacity} < num_tokens "
                f"{num_tokens}; increase max_cudagraph_capture_size or "
                f"max_num_batched_tokens."
            )

        # Lazy init on first call (profiling run, which precedes ACL graph
        # capture). Static buffers keep a stable device address across all
        # later capture/replay cycles — graph replay requires the same
        # address that was recorded at capture (comm_group.all_gather and
        # reduce_scatter internally torch.empty() per call, which would
        # desync the HCCL operator recorded at capture).
        # Mirrors the OTP v13 fix in dsa_v1.py:_forward_o_proj.
        if not hasattr(self, "_embed_ag_in_buf"):
            device = input_.device
            # all_gather buffers carry token IDs (int64).
            self._embed_ag_in_buf = torch.zeros((capacity,), dtype=input_.dtype, device=device)
            self._embed_ag_out_buf = torch.empty((self.tp_size * capacity,), dtype=input_.dtype, device=device)
            # reduce_scatter buffers carry bf16 embeddings.
            self._embed_rs_in_buf = torch.empty(
                (self.tp_size * capacity, self.embedding_dim), dtype=self.params_dtype, device=device
            )
            self._embed_rs_out_buf = torch.empty((capacity, self.embedding_dim), dtype=self.params_dtype, device=device)

        comm_tokens = capacity if active_tokens is None else active_tokens
        if comm_tokens == 0:
            return self._embed_rs_out_buf[:0]
        gather_input = self._embed_ag_in_buf[:comm_tokens]
        complete_input = self._embed_ag_out_buf[: self.tp_size * comm_tokens]
        scatter_input = self._embed_rs_in_buf[: self.tp_size * comm_tokens]
        scatter_output = self._embed_rs_out_buf[:comm_tokens]
        # Views keep stable base addresses while sending only the current
        # padded batch, not the scheduler's maximum capacity, on PCP steps.
        gather_input.zero_()
        gather_input[:num_tokens].copy_(input_)
        dist.all_gather_into_tensor(complete_input, gather_input, group=self.comm_group.device_group)

        # Mask tokens outside this rank's vocab shard. Capacity padding uses
        # token 0; its embeddings are discarded when stripping output padding.
        masked_input, input_mask = self._mask_input_for_vocab_range(
            complete_input,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        # Embedding lookup is a local op (F.embedding); its fresh allocation
        # does not affect ACL graph replay. Copy into the static rs_in
        # buffer so reduce_scatter reads from a stable address.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        scatter_input.copy_(output_parallel)
        scatter_input.masked_fill_(input_mask.unsqueeze(-1), 0)
        dist.reduce_scatter_tensor(scatter_output, scatter_input, group=self.comm_group.device_group)

        # Strip padding rows; preserve the original return shape.
        return self._embed_rs_out_buf[:num_tokens].view(num_tokens, self.embedding_dim)

    def _forward_origin(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = self._mask_input_for_vocab_range(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        else:
            return output_parallel

        # Reduce across all the model parallel GPUs.
        tp_group = get_tp_group()
        if tp_group.world_size == 1:
            return output_parallel
        # vLLM 0.26 model forwards expect the first decoder layer to receive
        # the complete token sequence. Sequence parallelism starts only after
        # that layer's attention output, so reducing-scattering the embedding
        # here would feed each TP rank only a token shard. The dedicated
        # embedding-TP path above owns its complete gather/scatter protocol;
        # the regular TP embedding path must keep upstream all-reduce semantics.
        return torch.ops.vllm.all_reduce(output_parallel, tp_group.unique_name)


class AscendParallelLMHead(ParallelLMHead):
    """
    Register ParallelLMHead as a custom op for Ascend."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ):
        AscendVocabParallelEmbedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
            disable_tp=disable_tp,
        )
        self.quant_config = quant_config
        if bias:
            self.bias = Parameter(torch.empty(self.num_embeddings_per_partition, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)


def lmhead_all_to_all(
    logits: torch.Tensor,
    comm_group: GroupCoordinator,
) -> torch.Tensor:
    """All-to-all for lm-head TP: redistribute `[N, V/P]` (all tokens, partial
    vocab) into `[N/P, V]` (partial tokens, full vocab).

    Uses ``all_to_all_single`` on the P axis made explicit by a ``view``: the
    input is reshaped to ``[P, N/P, V/P]`` so that dim 0 carries the per-rank
    token shard. After the single collective, ``permute(1, 0, 2)`` interleaves
    the vocab shards of each token, and a final ``view`` flattens back to
    ``[N/P, V]``. This is mathematically equivalent to the list-based
    ``tensor_split(dim=0) + all_to_all(list) + cat(dim=-1)`` but keeps a single
    contiguous buffer for better HCCL fusion, and avoids the Python list of
    per-rank tensors.

    The vocab shard ``V/P`` is identical on every rank because
    ``pad_vocab_size`` + ``divide`` align it at build time. The token count
    ``N`` must be divisible by ``world_size`` so ``all_to_all_single`` can
    redistribute dim 0 equally; this is checked explicitly to give a clear
    error instead of the cryptic ``view`` shape failure.
    """
    world_size = comm_group.world_size
    if world_size == 1:
        return logits
    # all_to_all_single in SPMD mode requires equal split along dim 0:
    # the view [P, N/P, V/P] below needs N divisible by P.
    if logits.shape[0] % world_size != 0:
        raise ValueError(
            f"logits.shape[0] ({logits.shape[0]}) must be divisible by world_size ({world_size}) for lmhead_all_to_all."
        )
    vocab_per_partition = logits.shape[-1]
    # [N, V/P] -> [P, N/P, V/P]. The `.contiguous()` is a no-op on the live
    # lm-head path (fresh matmul output), but load-bearing for the spec-decode
    # reduce-sample callers, whose input is a vocab-truncated last-dim slice
    # (non-contiguous whenever the vocab shard is padded): view() would fail.
    input_ = logits.contiguous().view(world_size, -1, vocab_per_partition)
    output = torch.empty_like(input_)
    dist.all_to_all_single(output, input_, group=comm_group.device_group)
    # [P, N/P, V/P] -> [N/P, P, V/P] -> [N/P, V]
    return output.permute(1, 0, 2).contiguous().view(-1, world_size * vocab_per_partition)


class AscendLogitsProcessor(LogitsProcessor):
    """
    Register LogitsProcessor as a custom op for Ascend.
    Added the feature of lmheadTP in pure dp scenario
    """

    def _apply_head(
        self,
        lm_head: AscendParallelLMHead,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return super()._apply_head(lm_head, hidden_states, embedding_bias)

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        if getattr(lm_head, "forward_type", None) in ("lmhead_pcp", "embed_pcp"):
            return self._get_logits_pcp(hidden_states, lm_head, embedding_bias, skip_gather)
        # vLLM #50465 added skip_gather; when set, upstream returns the
        # untruncated apply_head result for spec-decode/top-k callers.
        if skip_gather:
            return self._apply_head(lm_head, hidden_states, embedding_bias)
        # A replicated head (tp_size==1, e.g. the DSpark markov lm_head)
        # must take the normal path: the lmhead_tp path gathers hidden
        # states / scatters logits across the finegrained group, which a
        # replicated head must not participate in.
        if lmhead_tp_enable() and lm_head.tp_size > 1:
            return self._get_logits_lmheadtp(hidden_states, lm_head, embedding_bias)
        else:
            return self._get_logits_normal(hidden_states, lm_head, embedding_bias)

    def _get_logits_pcp(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: torch.Tensor | None,
        skip_gather: bool,
    ) -> torch.Tensor:
        # MRV2 restores PCP hidden states to the global batch before sampling
        # and prompt logprobs. Every rank projects the same token rows onto
        # its vocab shard; concatenate vocab columns, never sum logits.
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        logits = lm_head.comm_group.all_gather(logits, dim=-1)
        # skip_gather callers expect a TP-local vocabulary. TP=1 here, so
        # reconstruct the full padded vocabulary even on that path.
        if skip_gather:
            return logits
        return logits[..., : self.org_vocab_size]

    def _get_logits_lmheadtp(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Gather hidden states from all devices in tensor parallel group
        gathered_hidden_states = get_lmhead_tp_group().all_gather(hidden_states, dim=0)
        logits = self._apply_head(lm_head, gathered_hidden_states, embedding_bias)
        # Gather logits for tensor parallel
        if not get_ascend_config().enable_reduce_sample:
            logits = lmhead_all_to_all(logits, get_lmhead_tp_group())

        # Remove paddings in vocab (if any)
        if logits is not None:
            if not get_ascend_config().enable_reduce_sample:
                logits = logits[..., : self.org_vocab_size]
            else:
                logits = logits[..., : lm_head.num_org_embeddings_per_partition]
        return logits

    def _get_logits_normal(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        # Gather logits for tensor parallel. _gather_logits uses the global TP
        # group, so skip it for a replicated head (e.g. the DSpark Markov w2):
        # each rank already holds the full vocab logits locally and no
        # all-gather is needed.
        if not get_ascend_config().enable_reduce_sample and lm_head.tp_size > 1:
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any)
        if logits is not None:
            if not get_ascend_config().enable_reduce_sample:
                logits = logits[..., : self.org_vocab_size]
            else:
                logits = logits[..., : lm_head.num_org_embeddings_per_partition]

        return logits
