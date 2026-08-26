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


from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parameter import Parameter
from vllm.config import get_current_vllm_config
from vllm.distributed import divide
from vllm.distributed.parallel_state import get_pcp_group, get_tp_group
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

from vllm_ascend.distributed.parallel_state import (
    GroupCoordinator,
    get_embed_tp_group,
    get_lmhead_tp_group,
)
from vllm_ascend.utils import (
    embedding_tp_enable,
    enable_pcp_embedding_lmhead_weight_sharding,
    get_potential_max_tokens,
    lmhead_tp_enable,
)


class VocabParallelMode(Enum):
    """Vocabulary-weight layout and its corresponding execution strategy.

    REPLICATED keeps the complete weight on every rank and performs no
    vocabulary-parallel communication. It is used when ``disable_tp=True``.

    STANDARD shards the vocabulary over the standard TP group. Embedding ranks
    process the same token rows and all-reduce their local vocabulary
    contributions over TP.

    FINE_GRAINED shards the selected component over its independently
    configured DP-axis group. Embedding exchanges token rows over Embed-TP;
    LM head uses its LMHead-TP gather/all-to-all path.

    PCP_X_TP shards the vocabulary over the logical TP x PCP grid. Embedding
    exchanges token rows and reduce-scatters over PCP, then all-reduces the
    remaining vocabulary contributions over TP. LM head reconstructs logits by
    gathering vocabulary shards over PCP first and TP second.
    """

    REPLICATED = auto()
    STANDARD = auto()
    FINE_GRAINED = auto()
    PCP_X_TP = auto()


@dataclass(frozen=True)
class VocabParallelPlan:
    mode: VocabParallelMode
    shard_rank: int
    shard_world_size: int
    token_exchange_group: GroupCoordinator | None = None
    output_reduce_group: GroupCoordinator | None = None


def _resolve_vocab_parallel_plan(
    *,
    prefix: str,
    disable_tp: bool,
) -> VocabParallelPlan:
    # vLLM's DSpark Markov head constructs markov_w2 as a ParallelLMHead with
    # disable_tp=True. Resolve it first so its "markov_head" prefix cannot route
    # the replicated weight into LM-head or PCP sharding.
    if disable_tp:
        return VocabParallelPlan(VocabParallelMode.REPLICATED, 0, 1)

    is_token_embedding = "embed_tokens" in prefix
    is_lm_head = "head" in prefix
    use_pcp_sharding = (
        (is_token_embedding or is_lm_head)
        and enable_pcp_embedding_lmhead_weight_sharding()
        and get_current_vllm_config().parallel_config.prefill_context_parallel_size > 1
    )

    if use_pcp_sharding:
        pcp_group = get_pcp_group()
        if is_token_embedding and embedding_tp_enable():
            raise ValueError(
                "PCP embedding weight sharding cannot be combined with "
                "finegrained_tp_config.embedding_tensor_parallel_size."
            )
        if is_lm_head and lmhead_tp_enable():
            raise ValueError(
                "PCP LM head weight sharding cannot be combined with finegrained_tp_config.lmhead_tensor_parallel_size."
            )

        tp_group = get_tp_group()
        # Linearize the TP x PCP grid with PCP as the inner dimension. This
        # matches the reconstruction order: gather PCP vocabulary shards first,
        # then gather/reduce the remaining TP shards.
        return VocabParallelPlan(
            VocabParallelMode.PCP_X_TP,
            tp_group.rank_in_group * pcp_group.world_size + pcp_group.rank_in_group,
            tp_group.world_size * pcp_group.world_size,
            token_exchange_group=None if is_lm_head else pcp_group,
            output_reduce_group=None if is_lm_head else tp_group,
        )

    if is_lm_head and lmhead_tp_enable():
        group = get_lmhead_tp_group()
        return VocabParallelPlan(
            VocabParallelMode.FINE_GRAINED,
            group.rank_in_group,
            group.world_size,
        )

    if is_token_embedding and embedding_tp_enable():
        group = get_embed_tp_group()
        return VocabParallelPlan(
            VocabParallelMode.FINE_GRAINED,
            group.rank_in_group,
            group.world_size,
            token_exchange_group=group,
        )

    tp_group = get_tp_group()
    return VocabParallelPlan(
        VocabParallelMode.STANDARD,
        tp_group.rank_in_group,
        tp_group.world_size,
        output_reduce_group=tp_group,
    )


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
        self.disable_tp = disable_tp
        self.embedding_tp_capacity: int | None = None

        plan = _resolve_vocab_parallel_plan(
            prefix=prefix,
            disable_tp=disable_tp,
        )
        self.parallel_mode = plan.mode
        self.token_exchange_group = plan.token_exchange_group
        self.output_reduce_group = plan.output_reduce_group
        # tp_rank/tp_size describe the logical weight layout. For PCP_X_TP that
        # layout spans two real communication groups rather than one TP group.
        self.tp_rank = plan.shard_rank
        self.tp_size = plan.shard_world_size

        if self.token_exchange_group is not None:
            self.embedding_tp_capacity = max(
                get_potential_max_tokens(),
                get_current_vllm_config().scheduler_config.max_num_batched_tokens,
            )

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
        if self.token_exchange_group is not None:
            return self._forward_with_token_exchange(input_)
        return self._forward_standard(input_)

    def _forward_with_token_exchange(self, input_):
        """Exchange token rows before lookup and restore local token rows.

        Fine-grained Embedding TP exchanges rows over its DP-axis group; PCP
        weight sharding exchanges rows over the PCP group. Both gather token
        IDs, perform lookup against the local vocabulary shard, then
        reduce-scatter embeddings back to the original token owners.

        PCP_X_TP deliberately performs the PCP reduce-scatter before the final
        TP all-reduce, so the TP collective carries only this PCP rank's token
        rows instead of all PCP token rows.

        Communication-order example with TP=2, PCP=2,
        capacity=num_tokens=2, and hidden_size=4096::

            After PCP token all-gather and local embedding lookup:
                output_parallel shape = [4, 4096] (16,384 elements)

            PCP-first (current):
                PCP reduce-scatter: [4, 4096] -> [2, 4096]
                TP all-reduce:      [2, 4096] (8,192 elements)

            TP-first (alternative):
                TP all-reduce:      [4, 4096] (16,384 elements)
                PCP reduce-scatter: [4, 4096] -> [2, 4096]
        """
        num_tokens = input_.shape[0]
        exchange_group = self.token_exchange_group
        assert exchange_group is not None
        comm_size = exchange_group.world_size

        # potential_max_tokens covers the uniform decode path. Fine-grained
        # Embedding TP and DSA PCP also enter this path during prefill, including
        # the max_num_batched_tokens profiling run, so their static buffers must
        # cover the scheduler's full token capacity as well.
        capacity = self.embedding_tp_capacity
        assert capacity is not None
        if num_tokens > capacity:
            raise ValueError(
                f"embedding_tp static capacity {capacity} < num_tokens "
                f"{num_tokens}; increase max_cudagraph_capture_size or "
                f"max_num_batched_tokens."
            )

        # Lazy init on first call (profiling run, which precedes ACL graph
        # capture). Static buffers keep a stable device address across all
        # later capture/replay cycles — graph replay requires the same
        # address that was recorded at capture (the group helpers allocate new
        # collective outputs per call, which would desync the HCCL operator
        # recorded at capture).
        # Mirrors the OTP v13 fix in dsa_v1.py:_forward_o_proj.
        if not hasattr(self, "_embed_ag_in_buf"):
            device = input_.device
            # all_gather buffers carry token IDs (int64).
            self._embed_ag_in_buf = torch.zeros((capacity,), dtype=input_.dtype, device=device)
            self._embed_ag_out_buf = torch.empty((comm_size * capacity,), dtype=input_.dtype, device=device)
            # reduce_scatter buffers carry bf16 embeddings.
            self._embed_rs_in_buf = torch.empty(
                (comm_size * capacity, self.embedding_dim), dtype=self.params_dtype, device=device
            )
            self._embed_rs_out_buf = torch.empty((capacity, self.embedding_dim), dtype=self.params_dtype, device=device)

        # Pad input into the address-stable all_gather input buffer.
        self._embed_ag_in_buf.zero_()
        self._embed_ag_in_buf[:num_tokens].copy_(input_)
        dist.all_gather_into_tensor(
            self._embed_ag_out_buf,
            self._embed_ag_in_buf,
            group=exchange_group.device_group,
        )
        complete_input = self._embed_ag_out_buf

        # Masking unchanged; padding rows map to OOB and get masked to 0
        # via masked_fill_ below (token_id=0 stays in-range after shift).
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
        self._embed_rs_in_buf.copy_(output_parallel)
        self._embed_rs_in_buf.masked_fill_(input_mask.unsqueeze(-1), 0)
        dist.reduce_scatter_tensor(
            self._embed_rs_out_buf,
            self._embed_rs_in_buf,
            group=exchange_group.device_group,
        )

        # Strip padding rows; preserve the original return shape.
        output = self._embed_rs_out_buf[:num_tokens].view(num_tokens, -1)
        reduce_group = self.output_reduce_group
        if reduce_group is not None and reduce_group.world_size > 1:
            # PCP reduce-scatter reconstructs the TP-local contribution. This
            # reduction completes the vocabulary result across TP shards.
            output = torch.ops.vllm.all_reduce(output, reduce_group.unique_name)
        return output

    def _forward_standard(self, input_):
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

        reduce_group = self.output_reduce_group
        if reduce_group is None or reduce_group.world_size == 1:
            return output_parallel
        # Standard vocabulary parallelism keeps identical token rows on every
        # rank. Sum the vocabulary-shard contributions without scattering the
        # token dimension; the first decoder layer expects the complete sequence.
        return torch.ops.vllm.all_reduce(output_parallel, reduce_group.unique_name)


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
    # lm-head path (fresh matmul output).
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
        # vLLM #50465 added skip_gather; when set, upstream returns the
        # untruncated apply_head result for spec-decode/top-k callers.
        if skip_gather:
            return self._apply_head(lm_head, hidden_states, embedding_bias)
        if lm_head.parallel_mode is VocabParallelMode.PCP_X_TP:
            return self._get_logits_pcp_weight_sharding(hidden_states, lm_head, embedding_bias)
        # A replicated head (tp_size==1, e.g. the DSpark markov lm_head)
        # must take the normal path: the lmhead_tp path gathers hidden
        # states / scatters logits across the finegrained group, which a
        # replicated head must not participate in.
        if lm_head.parallel_mode is VocabParallelMode.FINE_GRAINED:
            return self._get_logits_lmheadtp(hidden_states, lm_head, embedding_bias)
        else:
            return self._get_logits_normal(hidden_states, lm_head, embedding_bias)

    def _get_logits_pcp_weight_sharding(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return full-vocab logits for globally restored token rows."""
        pcp_group = get_pcp_group()
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        # PCPManager restores the global token order before sampling, so only
        # the vocabulary shards need to be reconstructed here.
        if pcp_group.world_size > 1:
            logits = pcp_group.all_gather(logits, dim=-1)
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            logits = tp_group.all_gather(logits, dim=-1)
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
        logits = lmhead_all_to_all(logits, get_lmhead_tp_group())

        # Remove paddings in vocab (if any)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
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
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]

        return logits
