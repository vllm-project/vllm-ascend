# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA attention for the GLM-5.3-Flash kpool indexer.

GLM-5.3-Flash is NoPE MLA plus KDA, and its sparse layers attend to a subset of
the past chosen by a *pooled* indexer: groups of ``index_kpool`` consecutive
index keys compress to one cached vector, so both the index cache and the
scoring cost shrink by that factor.

This builds on ``AscendMLAImpl`` rather than ``AscendSFAImpl``. DeepSeek's
sparse impl assumes rope-ful MLA over a two-tensor main cache, while every
layer here has ``qk_rope_head_dim == 0``, which is also the configuration the
dense GLM path already runs correctly.

Per sparse layer the chain is: project the indexer key, its pooling gate and
its query; compress into the pooled key cache and the tail ring; score the
pools and expand the winners into token ids; attend over just those tokens.

Where the metadata comes from, since three cache groups are involved:

============================================  ==============================
``attn_metadata``                             block table, sequence lengths,
                                              token positions, the shared
                                              zero rope buffers
``attn_metadata[k_cache.prefix]``             pool-granular slot mapping
``attn_metadata[tail_cache.prefix]``          tail ring slot mapping
============================================  ==============================

One deliberate limit of this first implementation:

* The indexer query re-runs the q-lora projection instead of reusing the one
  the MLA preprocess already computed, which costs one extra GEMM per sparse
  layer. The preprocess does not hand ``q_c`` back and it is shared with the
  models that have no indexer, so threading it out is left for later.
"""

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context

from vllm_ascend.attention.mla_v1 import (
    AscendMLABackend,
    AscendMLAImpl,
    AscendMLAMetadataBuilder,
    _mla_nope_zero_rope,
)
from vllm_ascend.models.glm5next.ops.kpool_cache import write_decode, write_prefill
from vllm_ascend.models.glm5next.ops.kpool_indexer import rows_as_batch_entries, select_token_ids

# npu_sparse_flash_attention rejects a zero rope width, so a NoPE model has to
# pay for a rope half it does not have. Probing an Ascend 950 showed the
# operator accepts all-zero rope operands, and accepts one zero page aliased
# across the cache, which keeps that cost O(1) rather than cache-sized.
SPARSE_ROPE_DIM = 64

# The operator takes token ids: a probe rejected sparse_block_size=4, so the
# selected pools are expanded back into the tokens they were compressed from.
SPARSE_BLOCK_SIZE = 1

# Two independent axes that npu_sparse_flash_attention happens to name
# similarly. ``sparse_mode`` picks the causal crop -- 3 is RightDownCausal,
# which every other sparse-attention call in vllm-ascend uses and which the
# probe exercised. ``attention_mode`` selects the MLA kernel; a probe confirmed
# 0 is rejected outright.
SPARSE_CAUSAL_MODE = 3
SPARSE_ATTENTION_MODE = 2


class AscendKpoolMLABackend(AscendMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_KPOOL_MLA"

    @staticmethod
    def get_impl_cls() -> type["AscendKpoolMLAImpl"]:
        return AscendKpoolMLAImpl

    @staticmethod
    def get_builder_cls() -> type[AscendMLAMetadataBuilder]:
        # Everything the sparse chain reads -- block table, sequence lengths,
        # query offsets, token positions, the shared zero rope buffers -- the
        # dense builder already produces, capture support included.
        return AscendMLAMetadataBuilder


class AscendKpoolMLAImpl(AscendMLAImpl):
    """NoPE MLA whose layers attend to the tokens the kpool indexer selects."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        indexer = kwargs.get("indexer")
        if indexer is None or getattr(indexer, "index_kpool", None) is None:
            raise ValueError(
                "The kpool MLA impl needs an indexer carrying index_kpool, got "
                f"{type(indexer).__name__}. layer_name={self.layer_name}."
            )
        if self.qk_rope_head_dim != 0:
            raise ValueError(
                "The kpool MLA impl is for NoPE MLA; a rope-ful checkpoint belongs on "
                f"the SFA path. Got qk_rope_head_dim={self.qk_rope_head_dim}, "
                f"layer_name={self.layer_name}."
            )

        self.indexer = indexer
        self.index_kpool: int = indexer.index_kpool
        self.index_n_heads: int = indexer.n_head
        self.index_head_dim: int = indexer.head_dim
        self.topk_tokens: int = indexer.topk_tokens
        # The scores are a weights-weighted sum over heads, so folding the
        # softmax scale into the weights keeps it out of the operator's call.
        self.index_weight_scale: float = indexer.softmax_scale * indexer.n_head**-0.5

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        """No-op: nothing in the sparse call has to be rebound before a replay.

        The dense path re-issues its attention through a task-group handle
        because ``npu_fused_infer_attention_score`` takes the key lengths as a
        host-side Python list, and a captured graph has no way to refresh one.
        Every operand of ``npu_sparse_flash_attention`` is a tensor, so a replay
        picks up the new step by reading the same buffers, the way the SFA impl
        already relies on.
        """

    # ---- inputs -----------------------------------------------------------

    def _indexer_caches(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The pooled key cache and the tail ring.

        Both belong to their own cache layers rather than to this attention
        layer, so they arrive through the indexer instead of in the tuple the
        runner passes to ``forward``.
        """
        return (
            _single_tensor(self.indexer.k_cache.kv_cache, "key"),
            _single_tensor(self.indexer.tail_cache.kv_cache, "tail"),
        )

    def _indexer_slot_mappings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool-granular and tail-ring slot mappings, by cache layer name."""
        per_layer = get_forward_context().attn_metadata
        if not isinstance(per_layer, dict):
            raise RuntimeError(
                "The kpool indexer needs per-layer attention metadata to reach its "
                f"cache groups. layer_name={self.layer_name}."
            )
        return (
            _group_slot_mapping(per_layer, self.indexer.k_cache.prefix, "key"),
            _group_slot_mapping(per_layer, self.indexer.tail_cache.prefix, "tail"),
        )

    def _project_indexer_key(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The per-token indexer key, its pooling gate, and the head weights.

        No rope: this is a NoPE checkpoint, so the key is entirely nope and the
        split-rotate-concatenate of the rope-ful path would only ever build
        zero-width tensors.
        """
        fused, _ = self.indexer.wk_weights_proj(hidden_states)
        key = self.indexer.k_norm(fused[:, : self.index_head_dim])
        weights = fused[:, self.index_head_dim :]
        # F.linear(x, gate) == x @ gate.T, for gate [head_dim, hidden_size].
        gate = F.linear(hidden_states, self.indexer.index_kpool_compress_gate)
        return key, gate, weights

    def _project_indexer_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.fused_qkv_a_proj is not None, "The kpool indexer needs a q-lora."
        assert self.q_a_layernorm is not None, "The kpool indexer needs q_a_layernorm."
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c = qkv_lora[..., : self.q_lora_rank]
        query, _ = self.indexer.wq_b(self.q_a_layernorm(q_c))
        return query.view(-1, self.index_n_heads, self.index_head_dim)

    # ---- the indexer chain ------------------------------------------------

    def select_tokens(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        max_query_len: int,
        is_decode: bool,
        token_offset: int = 0,
        score: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compress this step's keys into the caches, then pick token ids.

        Returns the selected ids and how many of them each row filled, which is
        what the sparse attention needs as its key length.

        ``token_offset`` is where this phase's tokens start in the batch. The
        cache groups publish one slot mapping spanning the whole batch, decode
        rows first, so the prefill phase reads from partway in.

        ``score=False`` writes the caches and stops. Prefill needs the writes
        -- decode cannot see history that was never compressed -- but not the
        selection, because prefill attends densely.
        """
        index_cache, tail_cache = self._indexer_caches()
        pool_slots, tail_slots = self._indexer_slot_mappings()
        key, gate, weights = self._project_indexer_key(hidden_states)

        num_tokens = key.shape[0]
        end = token_offset + num_tokens
        pool_slots = pool_slots[token_offset:end]
        tail_slots = tail_slots[token_offset:end]
        positions = positions[:num_tokens]

        if is_decode:
            # Decode groups the batch by request so a pool completing partway
            # through a draft-verify step still sees the rows before it.
            num_requests = query_lens.shape[0]
            rows = num_tokens // max(num_requests, 1)
            write_decode(
                index_cache,
                tail_cache,
                key.view(num_requests, rows, self.index_head_dim),
                gate.view(num_requests, rows, self.index_head_dim),
                self.indexer.index_kpool_compress_ape,
                pool_slots=pool_slots.view(num_requests, rows),
                tail_slots=tail_slots.view(num_requests, rows),
                positions=positions.view(num_requests, rows),
                pool_size=self.index_kpool,
                pools_per_block=self.indexer.k_cache.pools_per_block,
            )
        else:
            write_prefill(
                index_cache,
                tail_cache,
                key,
                gate,
                self.indexer.index_kpool_compress_ape,
                pool_slots=pool_slots,
                tail_slots=tail_slots,
                pool_size=self.index_kpool,
                pools_per_block=self.indexer.k_cache.pools_per_block,
            )

        if not score:
            return None

        return select_token_ids(
            self._project_indexer_query(hidden_states),
            (weights * self.index_weight_scale).to(key.dtype),
            index_cache,
            block_table=block_table,
            query_lens=query_lens,
            seq_lens=seq_lens,
            pool_size=self.index_kpool,
            pools_per_block=self.indexer.k_cache.pools_per_block,
            topk_tokens=self.topk_tokens,
            max_query_len=max_query_len,
        )

    # ---- attention --------------------------------------------------------

    def _sparse_attention(
        self,
        q_nope: torch.Tensor,
        key_cache: torch.Tensor,
        token_ids: torch.Tensor,
        block_table: torch.Tensor,
        cumulative_query_lens: torch.Tensor,
        selected_lens: torch.Tensor,
        zero_rope_cache: dict,
    ) -> torch.Tensor:
        """Attend over the selected tokens only.

        The batch is one entry per query row, so ``block_table`` and both length
        arguments all carry one entry per row: ``cumulative_query_lens`` because
        the operator sizes its batch from it, ``selected_lens`` because it states
        how many entries of ``token_ids`` that row filled.

        Both lengths arrive on the query's device, which is what lets this call
        be captured: every operand being a tensor means a replay reads the new
        step out of the same buffers and nothing has to be rebound.
        """
        import torch_npu

        # A NoPE model has no rope operands, and the operator will not take a
        # zero width, so both halves are the shared all-zero buffer.
        q_rope = _mla_nope_zero_rope(q_nope, SPARSE_ROPE_DIM, zero_rope_cache)
        k_rope = _mla_nope_zero_rope(key_cache, SPARSE_ROPE_DIM, zero_rope_cache)

        # The operator names the key-side length and layout ``*_kv``, not
        # ``*_key``, and always answers with (attention_out, softmax_max,
        # softmax_sum) even when the softmax terms are not asked for.
        attn_output, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope,
            key=key_cache,
            value=key_cache,
            # The index list is laid out like the query, so TND wants a KV-head
            # axis between the rows and the ids. The selection chain reports one
            # row of ids per query row; the single MLA KV head is added here.
            sparse_indices=token_ids.unsqueeze(1),
            scale_value=self.scale,
            sparse_block_size=SPARSE_BLOCK_SIZE,
            block_table=block_table,
            actual_seq_lengths_query=cumulative_query_lens,
            actual_seq_lengths_kv=selected_lens,
            query_rope=q_rope,
            key_rope=k_rope,
            sparse_mode=SPARSE_CAUSAL_MODE,
            attention_mode=SPARSE_ATTENTION_MODE,
            layout_query="TND",
            layout_kv="PA_BSND",
        )
        return attn_output

    # ---- driving the phases ----------------------------------------------

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the indexer for each phase, then the dense forward's structure.

        The selected ids are stashed per phase rather than threaded through,
        because ``_forward_decode`` and ``_forward_prefill`` are called by the
        parent and take no place to pass them. One forward call runs one layer
        to completion before the next starts, which is the same assumption the
        module-level forward context in this package already makes.
        """
        if attn_metadata is None:
            # Profiling run.
            assert output is not None, "Output tensor must be provided."
            return output.fill_(0)

        self._decode_selection: tuple[torch.Tensor, torch.Tensor] | None = None
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.decode is not None:
            decode = attn_metadata.decode
            num_requests = decode.seq_lens.shape[0]
            rows = num_decode_tokens // max(num_requests, 1)
            # Both lengths have to reach the selection already on the device.
            # Copying them in would work while eager and break under ACL-graph
            # capture, which records the host address the copy read and replays
            # against whatever occupies it by then. decode.seq_lens is a host
            # tensor, but the positions are not, and a decode row's position is
            # one short of the tokens its request knows.
            decode_positions = decode.input_positions[:num_decode_tokens].view(num_requests, rows)
            self._decode_selection = self.select_tokens(
                hidden_states[:num_decode_tokens],
                decode.input_positions,
                decode.block_table,
                # A decode batch gives every request the same number of rows:
                # one when plain, num_spec + 1 when verifying drafts.
                query_lens=torch.full((num_requests,), rows, dtype=torch.int32, device=decode_positions.device),
                seq_lens=(decode_positions[:, -1] + 1).to(torch.int32),
                max_query_len=rows,
                is_decode=True,
            )

        if attn_metadata.prefill is not None:
            prefill = attn_metadata.prefill
            # Writes only. Prefill attends densely, which is exact and is a
            # superset of what the selection would have picked, but the pools it
            # produces are what decode later scores against.
            self.select_tokens(
                hidden_states[num_decode_tokens : attn_metadata.num_actual_tokens],
                prefill.input_positions,
                prefill.block_table,
                query_lens=prefill.query_lens,
                # prefill.seq_lens is a host list; the batch-wide tensor already
                # holds the same values with the prefill requests last.
                seq_lens=attn_metadata.seq_lens[attn_metadata.num_decodes :],
                max_query_len=prefill.max_query_len,
                is_decode=False,
                token_offset=num_decode_tokens,
                score=False,
            )

        return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)

    def _forward_decode(
        self,
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        block_size,
        attn_metadata,
        dequant_scale_q_nope=None,
    ):
        decode = attn_metadata.decode
        assert decode is not None
        assert self._decode_selection is not None, "The decode indexer did not run."
        assert decode.nope_zero_rope_cache is not None, (
            "Sparse NoPE decode needs the zero rope buffers owned by the metadata "
            "builder, but none were created: hf_text_config.qk_rope_head_dim "
            "disagrees with this layer's qk_rope_head_dim."
        )
        token_ids, selected_lens = self._decode_selection
        num_tokens = q_nope.shape[0]
        # One batch entry per query row, so that a draft-verify step's rows can
        # each state how much of their own index list the operator should read.
        per_row_block_table, query_lens = rows_as_batch_entries(
            decode.block_table, num_tokens, decode.seq_lens.shape[0]
        )
        attn_output = self._sparse_attention(
            q_nope.view(num_tokens, self.num_heads, -1),
            k_nope,
            token_ids,
            per_row_block_table,
            query_lens,
            # How many ids each row actually selected, not how many tokens the
            # request knows: the operator reads that many entries of the index
            # list. A probe on an Ascend 950 showed the sequence length instead
            # both stops short of the tail and pulls in unfilled budget slots.
            selected_lens,
            decode.nope_zero_rope_cache,
        )
        return self._v_up_proj(attn_output)


def _single_tensor(cache, name: str) -> torch.Tensor:
    """Unwrap a cache layer's handle, a tensor or a one-tensor tuple."""
    if cache is None:
        raise RuntimeError(f"The kpool {name} cache is not bound to its layer.")
    if isinstance(cache, torch.Tensor):
        return cache
    if len(cache) != 1:
        raise RuntimeError(f"The kpool {name} cache should own one tensor, got {len(cache)}.")
    return cache[0]


def _group_slot_mapping(per_layer: dict, prefix: str, name: str) -> torch.Tensor:
    metadata = per_layer.get(prefix)
    if metadata is None or getattr(metadata, "slot_mapping", None) is None:
        raise RuntimeError(
            f"The kpool {name} cache group published no slot mapping under {prefix!r}. "
            "Its cache layer's attention backend is not the one that builds it."
        )
    return metadata.slot_mapping
