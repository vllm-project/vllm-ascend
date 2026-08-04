#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
import torch_npu
from vllm.forward_context import is_forward_context_available

from vllm_ascend._310p.attention.metadata_builder import get_query_lens_cpu
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

# Scope of this route. TP is not pinned: it only shards heads across ranks and
# does not change the numerics, so what matters is the per-rank layout, checked
# structurally below.
FIA_BLOCK_SIZE = 128  # the only KV page size the kernel's block selection covers
# headDim <= 256 and headDim * blockSize <= 16384 are the operator's own tiling
# limits; the head count is not a kernel limit but the envelope its unit test
# covers (1..16 heads, MHA and GQA), so it is where this scope stops.
FIA_MAX_HEAD_DIM = 256
FIA_MAX_HEAD_DIM_X_BLOCK = 16384  # with block=128 this caps head_dim at 128
FIA_MAX_NUM_HEADS = 16
FIA_SUPPORTED_METHODS = {"dflash", "dspark"}
# config.json architecture strings (the vLLM registry keys), not model class
# names -- matching against the class names never fires.
FIA_SUPPORTED_ARCHITECTURES = {"DFlashDraftModel", "Qwen3DSparkModel"}

def expected_queries_per_request(method, num_speculative_tokens):
    """Queries each request issues per draft step.

    DFlash prepends an anchor to the K mask tokens, DSpark does not.
    """
    return num_speculative_tokens + 1 if method == "dflash" else num_speculative_tokens


def _fia_op():
    """Resolve the custom FIA op wrapper.

    The kernel is built only for the ascend310 SOC branch, so a wheel produced
    for another target imports fine and simply does not carry the symbol. Going
    through the operator's own wrapper rather than torch.ops keeps the argument
    marshalling in one place and picks up @allow_in_graph.
    """
    if not hasattr(torch.ops._C_ascend, "npu_custom_fused_infer_attention_v310"):
        raise RuntimeError(
            "310P parallel draft attention needs "
            "torch.ops._C_ascend.npu_custom_fused_infer_attention_v310, which is not "
            "registered. Rebuild with SOC_VERSION=ascend310p1. There is no fallback: "
            "the causal split-fuse kernel would return plausible but wrong numbers."
        )
    from vllm_ascend._310p.ops.custom_fused_infer_attention import (
        custom_fused_infer_attention_v310,
    )

    return custom_fused_infer_attention_v310


def validate_fia_scope(*, vllm_config, query, key_cache, value_cache, num_heads, num_kv_heads, head_size):
    """Check every startup invariant once, before the first custom FIA call."""
    spec_config = vllm_config.speculative_config
    method = getattr(spec_config, "method", None)
    if spec_config is None or method not in FIA_SUPPORTED_METHODS:
        raise RuntimeError(
            f"custom FIA draft attention only covers {sorted(FIA_SUPPORTED_METHODS)}, got {method}"
        )
    num_spec = spec_config.num_speculative_tokens
    if num_spec < 1:
        raise RuntimeError(f"num_speculative_tokens must be >= 1, got {num_spec}")

    hf_config = spec_config.draft_model_config.hf_config
    architectures = getattr(hf_config, "architectures", None) or []
    arch = architectures[0] if architectures else None
    if arch not in FIA_SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            f"draft architecture {arch} is outside this scope "
            f"({sorted(FIA_SUPPORTED_ARCHITECTURES)})"
        )

    # This route passes attn_mask=None, which the kernel maps to NO_MASK: every
    # query row sees the whole [0, kv_len) range. That is only correct for a
    # drafter whose layers are all full non-causal attention. A causal or
    # sliding-window checkpoint would still run and return plausible numbers, so
    # refuse it here rather than discover it as an accuracy loss.
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None and any(t != "full_attention" for t in layer_types):
        raise RuntimeError(
            f"this route sends attn_mask=None (non-causal); draft layer_types "
            f"{sorted(set(layer_types))} include something other than full_attention"
        )
    if getattr(hf_config, "use_sliding_window", False) or getattr(hf_config, "sliding_window", None):
        raise RuntimeError(
            "this route sends attn_mask=None (non-causal); the draft checkpoint "
            "requests sliding-window attention, which it cannot express"
        )
    if getattr(getattr(hf_config, "dflash_config", None), "causal", False):
        raise RuntimeError("this route sends attn_mask=None; the draft checkpoint asks for causal attention")

    # A draft step's query rows must fit in one diffusion block. DFlash spends
    # one row on the anchor and DSpark does not, so this caps DFlash at
    # block_size - 1 speculative tokens and DSpark at block_size. Unrelated to
    # the CLI --block-size, which is the KV page size (FIA_BLOCK_SIZE).
    diffusion_block = getattr(hf_config, "block_size", None)
    if diffusion_block:
        q = expected_queries_per_request(method, num_spec)
        if q > diffusion_block:
            raise RuntimeError(
                f"num_speculative_tokens={num_spec} needs {q} query rows per request, "
                f"more than the checkpoint's diffusion block holds ({diffusion_block}). "
                f"The ceiling is {diffusion_block - 1 if method == 'dflash' else diffusion_block} "
                f"for {method}."
            )

    # No engine-wide eager check here: "target in ACLGraph, drafter eager" is a
    # supported configuration, and enforce_eager describes the engine rather than
    # this route. What must not happen is checked per call below.

    # Per-rank head layout, constrained structurally rather than by a fixed count.
    # TP is not checked: it only shards these heads across ranks and does not
    # change the numerics, so any TP whose resulting layout satisfies the rules
    # below is fine (e.g. Qwen3-8B is 32Q/8KV -> 16/4 at TP=2, 8/2 at TP=4).
    if not 0 < head_size <= FIA_MAX_HEAD_DIM:
        raise RuntimeError(f"custom FIA requires 0 < head_dim <= {FIA_MAX_HEAD_DIM}, got {head_size}")
    if head_size * FIA_BLOCK_SIZE > FIA_MAX_HEAD_DIM_X_BLOCK:
        # With the fixed 128-token block this caps head_dim at 128.
        raise RuntimeError(
            f"head_dim * block_size = {head_size * FIA_BLOCK_SIZE} exceeds the kernel's "
            f"{FIA_MAX_HEAD_DIM_X_BLOCK}"
        )
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise RuntimeError(f"invalid GQA layout: {num_heads} query heads / {num_kv_heads} KV heads")
    if num_heads > FIA_MAX_NUM_HEADS:
        # Qwen3-8B is 32 query heads, so TP=1 lands outside the covered envelope
        # while TP=2 (16) and TP=4 (8) are inside it.
        raise RuntimeError(
            f"per-rank query heads {num_heads} exceeds the {FIA_MAX_NUM_HEADS} this scope "
            f"covers; raise the tensor-parallel size so each rank holds fewer heads"
        )
    if (num_kv_heads * head_size) % 16 != 0:
        raise RuntimeError(
            f"NZ alignment: num_kv_heads * head_dim = {num_kv_heads * head_size} must be a "
            f"multiple of 16"
        )

    for name, tensor in (("query", query), ("key_cache", key_cache), ("value_cache", value_cache)):
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"custom FIA on 310P only supports float16 in this scope, but {name} is "
                f"{tensor.dtype}. Start the engine with dtype=float16."
            )

    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError(f"custom FIA needs rank-4 NZ K/V caches, got {key_cache.ndim}/{value_cache.ndim}")
    if key_cache.shape != value_cache.shape:
        raise RuntimeError(f"K/V cache shapes differ: {key_cache.shape} vs {value_cache.shape}")
    if key_cache.device != value_cache.device:
        raise RuntimeError(f"K/V caches are on different devices: {key_cache.device} vs {value_cache.device}")

    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        fmt = int(torch_npu.get_npu_format(cache))
        if fmt != ACL_FORMAT_FRACTAL_NZ:
            raise RuntimeError(
                f"{name} is in acl format {fmt}, expected ACL_FORMAT_FRACTAL_NZ "
                f"({ACL_FORMAT_FRACTAL_NZ}); the kernel reads the NZ layout directly"
            )

    if key_cache.shape[-1] != 16:
        raise RuntimeError(f"NZ cache last dim must be 16, got {key_cache.shape[-1]}")
    # Compared against the scope constant, not against a value derived from the
    # cache itself -- the latter would be tautological.
    if key_cache.shape[-2] != FIA_BLOCK_SIZE:
        raise RuntimeError(
            f"cache physical block size is {key_cache.shape[-2]}, this scope only covers "
            f"{FIA_BLOCK_SIZE}"
        )
    expected_dim1 = num_kv_heads * head_size // 16
    if key_cache.shape[1] != expected_dim1:
        raise RuntimeError(
            f"NZ cache dim1 is {key_cache.shape[1]}, expected num_kv_heads*head_size/16 = {expected_dim1}"
        )


def forward_parallel_draft_fia(self, query, attn_metadata, output):
    """Non-causal parallel-draft attention via the in-tree custom FIA op.

    ``attn_mask=None`` is what makes the kernel non-causal: its host tiling maps an empty
    mask to NO_MASK and the kernel neither loads nor applies one, so every query row
    sees the full ``[0, actual_seq_lengths_kv[b])`` range -- context plus this
    round's entire query block. Never pass the 310P compressed split-fuse mask here,
    and never synthesize an all-zero causal mask to imitate it.
    """
    fia_op = _fia_op()

    num_tokens = int(attn_metadata.num_actual_tokens)
    query_slice = query[:num_tokens]
    output_slice = output[:num_tokens]

    # Raw per-request q-lens come from the 310P builder, which diffed the CPU
    # endpoints outside the forward. The tensor is host/pinned, so .tolist() costs
    # no device sync. Never rebuild these from the base metadata's
    # actual_seq_lengths_q (cumulative endpoints) or from max_query_len.
    raw_q_lens = get_query_lens_cpu(attn_metadata)
    if raw_q_lens is None:
        raise RuntimeError(
            "310P parallel draft attention needs raw per-request query lengths on "
            "attn_metadata, but query_lens_cpu is missing. It is set by "
            "AscendAttentionMetadataBuilder310.build() for ChunkedPrefill/SpecDecoding; "
            "check that the draft metadata went through that builder."
        )
    q_lens = raw_q_lens.tolist()
    kv_lens = attn_metadata.seq_lens_list

    # actual_seq_lengths_q / _kv are host SymInt[] read at dispatch, so capturing
    # this op would freeze one step's lengths into the graph and replay them for
    # every later step -- plausible numbers, silently wrong. The availability
    # check must short-circuit: reading _EXTRA_CTX with no forward context raises
    # "Forward context is not set".
    if is_forward_context_available() and _EXTRA_CTX.capturing:
        raise RuntimeError(
            "310P parallel draft attention was reached during ACLGraph capture. "
            "This op reads host sequence lengths at dispatch and cannot be captured; "
            "the drafter must stay eager. Capturing the target model is supported."
        )

    method = getattr(self.vllm_config.speculative_config, "method", None)
    if method not in FIA_SUPPORTED_METHODS:
        raise RuntimeError(f"custom FIA draft attention reached with unsupported method {method}")

    # Scope first, shapes second. The q-len check below assumes the configuration
    # is in scope, so if K is out of scope it reports a wrong q-len and blames the
    # metadata -- which is what an out-of-scope K=15 DFlash run used to hit,
    # pointing at cumulative endpoints instead of at the K it was launched with.
    if not self._fia_scope_validated:
        validate_fia_scope(
            vllm_config=self.vllm_config,
            query=query_slice.reshape(num_tokens, self.num_heads, self.head_size),
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
        )
        self._fia_scope_validated = True

    expected_q = expected_queries_per_request(
        method, self.vllm_config.speculative_config.num_speculative_tokens
    )
    if any(q != expected_q for q in q_lens):
        raise RuntimeError(
            f"{method} expects every request to query {expected_q} positions, got {q_lens}. "
            f"A cumulative-endpoint tensor was most likely passed instead of raw "
            f"per-request lengths."
        )
    if sum(q_lens) != num_tokens or query_slice.shape[0] != num_tokens:
        raise RuntimeError(
            f"sum(q_lens)={sum(q_lens)}, num_actual_tokens={num_tokens}, query rows="
            f"{query_slice.shape[0]} must all agree"
        )

    block_table = attn_metadata.block_tables[: len(q_lens)]
    if len(kv_lens) != len(q_lens) or block_table.shape[0] != len(q_lens):
        raise RuntimeError(
            f"batch size disagreement: {len(q_lens)} q-lens, {len(kv_lens)} kv-lens, "
            f"{block_table.shape[0]} block-table rows"
        )
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise RuntimeError(
            f"block table must be a rank-2 int32 tensor, got ndim={block_table.ndim} "
            f"dtype={block_table.dtype}"
        )

    # Fixed by scope rather than read back from the cache; validate_fia_scope
    # checks the cache's physical block size against this same constant.
    capacity = block_table.shape[1] * FIA_BLOCK_SIZE
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        if not 0 < q_len <= kv_len:
            raise RuntimeError(f"request {b}: need 0 < q_len({q_len}) <= kv_len({kv_len})")
        if kv_len > capacity:
            raise RuntimeError(
                f"request {b}: kv_len {kv_len} exceeds what its block table can address "
                f"({block_table.shape[1]} pages x {FIA_BLOCK_SIZE})"
            )

    key_cache = self.key_cache
    value_cache = self.value_cache
    query_tnd = query_slice.reshape(num_tokens, self.num_heads, self.head_size)

    # The op allocates its own output and has no `out=` overload, so this path
    # eats one copy. Do not add an `_out` variant locally to avoid it: the ABI is
    # owned by the shared op, and forking it here is how the two drift.
    fia_out = fia_op(
        query_tnd,
        key_cache,
        value_cache,
        attn_mask=None,
        actual_seq_lengths_q=q_lens,
        actual_seq_lengths_kv=kv_lens,
        block_table=block_table,
        num_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        block_size=FIA_BLOCK_SIZE,
        input_layout="TND",
        scale_value=self.scale,
        # inner_precise is not a wrapper argument: it pins it to 2 internally.
        # Every other value is passed explicitly rather than left to a default,
        # so a change to the wrapper's defaults cannot move our numerics silently
        # -- input_layout's default already went from "BSH" to "BSND".
    )

    # The op's contract is "output has the same shape as query". Check that rather
    # than numel against the possibly flat output slice: a numel match would also
    # accept a transposed or mis-headed result.
    if tuple(fia_out.shape) != tuple(query_tnd.shape) or fia_out.dtype != query_tnd.dtype:
        raise RuntimeError(
            f"custom FIA returned {tuple(fia_out.shape)}/{fia_out.dtype}, expected the "
            f"query shape {tuple(query_tnd.shape)}/{query_tnd.dtype}"
        )

    output_slice.copy_(fia_out.reshape(output_slice.shape))
    return output
