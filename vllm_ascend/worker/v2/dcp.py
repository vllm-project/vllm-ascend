# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host metadata for the MLA DCP backend used by ModelRunner V2."""

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, AscendDCPMetadata, split_decodes_and_prefills
from vllm_ascend.utils import is_pd_decode_recompute_scheduler_enabled


def prepare_mla_dcp_metadata(
    common: AscendCommonAttentionMetadata,
    builder: AscendMlaDCPMetadataBuilder,
) -> None:
    """Decode reads the complete local KV; prefill reads only cached history.

    Use the same request classification as the MLA builder, including short
    prefills and graph padding. The CPU lengths must describe this execution,
    rather than the scheduler's speculative upper bound.
    """
    num_decodes, _, _, _ = split_decodes_and_prefills(
        common,
        decode_threshold=builder.decode_threshold,
        treat_short_extends_as_decodes=is_pd_decode_recompute_scheduler_enabled(builder.vllm_config),
    )
    query_lens = common.query_start_loc_cpu[1:] - common.query_start_loc_cpu[:-1]
    seq_lens = common.seq_lens_cpu[: common.num_reqs]
    history_lens = (seq_lens - query_lens).clamp(min=0)
    context_lens = history_lens.clone()
    context_lens[:num_decodes] = seq_lens[:num_decodes]
    local_lens = get_dcp_local_seq_lens(
        context_lens, dcp_size=builder.dcp_size, cp_kv_cache_interleave_size=builder.cp_local_block_size
    )
    common.num_computed_tokens_cpu = history_lens
    common._num_computed_tokens_cpu = history_lens
    common.context_parallel_metadata = AscendDCPMetadata(
        num_computed_tokens_of_dcp=local_lens.numpy(),
        query_lens_cpu=query_lens,
        max_query_len=int(query_lens.max()) if query_lens.numel() else 0,
    )
