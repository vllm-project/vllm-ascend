# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.v2.spec_decode.standalone_ar import speculator as standalone_ar


@pytest.mark.parametrize(
    ("use_v2", "method", "parallel_drafting", "pcp_size", "dcp_size", "dtype", "k", "expected"),
    [
        (True, "draft_model", False, 1, 1, torch.bfloat16, 3, 5),
        (True, "draft_model", False, 1, 1, torch.float16, 14, 16),
        (True, "draft_model", False, 1, 1, torch.bfloat16, 15, 16),
        (False, "draft_model", False, 1, 1, torch.bfloat16, 3, 4),
        (True, "mtp", False, 1, 1, torch.bfloat16, 3, 4),
        (True, "draft_model", True, 1, 1, torch.bfloat16, 3, 4),
        (True, "draft_model", False, 2, 1, torch.bfloat16, 3, 4),
        (True, "draft_model", False, 1, 2, torch.bfloat16, 3, 4),
        (True, "draft_model", False, 1, 1, torch.int8, 3, 4),
    ],
)
def test_standalone_ar_decode_and_prefill_split(
    use_v2, method, parallel_drafting, pcp_size, dcp_size, dtype, k, expected
):
    config = cast(
        VllmConfig,
        SimpleNamespace(
            use_v2_model_runner=use_v2,
            speculative_config=SimpleNamespace(
                method=method,
                num_speculative_tokens=k,
                parallel_drafting=parallel_drafting,
                max_num_new_slots_for_drafting=k if parallel_drafting else 1,
            ),
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=pcp_size, decode_context_parallel_size=dcp_size
            ),
            model_config=SimpleNamespace(max_model_len=128),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
            compilation_config=None,
        ),
    )
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=dtype)
    builder = AscendAttentionMetadataBuilder(spec, ["attn"], config, torch.device("cpu"))

    # One single-token decode, one expanded verification query, one long prefill.
    expanded_query_len = k + 2
    prompt_query_len = expanded_query_len + 1
    total = 1 + expanded_query_len + prompt_query_len
    common = cast(
        AscendCommonAttentionMetadata,
        SimpleNamespace(
            context_parallel_metadata=None,
            max_query_len=prompt_query_len,
            num_reqs=3,
            num_actual_tokens=total,
            query_start_loc_cpu=torch.tensor([0, 1, 1 + expanded_query_len, total]),
            is_prefilling=torch.tensor([False, False, True]),
        ),
    )

    assert builder.reorder_batch_threshold == expected
    if expanded_query_len <= expected:
        assert builder._split_decodes_and_prefills(common) == (2, 1, 1 + expanded_query_len, prompt_query_len)
    else:
        assert builder._split_decodes_and_prefills(common) == (1, 2, 1, expanded_query_len + prompt_query_len)


@pytest.mark.parametrize(("upstream_adds_correction", "step"), [(False, 0), (True, 0), (False, 1), (True, 1)])
def test_standalone_ar_passes_sequence_length_with_correction_once(monkeypatch, upstream_adds_correction, step):
    monkeypatch.setattr(standalone_ar, "_UPSTREAM_STANDALONE_AR_GRAPHS", upstream_adds_correction)
    monkeypatch.setattr(standalone_ar, "build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(standalone_ar, "build_attn_metadata_factory", lambda *args, **kwargs: nullcontext())
    upstream_build = MagicMock()
    monkeypatch.setattr(standalone_ar.StandaloneARSpeculator, "_build_attn_metadata", upstream_build)

    proposer = standalone_ar.AscendStandaloneARSpeculator.__new__(standalone_ar.AscendStandaloneARSpeculator)
    monkeypatch.setattr(proposer, "max_model_len", 128, raising=False)
    monkeypatch.setattr(proposer, "expanded_positions", torch.empty(5), raising=False)
    monkeypatch.setattr(proposer, "input_buffers", SimpleNamespace(positions=torch.empty(1)), raising=False)
    monkeypatch.setattr(proposer, "draft_is_prefilling", torch.zeros(1, dtype=torch.bool), raising=False)
    monkeypatch.setattr(proposer, "vllm_config", SimpleNamespace(parallel_config=None), raising=False)

    width = 1 if step else 5
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=width, num_reqs=1)
    lengths = torch.tensor([41 if upstream_adds_correction else 40])
    original = lengths.clone()
    proposer._build_attn_metadata(1, desc, np.array([0, width], dtype=np.int32), lengths, step)

    passed_lengths = upstream_build.call_args.args[3]
    torch.testing.assert_close(passed_lengths, original if step else torch.tensor([41]))
    torch.testing.assert_close(lengths, original)
