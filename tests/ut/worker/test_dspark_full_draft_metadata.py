# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.worker.v2.spec_decode.dspark import speculator as dspark


@pytest.mark.parametrize(
    "scenario", ["match", "shape", "length_source", "boundaries", "eager", "other_backend", "pcp", "dcp", "exception"]
)
def test_full_draft_metadata_reused_only_within_proposal(monkeypatch, scenario):
    spec = dspark.AscendDSparkSpeculator.__new__(dspark.AscendDSparkSpeculator)
    spec.pcp_manager = object() if scenario == "pcp" else None
    spec.use_dcp = scenario == "dcp"
    spec.attn_architecture = None
    spec._uses_v41_draft_backend = scenario != "other_backend"
    spec.arange_np = np.arange(5, dtype=np.int32)
    spec.num_query_per_req = 5
    spec._group_causal = {0: False}
    lengths = torch.tensor([20, 30], dtype=torch.int32)
    spec.input_batch = SimpleNamespace(num_reqs=2)
    spec.input_buffers = SimpleNamespace(positions=torch.arange(20))
    monkeypatch.setattr(dspark.AscendDSparkSpeculator, "attn_vllm_config", SimpleNamespace(parallel_config=None))
    monkeypatch.setattr(dspark, "build_attn_metadata_wrapper", nullcontext)
    factory = Mock(side_effect=lambda *a, **kw: nullcontext())
    monkeypatch.setattr(dspark, "build_attn_metadata_factory", factory)
    if scenario == "dcp":
        monkeypatch.setattr(spec, "_prepare_draft_dcp_metadata_inputs", lambda *a: (None, None))
    metadata = {"layer": object()}
    build = Mock(return_value=metadata)
    monkeypatch.setattr(DSparkSpeculator, "_build_attn_metadata", build)
    fallback = Mock(return_value={"rebuilt": object()})
    monkeypatch.setattr(spec, "_build_uniform_attn_metadata", fallback)

    def prepare():
        return spec._build_attn_metadata(
            2,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE if scenario == "eager" else CUDAGraphMode.FULL,
                num_tokens=20,
                num_reqs=4,
            ),
            np.array([0, 4 if scenario == "boundaries" else 5, 10], dtype=np.int32),
            lengths,
            step=5,
            causal=spec._group_causal,
        )

    if scenario == "exception":
        with pytest.raises(RuntimeError), spec._draft_metadata_scope():
            assert prepare() is metadata
            raise RuntimeError("proposal failed")
    else:
        with spec._draft_metadata_scope():
            assert prepare() is metadata
            assert build.call_count == 1
            result = spec.build_draft_attn_metadatas(
                3 if scenario == "shape" else 4,
                lengths.clone() if scenario == "length_source" else lengths,
            )
            if scenario == "match":
                assert result[0] is metadata
                factory.assert_called_once()
                positions, padded_tokens = factory.call_args.args
                flags = factory.call_args.kwargs["is_prefilling"]
                assert positions is spec.input_buffers.positions
                assert padded_tokens == 20
                assert flags.tolist() == [False] * 4
                assert factory.call_args.kwargs["attn_state"] == dspark.AscendAttentionState.ChunkedPrefill
                fallback.assert_not_called()
                assert spec.build_draft_attn_metadatas(4, lengths)[0] is fallback.return_value
            else:
                assert result[0] is fallback.return_value

    assert spec._full_draft_metadata is None
    assert not spec._reuse_full_draft_metadata
    assert spec.build_draft_attn_metadatas(4, lengths)[0] is fallback.return_value
