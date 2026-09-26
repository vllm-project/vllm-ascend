# SPDX-License-Identifier: Apache-2.0
"""Controlled MTP state coverage; synthetic acceptance is not accuracy evidence."""

import dataclasses
import json

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize("graph_mode", [None])
def test_qrep_mtp_controlled_acceptance(graph_mode, monkeypatch, dcp_qrep_sparse_mtp_model, tmp_path):
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    outputs = []
    for enabled in (False, True):
        with VllmRunner(
            dcp_qrep_sparse_mtp_model,
            dtype="bfloat16",
            tensor_parallel_size=4,
            decode_context_parallel_size=4,
            dcp_q_replicate=enabled,
            enforce_eager=graph_mode is None,
            compilation_config={"cudagraph_mode": graph_mode, "cudagraph_capture_sizes": [1, 2, 4, 8, 16]}
            if graph_mode
            else {},
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 3,
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_rates": [1.0, 0.0, 0.0],
            },
            max_model_len=2048,
            max_num_seqs=3,
            max_num_batched_tokens=128,
            block_size=128,
            cp_kv_cache_interleave_size=128,
            attention_config={"indexer_kv_dtype": "bf16"},
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            disable_log_stats=False,
            additional_config={"enable_mlapo": False, "enable_dsa_cp": False},
        ) as runner:
            prompts = [
                "The capital of France is",
                "Explain context parallel attention. " * 40,
                "Continue: " + "one two three " * 30,
            ]
            params = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
            before = {m.name: m.value for m in runner.model.get_metrics() if hasattr(m, "value")}
            first = runner.model.generate(prompts, params)
            again = runner.model.generate(prompts, params)
            tokens = [o.outputs[0].token_ids for o in first]
            stats = [dataclasses.asdict(m) for m in runner.model.get_metrics() if "spec_decode" in m.name]
            deltas = {m["name"]: m["value"] - before.get(m["name"], 0) for m in stats if "value" in m}
            record = dict(
                qrep=enabled,
                graph=graph_mode,
                synthetic_acceptance_rates=[1, 0, 0],
                deltas=deltas,
                stats=stats,
                tokens=tokens,
            )
            (tmp_path / f"mtp-controlled-{graph_mode}-{enabled}.json").write_text(json.dumps(record, indent=2))
            print("MTP_CONTROLLED " + json.dumps(record), flush=True)
            assert tokens == [o.outputs[0].token_ids for o in again]
            accepted = deltas["vllm:spec_decode_num_accepted_tokens"]
            proposed = deltas["vllm:spec_decode_num_draft_tokens"]
            assert 0 < accepted < proposed
            assert all(len(o.outputs[0].token_ids) == 24 for o in first + again)
            outputs.append(tokens)
    assert outputs[0] == outputs[1]
