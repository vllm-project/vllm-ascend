# SPDX-License-Identifier: Apache-2.0
"""Qrep feature overlays with observed cache hits and scheduler batches."""

import json

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from tests.e2e.conftest import VllmRunner


class OverlayWorkerExtension:
    def overlay_activation(self):
        model = self.model_runner.get_model()
        layers = [
            bool(getattr(m, "qrep_active", False))
            for name, m in model.named_modules()
            if name.rsplit(".", 1)[-1] in ("q_proj", "q_b_proj")
        ]
        return {"runner": type(self.model_runner).__module__, "qrep": layers}


class ScheduleEvidence:
    """Observe CPU scheduling metadata without changing scheduling decisions."""

    def _update_after_schedule(self, scheduler_output):
        rows = []
        for request_id, count in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            rows.append(
                {
                    "id": request_id,
                    "computed": request.num_computed_tokens,
                    "prompt": request.num_prompt_tokens,
                    "scheduled": count,
                    "in_flight": request.num_in_flight_tokens,
                }
            )
        if rows:
            print("QREP_SCHEDULE " + json.dumps(rows), flush=True)
        return super()._update_after_schedule(scheduler_output)


class ObservedScheduler(ScheduleEvidence, Scheduler):
    pass


class ObservedAsyncScheduler(ScheduleEvidence, AsyncScheduler):
    pass


@pytest.mark.parametrize(
    "case",
    [
        "baseline",
        "chunk",
        "prefix_chunk",
        "p0_graph",
        "async_graph",
        "v2_eager",
        "v2_graph",
        "lmhead_tp",
    ],
)
def test_qrep_overlays(case, monkeypatch, dcp_qrep_model, tmp_path):
    model = dcp_qrep_model
    extended_len = 1537
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if case.startswith("v2_") else "0")
    prefix = case not in ("baseline", "chunk")
    chunk = case != "baseline"
    graph = case in ("p0_graph", "async_graph", "v2_graph")
    asynchronous = case == "async_graph"
    results = []
    repeat_matches = []
    for enabled in (False, True):
        additional = {"enable_mlapo": False, "enable_dsa_cp": False}
        if case == "lmhead_tp":
            additional["finegrained_tp_config"] = {"lmhead_tensor_parallel_size": 1}
        with VllmRunner(
            model,
            dtype="bfloat16",
            tensor_parallel_size=4,
            worker_extension_cls=f"{__name__}.OverlayWorkerExtension",
            decode_context_parallel_size=2,
            dcp_q_replicate=enabled,
            enforce_eager=not graph,
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8],
            }
            if graph
            else {},
            max_model_len=2048,
            max_num_seqs=3,
            max_num_batched_tokens=128 if chunk else 2048,
            block_size=128,
            enable_chunked_prefill=chunk,
            enable_prefix_caching=prefix,
            async_scheduling=asynchronous,
            scheduler_cls=f"{__name__}.{'ObservedAsyncScheduler' if asynchronous else 'ObservedScheduler'}",
            additional_config=additional,
        ) as runner:
            activation = runner.model.collective_rpc("overlay_activation")
            print("QREP_OVERLAY_ACTIVATION " + json.dumps(activation), flush=True)
            assert all(r["qrep"] and all(v == enabled for v in r["qrep"]) for r in activation)
            assert all((".v2." in r["runner"]) == case.startswith("v2_") for r in activation)
            tokenizer = runner.model.get_tokenizer()
            seed = tokenizer.encode(
                "Context parallel attention distributes the key value cache across devices. " * 400,
                add_special_tokens=False,
            )
            assert len(seed) > extended_len
            warm_prompt = {"prompt_token_ids": seed[:1024]}
            extended_prompt = {"prompt_token_ids": seed[:extended_len]}
            short_prompt = {"prompt_token_ids": tokenizer.encode("The capital of France is", add_special_tokens=False)}
            params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
            runner.model.reset_prefix_cache()
            warm = runner.model.generate([warm_prompt], params)
            # Full-block hit, followed by allocation of an uncached suffix.
            repeated = runner.model.generate([warm_prompt], params)
            mixed = runner.model.generate([short_prompt, extended_prompt], params)
            # A second mixed batch uses a different batch size and block boundary.
            again = runner.model.generate([extended_prompt], params)
            batches = [warm, repeated, mixed, again]
            record = {
                "case": case,
                "qrep": enabled,
                "prefix": prefix,
                "chunk": chunk,
                "graph": graph,
                "async": asynchronous,
                "batches": [
                    [
                        {
                            "ids": o.outputs[0].token_ids,
                            "cached": o.num_cached_tokens,
                            "prompt_len": len(o.prompt_token_ids),
                            "logprobs": [
                                {token: value.logprob for token, value in step.items()}
                                for step in (o.outputs[0].logprobs or [])
                            ],
                        }
                        for o in batch
                    ]
                    for batch in batches
                ],
            }
            (tmp_path / f"{case}-{enabled}.json").write_text(json.dumps(record, indent=2))
            print("QREP_OVERLAY " + json.dumps(record), flush=True)
            # Collect both Qrep states before reporting repeat discrepancies.
            repeat_matches.append(warm[0].outputs[0].token_ids == repeated[0].outputs[0].token_ids)
            repeat_matches.append(mixed[1].outputs[0].token_ids == again[0].outputs[0].token_ids)
            assert (warm[0].num_cached_tokens or 0) == 0
            if prefix:
                assert repeated[0].num_cached_tokens >= 256
                assert 256 <= mixed[1].num_cached_tokens < extended_len - 128
                assert again[0].num_cached_tokens >= mixed[1].num_cached_tokens
            else:
                assert all((o.num_cached_tokens or 0) == 0 for batch in batches for o in batch)
            results.append([[o.outputs[0].token_ids for o in batch] for batch in batches])
    assert results[0] == results[1]
    assert all(repeat_matches), "Prefix-cache repeat output differs; inspect both Qrep states"
