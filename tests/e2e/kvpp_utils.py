# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict

from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

BLOCK_SIZE = 128
NUM_BLOCKS = 64
MAX_TOKENS = 16


def read_kvpp_worker_state(worker):
    from vllm.distributed import get_pp_group

    from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
    from vllm_ascend.core.kv_cache_placement import find_mtp_layers
    from vllm_ascend.distributed.parallel_state import get_kvpp_group
    from vllm_ascend.worker.kvpp_cache import get_kvpp_cache_specs

    runner = worker.model_runner
    group = get_kvpp_group()
    specs = get_kvpp_cache_specs(runner.kv_cache_config)
    context = worker.vllm_config.compilation_config.static_forward_context
    scheduler = runner.kvpp.scheduler
    mtp_layers = find_mtp_layers(worker.vllm_config, specs)
    return {
        "runner_module": type(runner).__module__,
        "stage": get_pp_group().rank_in_group,
        "rank": group.rank_in_group,
        "ranks": list(group.ranks),
        "targets": list(scheduler.attention_layer_names) if scheduler is not None else [],
        "expected_targets": sorted(
            name
            for name, spec in specs.items()
            if name not in mtp_layers and not isinstance(spec, AscendSFAIndexerCacheSpec)
        ),
        "num_blocks": runner.kv_cache_config.num_blocks,
        "mtp": sorted(mtp_layers),
        "indexer_c8": [
            (
                name,
                spec.scale_dim,
                str(spec.scale_dtype),
                [(str(view.dtype), list(view.shape)) for view in context[name].kv_cache],
            )
            for name, spec in specs.items()
            if getattr(spec, "cache_sparse_li_c8", False)
        ],
        "packed_main": [name for name, spec in specs.items() if getattr(spec, "cache_sparse_sfa_c8", False)],
    }


def assert_worker_state(runner, enabled, tp, pp, num_blocks, *, v2=False, mtp=False, indexer_c8=False):
    states = runner.model.collective_rpc(read_kvpp_worker_state)
    expected_runner = "vllm_ascend.worker.v2.model_runner" if v2 else "vllm_ascend.worker.model_runner_v1"
    assert {state["runner_module"] for state in states} == {expected_runner}
    assert len(states) == tp * pp
    assert {state["stage"] for state in states} == set(range(pp))
    for stage in range(pp):
        workers = [state for state in states if state["stage"] == stage]
        assert len(workers) == tp
        assert {state["num_blocks"] for state in workers} == {num_blocks}
        if enabled:
            assert {state["rank"] for state in workers} == set(range(tp))
            assert {tuple(state["ranks"]) for state in workers} == {tuple(range(stage * tp, (stage + 1) * tp))}
            assert all(sorted(state["targets"]) == state["expected_targets"] for state in workers)
        else:
            assert all(not state["targets"] and len(state["ranks"]) == 1 for state in workers)
    if mtp:
        assert any(state["mtp"] for state in states)
        assert all(set(state["mtp"]).isdisjoint(state["targets"]) for state in states)
    if indexer_c8:
        assert all(state["indexer_c8"] for state in states)
        for state in states:
            for _name, scale_dim, scale_dtype, views in state["indexer_c8"]:
                assert scale_dim > 0
                assert len(views) == 2
                assert views[0][0] in ("torch.int8", "torch.float8_e4m3fn")
                assert views[1][0] == scale_dtype
                assert views[1][1][-1] == scale_dim
                assert views[0][1][0] == num_blocks
        # This model's reference configuration quantizes LI, not the main cache.
        assert all(not state["packed_main"] for state in states)
    return states


class SchedulerTrace:
    """Observe real scheduling and written block IDs without changing execution."""

    def __init__(self, runner, monkeypatch):
        engine = runner.model.llm_engine
        scheduler = engine.engine_core.engine_core.scheduler
        manager = scheduler.kv_cache_manager
        block_sizes = [group.kv_cache_spec.block_size for group in manager.kv_cache_config.kv_cache_groups]
        null_block = manager.block_pool.null_block.block_id
        self.steps = defaultdict(list)
        self.written_blocks = defaultdict(set)
        original_schedule = scheduler.schedule

        def schedule(*args, **kwargs):
            output = original_schedule(*args, **kwargs)
            starts = {request.req_id: request.num_computed_tokens for request in output.scheduled_new_reqs}
            cached = output.scheduled_cached_reqs
            starts.update(zip(cached.req_ids, cached.num_computed_tokens))
            for request_id, count in output.num_scheduled_tokens.items():
                external_id = engine.output_processor.request_states[request_id].external_req_id
                start = starts[request_id]
                prompt_len = len(scheduler.requests[request_id].prompt_token_ids)
                self.steps[external_id].append((start, count, prompt_len))
                for group_id, (ids, block_size) in enumerate(zip(manager.get_block_ids(request_id), block_sizes)):
                    end_block = (start + count + block_size - 1) // block_size
                    touched = ids[start // block_size : end_block]
                    self.written_blocks[external_id].update(
                        (group_id, block_id) for block_id in touched if block_id != null_block
                    )
            return output

        monkeypatch.setattr(scheduler, "schedule", schedule)

    def assert_chunked_prefill_and_decode(self, request_id):
        steps = self.steps[request_id]
        prefill = [(start, count, length) for start, count, length in steps if start < length]
        assert len(prefill) >= 2, steps
        assert any(0 < start < length for start, _, length in prefill), steps
        assert any(start >= length for start, _, length in steps), steps


def token_prompt(tokenizer, text, length):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    assert tokens
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


def generate(runner, prompts):
    outputs = runner.model.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        SamplingParams(temperature=0, ignore_eos=True, max_tokens=MAX_TOKENS),
        use_tqdm=False,
    )
    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.finished
        assert len(output.outputs) == 1
        assert len(output.outputs[0].token_ids) == MAX_TOKENS
        assert output.outputs[0].finish_reason == "length"
    return outputs


def compare_outputs(baseline, actual):
    assert [output.prompt_token_ids for output in baseline] == [output.prompt_token_ids for output in actual]
    check_outputs_equal(
        outputs_0_lst=[(list(output.outputs[0].token_ids), output.outputs[0].text) for output in baseline],
        outputs_1_lst=[(list(output.outputs[0].token_ids), output.outputs[0].text) for output in actual],
        name_0="KVPP off",
        name_1="KVPP on",
    )


def run_basic_comparison(
    monkeypatch,
    model,
    *,
    tp=2,
    pp=1,
    v2=False,
    mtp=False,
    indexer_c8=False,
    additional_config=None,
    block_size=BLOCK_SIZE,
    quantization=None,
):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(v2)))
    model = maybe_model_redirect(model)
    token_budget = max(block_size, 128 if mtp else 64)
    results = []
    for enabled in (False, True):
        with VllmRunner(
            model,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enforce_eager=True,
            async_scheduling=True,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            max_model_len=512,
            max_num_seqs=4,
            max_num_batched_tokens=token_budget,
            block_size=block_size,
            num_gpu_blocks_override=NUM_BLOCKS,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.8,
            seed=42,
            disable_log_stats=False,
            quantization=quantization,
            dtype="bfloat16" if quantization is None else "auto",
            speculative_config={"method": "mtp", "num_speculative_tokens": 1, "enforce_eager": True} if mtp else None,
            additional_config={**(additional_config or {}), "enable_kvpp": enabled},
        ) as runner:
            assert_worker_state(runner, enabled, tp, pp, NUM_BLOCKS, v2=v2, mtp=mtp, indexer_c8=indexer_c8)
            tokenizer = runner.model.get_tokenizer()
            prompts = [
                token_prompt(tokenizer, "Explain how a computer stores information. ", 16),
                token_prompt(tokenizer, "Describe rivers, mountains and forests in detail. ", 3 * token_budget),
            ]
            with monkeypatch.context() as observation:
                trace = SchedulerTrace(runner, observation)
                outputs = generate(runner, prompts)
                if not mtp and pp == 1:
                    trace.assert_chunked_prefill_and_decode(outputs[1].request_id)
            if mtp:
                drafts = [
                    metric.value
                    for metric in runner.model.get_metrics()
                    if metric.name == "vllm:spec_decode_num_drafts"
                ]
                assert drafts and sum(drafts) > 0
            results.append(outputs)
    compare_outputs(*results)
