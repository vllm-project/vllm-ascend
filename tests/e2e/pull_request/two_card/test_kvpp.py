# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict

import pytest
from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

MODEL = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"
TP_SIZE = 2
BLOCK_SIZE = 128
NUM_BLOCKS = 64
TOKEN_BUDGET = BLOCK_SIZE
PREFIX_LENGTH = 3 * BLOCK_SIZE
SUFFIX_LENGTH = 16
MAX_TOKENS = 16

pytestmark = pytest.mark.e2e_model(MODEL)


def read_worker_state(worker):
    # Resolve worker-local objects inside the executor process.
    from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
    from vllm_ascend.core.kv_cache_placement import find_mtp_layers
    from vllm_ascend.distributed import parallel_state
    from vllm_ascend.worker.kvpp_cache import get_kvpp_cache_specs

    runner = worker.model_runner
    config = worker.vllm_config
    group = parallel_state._KVPP
    specs = get_kvpp_cache_specs(runner.kv_cache_config)
    mtp = find_mtp_layers(config, specs)
    scheduler = runner.kvpp.scheduler
    return {
        "rank": group.rank_in_group if group is not None else None,
        "ranks": tuple(group.ranks) if group is not None else (),
        "targets": set(scheduler.attention_layer_names) if scheduler is not None else set(),
        "expected_targets": {
            name for name, spec in specs.items() if name not in mtp and not isinstance(spec, AscendSFAIndexerCacheSpec)
        },
        "mtp": mtp,
        "num_blocks": runner.kv_cache_config.num_blocks,
        "ep": config.parallel_config.enable_expert_parallel,
        "async": config.scheduler_config.async_scheduling,
    }


def assert_worker_state(runner, enabled):
    states = runner.model.collective_rpc(read_worker_state)
    assert len(states) == TP_SIZE
    for state in states:
        assert state["ep"] and state["async"]
        assert state["num_blocks"] == NUM_BLOCKS
        assert state["mtp"] and state["mtp"].isdisjoint(state["targets"])
    if enabled:
        assert {state["rank"] for state in states} == set(range(TP_SIZE))
        assert {state["ranks"] for state in states} == {tuple(range(TP_SIZE))}
        assert all(state["targets"] and state["targets"] == state["expected_targets"] for state in states)
    else:
        assert all(not state["targets"] and state["rank"] is None and not state["ranks"] for state in states)


def observe_prefill(runner, monkeypatch):
    """Count actual prompt chunks, using schedule outputs before async progress."""
    scheduler = runner.model.llm_engine.engine_core.engine_core.scheduler
    prompt_lengths = {}
    chunks = defaultdict(list)
    original_schedule = scheduler.schedule

    def schedule(*args, **kwargs):
        output = original_schedule(*args, **kwargs)
        starts = {}
        for request in output.scheduled_new_reqs:
            prompt_lengths[request.req_id] = len(request.prompt_token_ids)
            starts[request.req_id] = request.num_computed_tokens
        cached = output.scheduled_cached_reqs
        starts.update(zip(cached.req_ids, cached.num_computed_tokens))
        for request_id, count in output.num_scheduled_tokens.items():
            start = starts[request_id]
            if start < prompt_lengths[request_id]:
                chunks[request_id].append((start, count))
        return output

    monkeypatch.setattr(scheduler, "schedule", schedule)
    return chunks


def token_prompt(tokenizer, text, length):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    assert tokens
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching,mtp",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_kvpp_combined_features(monkeypatch):
    """Compare KVPP off/on with chunk, prefix, TP, EP, async and MTP."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    results = []
    for enabled in (False, True):
        with VllmRunner(
            maybe_model_redirect(MODEL),
            dtype="auto",
            quantization="ascend",
            tensor_parallel_size=TP_SIZE,
            enable_expert_parallel=True,
            enforce_eager=True,
            async_scheduling=True,
            distributed_executor_backend="mp",
            max_model_len=4 * BLOCK_SIZE,
            max_num_seqs=4,
            max_num_batched_tokens=TOKEN_BUDGET,
            block_size=BLOCK_SIZE,
            num_gpu_blocks_override=NUM_BLOCKS,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.8,
            seed=42,
            disable_log_stats=False,
            speculative_config={"method": "mtp", "num_speculative_tokens": 1, "enforce_eager": True},
            additional_config={"enable_kvpp": enabled},
        ) as runner:
            assert_worker_state(runner, enabled)
            tokenizer = runner.model.get_tokenizer()
            prefix = token_prompt(tokenizer, "Explain how computers store historical information. ", PREFIX_LENGTH)
            prompts = [
                prefix + token_prompt(tokenizer, suffix, SUFFIX_LENGTH)
                for suffix in ("First answer: ", "Second answer: ")
            ]
            assert prompts[0] != prompts[1]
            outputs = []
            with monkeypatch.context() as observation:
                chunks = observe_prefill(runner, observation)
                for prompt in prompts:
                    (output,) = runner.model.generate(
                        [{"prompt_token_ids": prompt}],
                        SamplingParams(temperature=0, ignore_eos=True, max_tokens=MAX_TOKENS),
                        use_tqdm=False,
                    )
                    assert output.finished
                    assert len(output.outputs) == 1
                    assert len(output.outputs[0].token_ids) == MAX_TOKENS
                    assert output.outputs[0].finish_reason == "length"
                    outputs.append(output)
            assert outputs[0].num_cached_tokens == 0
            # MTP excludes the last matching block to protect prefill lookahead.
            assert outputs[1].num_cached_tokens == PREFIX_LENGTH - BLOCK_SIZE
            assert any(
                len(steps) >= 2 and steps[0][0] == 0 and any(start > 0 for start, _ in steps)
                for steps in chunks.values()
            ), chunks
            drafts = [
                metric.value for metric in runner.model.get_metrics() if metric.name == "vllm:spec_decode_num_drafts"
            ]
            assert drafts and sum(drafts) > 0
            results.append(outputs)
    assert [output.prompt_token_ids for output in results[0]] == [output.prompt_token_ids for output in results[1]]
    check_outputs_equal(
        outputs_0_lst=[(list(output.outputs[0].token_ids), output.outputs[0].text) for output in results[0]],
        outputs_1_lst=[(list(output.outputs[0].token_ids), output.outputs[0].text) for output in results[1]],
        name_0="KVPP off",
        name_1="KVPP on",
    )
