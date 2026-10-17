# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
from vllm import LLM, SamplingParams

from tests.e2e.pull_request.two_card.qwen3_5_dense_sp_probe import (
    inspect_dense_sp_model,
    instrument_dense_sp_collectives,
)


def test_qwen3_5_dense_sp_full_weights() -> None:
    model_path = os.getenv("SP_TEST_MODEL")
    if model_path is None:
        pytest.skip("SP_TEST_MODEL must point to a local Qwen3.8 checkpoint")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        trust_remote_code=True,
        quantization="ascend",
        dtype="bfloat16",
        kv_cache_dtype="mxfp8",
        enforce_eager=True,
        max_model_len=16,
        max_num_batched_tokens=16,
        max_num_seqs=1,
        gpu_memory_utilization=0.02,
        kv_cache_memory_bytes=512 * 1024 * 1024,
        language_model_only=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        additional_config={"enable_flashcomm1": True},
    )

    initial_states = llm.apply_model(instrument_dense_sp_collectives)
    assert len(initial_states) == 2
    for state in initial_states:
        assert state["layer_class"] == "AscendQwen3_5DecoderLayer"
        assert state["dense_sp"]
        assert state["model_sp_contract"]
        assert not state["attention_reduce_results"]
        assert not state["mlp_reduce_results"]
        assert state["hidden_size"] == 5120
        assert state["intermediate_size"] == 17408
        assert state["num_layers"] == 64
        assert not state["collectives"]

    outputs = llm.generate(
        [{"prompt_token_ids": [1, 2, 3, 4]}],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
    )
    final_states = llm.apply_model(inspect_dense_sp_model)

    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 1
    collective_summaries = []
    for state in final_states:
        names = [event[0] for event in state["collectives"]]
        all_gathers = names.count("all_gather")
        reduce_scatters = names.count("reduce_scatter")
        assert all_gathers == reduce_scatters
        assert all_gathers >= 2 * state["num_layers"]
        collective_summaries.append(
            {
                "all_gathers": all_gathers,
                "reduce_scatters": reduce_scatters,
                "first_events": state["collectives"][:4],
                "last_events": state["collectives"][-4:],
            }
        )
    print(
        {
            "generated_token_ids": outputs[0].outputs[0].token_ids,
            "initial_states": initial_states,
            "collective_summaries": collective_summaries,
        }
    )


if __name__ == "__main__":
    test_qwen3_5_dense_sp_full_weights()
