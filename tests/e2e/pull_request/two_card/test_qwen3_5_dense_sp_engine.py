# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from vllm import LLM, SamplingParams

from tests.e2e.pull_request.two_card.qwen3_5_dense_sp_probe import (
    inspect_dense_sp_model,
    instrument_dense_sp_collectives,
)


def test_qwen3_5_dense_sp_dummy_engine() -> None:
    model_path = os.getenv("SP_TEST_MODEL")
    if model_path is None:
        pytest.skip("SP_TEST_MODEL must point to a local Qwen3.8 checkpoint")
    enforce_eager = os.getenv("SP_TEST_ENFORCE_EAGER", "1") != "0"
    enable_mtp = os.getenv("SP_TEST_ENABLE_MTP", "0") == "1"
    enable_prefix_caching = os.getenv("SP_TEST_ENABLE_PREFIX_CACHING", "0") == "1"
    layer_type = os.getenv("SP_TEST_LAYER_TYPE", "full_attention")
    assert layer_type in ("full_attention", "linear_attention")
    compilation_config = None
    if not enforce_eager:
        compilation_config = {
            "cudagraph_capture_sizes": [1, 4],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }

    # Preserve the real Qwen3.8 hidden/MLP/attention dimensions, but use one
    # decoder layer and a small vocabulary so this integration test can run in
    # little HBM. Dummy weights are sufficient to exercise the complete model
    # runner and dense-SP communication path.
    # Point the model loader at a config-only directory so it does not
    # auto-detect the checkpoint's W8A8 quantization description. The real
    # checkpoint directory remains the tokenizer/config source of truth, while
    # dummy BF16 weights make the one-layer engine small enough for CI.
    with tempfile.TemporaryDirectory(prefix="qwen3_5_dense_sp_") as model_dir:
        source_dir = Path(model_path)
        compile_cache_dir = Path(model_dir) / "compile_cache"
        if compilation_config is not None:
            compilation_config["cache_dir"] = str(compile_cache_dir)
        for config_name in (
            "chat_template.jinja",
            "config.json",
            "configuration.json",
            "generation_config.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ):
            source = source_dir / config_name
            if source.exists():
                shutil.copy2(source, model_dir)
        llm = LLM(
            model=model_dir,
            tokenizer=model_path,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            trust_remote_code=True,
            load_format="dummy",
            dtype="bfloat16",
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=1,
            gpu_memory_utilization=0.02,
            language_model_only=True,
            enable_prefix_caching=enable_prefix_caching,
            speculative_config=(
                {
                    "method": "qwen3_5_mtp",
                    "num_speculative_tokens": 1,
                }
                if enable_mtp
                else None
            ),
            additional_config={"enable_flashcomm1": True},
            hf_overrides={
                "text_config": {
                    "num_hidden_layers": 1,
                    "layer_types": [layer_type],
                    "vocab_size": 1024,
                }
            },
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
            assert not state["collectives"]

        outputs = llm.generate(
            [{"prompt_token_ids": [1, 2, 3, 4]}],
            SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
        )
        final_states = llm.apply_model(inspect_dense_sp_model)
        if not enforce_eager:
            graph_files = sorted(compile_cache_dir.glob("rank_*_*/backbone/computation_graph.py"))
            assert len(graph_files) == 2
            for graph_file in graph_files:
                graph = graph_file.read_text(encoding="utf-8")
                assert graph.count("torch.ops.vllm.all_gather") >= 2
                assert graph.count("torch.ops.vllm.ascend_sp_reduce_scatter_impl") >= 2
    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 2
    for state in final_states:
        names = [event[0] for event in state["collectives"]]
        if enforce_eager:
            assert names
            assert names.count("all_gather") == names.count("reduce_scatter")
            assert names.count("all_gather") >= (4 if enable_mtp else 2)
    print(
        {
            "enforce_eager": enforce_eager,
            "enable_mtp": enable_mtp,
            "enable_prefix_caching": enable_prefix_caching,
            "layer_type": layer_type,
            "generated_token_ids": outputs[0].outputs[0].token_ids,
            "initial_states": initial_states,
            "final_states": final_states,
        }
    )


if __name__ == "__main__":
    test_qwen3_5_dense_sp_dummy_engine()
