#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#

import huggingface_hub
from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download as modelscope_snapshot_download  # type: ignore[import-untyped]
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import CompilationConfig

from tests.e2e.conftest import (
    HfRunner,
    VllmRunner,
    cleanup_dist_env_and_memory,
    qwen_prompt,
    wait_until_npu_memory_free,
)
from tests.e2e.singlecard.utils import PROMPTS_SHORT, compare_logprobs
from tests.e2e.utils import check_embeddings_close


@wait_until_npu_memory_free()
def test_qwen3_0_6b_dense_piecewise_graph():
    runner_kwargs = {
        "model_name": "Qwen/Qwen3-0.6B",
        "max_model_len": 1024,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)


@wait_until_npu_memory_free()
def test_qwen3_embedding_full_decode_only():
    queries = ["What is the capital of China?", "Explain gravity"]
    model = "Qwen/Qwen3-Embedding-0.6B"
    model_name = modelscope_snapshot_download(
        model,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )
    with VllmRunner(
        model_name,
        runner="pooling",
        max_model_len=None,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4]},
    ) as vllm_runner:
        vllm_outputs = vllm_runner.embed(queries)

    with HfRunner(
        model_name,
        dtype="float32",
        is_sentence_transformer=True,
    ) as hf_runner:
        hf_outputs = hf_runner.encode(queries)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )


@wait_until_npu_memory_free()
def test_multimodal_vl_qwen3_5():
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

    img_questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]

    images = [image] * len(img_questions)
    prompts = qwen_prompt(img_questions)

    model_path = hf_snapshot_download(
        "Qwen/Qwen3.5-0.8B",
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )
    with VllmRunner(
        model_path,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_batched_tokens=1024,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4],
        },
        speculative_config={
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": 2,
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

        assert len(outputs) == len(prompts)

        for _, output_str in outputs:
            assert output_str, "Generated output should not be empty."


@wait_until_npu_memory_free()
def test_qwen3_w8a8_full_graph_eagle3():
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(
        max_tokens=300,
        temperature=0.0,
        ignore_eos=False,
    )

    with VllmRunner(
        "vllm-ascend/Qwen3-8B-W8A8",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        disable_log_stats=False,
        max_model_len=4096,
        seed=1024,
        async_scheduling=False,
        quantization="ascend",
        speculative_config={
            "disable_padded_drafter_batch": False,
            "method": "eagle3",
            "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
            "num_speculative_tokens": 2,
            "draft_tensor_parallel_size": 1,
            "max_model_len": 128,
        },
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1, 2, 4, 8]),
    ) as llm:
        spec_outputs = llm.generate(example_prompts, sampling_params)
        cleanup_dist_env_and_memory()
        del llm

    with VllmRunner(
        "vllm-ascend/Qwen3-8B-W8A8",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        disable_log_stats=False,
        max_model_len=4096,
        seed=1024,
        async_scheduling=False,
        quantization="ascend",
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1, 2, 4, 8]),
    ) as llm:
        ref_outputs = llm.generate(example_prompts, sampling_params)
        cleanup_dist_env_and_memory()
        del llm

    matches = 0
    threshold = 0.66
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0][0]
        spec_token_ids = spec_output[0][0]
        if ref_token_ids == spec_token_ids[: len(ref_token_ids)]:
            matches += 1
        else:
            print(f"ref_output: {ref_output[1][0]}")
            print(f"spec_output: {spec_output[1][0]}")

    assert matches > int(threshold * len(ref_outputs))
    cleanup_dist_env_and_memory()
