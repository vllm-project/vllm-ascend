# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import gc
from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset

from tests.e2e.conftest import VllmRunner


def _samples():
    image = ImageAsset("stop_sign").get_path("jpg")
    audio = AudioAsset("mary_had_lamb").get_local_path()
    image_b64 = base64.b64encode(image.read_bytes()).decode()
    audio_b64 = base64.b64encode(audio.read_bytes()).decode()
    vllm_image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}}
    vllm_audio = {"type": "audio_url", "audio_url": {"url": "data:audio/ogg;base64," + audio_b64}}
    hf_image = {"type": "image", "url": str(image)}
    hf_audio = {"type": "audio", "url": str(audio)}
    return [
        ([], [], "What is 2 + 2? Answer with just one digit."),
        ([vllm_image], [hf_image], "What kind of traffic sign is in the image?"),
        ([vllm_audio], [hf_audio], "Transcribe the speech."),
        ([vllm_image, vllm_audio], [hf_image, hf_audio], "Describe the traffic sign and transcribe the speech."),
    ]


def _messages(media, prompt):
    return [{"role": "user", "content": media + [{"type": "text", "text": prompt}]}]


def _check_reference(actual, expected):
    expected_ids = expected.sequences[0, len(actual.prompt_token_ids) :].tolist()
    actual_ids = list(actual.outputs[0].token_ids)
    assert actual_ids and expected_ids
    # Match upstream check_logprobs_close: compare the first divergent token
    # under the same prefix, requiring mutual top-5 membership without a tie allowance.
    for index, (actual_id, expected_id) in enumerate(zip(actual_ids, expected_ids)):
        if actual_id != expected_id:
            reference_topk = expected.scores[index][0].float().topk(5).indices.tolist()
            assert actual_id in reference_topk
            assert expected_id in actual.outputs[0].logprobs[index]
            break
    else:
        assert actual_ids == expected_ids


@pytest.mark.parametrize("graph_mode", ["NONE", "FULL_DECODE_ONLY"])
def test_gemma4_multimodal_reference(config_filename, tp_size, graph_mode):
    config = yaml.safe_load(config_filename.read_text())
    model_name = config["model_name"]
    if Path(model_name).name.lower() not in ("gemma-4-e2b-it", "gemma-4-e4b-it"):
        pytest.skip("Select a Gemma4 E2B/E4B config with --config")
    samples = _samples()
    compilation_config = {"cudagraph_mode": graph_mode}
    if graph_mode != "NONE":
        compilation_config["cudagraph_capture_sizes"] = [1, 2, 4]
    params = SamplingParams(temperature=0, max_tokens=160, logprobs=5)
    with VllmRunner(
        model_name,
        tensor_parallel_size=int(tp_size),
        distributed_executor_backend="mp",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=2 * 1024**3,
        limit_mm_per_prompt={"image": 1, "audio": 1},
        enforce_eager=graph_mode == "NONE",
        compilation_config=compilation_config,
        seed=42,
    ) as runner:
        actual = [
            runner.model.chat(
                _messages(media, prompt), params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False}
            )[0]
            for media, _, prompt in samples
        ]
        batched = runner.model.chat(
            [_messages(media, prompt) for media, _, prompt in samples],
            params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        assert actual[0].outputs[0].text.strip() == "4"
        assert batched[0].outputs[0].text.strip() == "4"
        for index in (1, 3):
            assert "stop" in actual[index].outputs[0].text.lower()
            assert "stop" in batched[index].outputs[0].text.lower()
        assert all(item.outputs[0].token_ids for item in batched)

    processor = AutoProcessor.from_pretrained(model_name)
    reference = (
        AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .eval()
        .to("npu")
    )
    # Audio uses hand-written blocked attention with a boolean visibility mask,
    # not the additive floating-point mask produced by the eager mask builder.
    reference.model.audio_tower.config._attn_implementation = "sdpa"
    try:
        for index, (_, media, prompt) in enumerate(samples):
            inputs = processor.apply_chat_template(
                _messages(media, prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to("npu")
            assert inputs["input_ids"][0].tolist() == actual[index].prompt_token_ids
            with torch.inference_mode():
                expected = reference.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            _check_reference(actual[index], expected)
            _check_reference(batched[index], expected)
    finally:
        del reference
        gc.collect()
        torch.npu.empty_cache()
