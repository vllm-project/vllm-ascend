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

import contextlib
import multiprocessing
import os
import queue
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import cleanup_dist_env_and_memory, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_LONG, PROMPTS_SHORT

QWEN3 = "Qwen/Qwen3-0.6B"
DEEPSEEK_V2_LITE = "vllm-ascend/DeepSeek-V2-Lite-W8A8"

QWEN3_PROMPTS_SHORT_BASELINE = [
    {
        "token_ids": [444, 2210, 13, 358, 2776, 264, 220, 17, 17, 4666, 6284, 5458, 504, 5616, 13, 358],
        "logprobs": [
            -4.082355976104736,
            -1.8239692449569702,
            -0.7117747664451599,
            -0.19997872412204742,
            -0.9261042475700378,
            -1.156473994255066,
            -1.6240551471710205,
            -0.4462895393371582,
            -1.7280645370483398,
            -0.03872159123420715,
            -0.006691192742437124,
            -1.729214072227478,
            -1.4082962274551392,
            -1.2914694547653198,
            -0.30865585803985596,
            -0.19352123141288757,
        ],
    },
    {
        "token_ids": [279, 1852, 438, 279, 4767, 315, 279, 3639, 19140, 13, 1096, 374, 1576, 279, 4767, 315],
        "logprobs": [
            -1.329056978225708,
            -1.9254618883132935,
            -0.16430772840976715,
            -0.07656584680080414,
            -0.20014727115631104,
            -0.009215084835886955,
            -0.6130291223526001,
            -0.7763887047767639,
            -0.7629368901252747,
            -0.7878186702728271,
            -1.5039337873458862,
            -0.34936416149139404,
            -1.1888322830200195,
            -0.7042690515518188,
            -0.7195678949356079,
            -0.5817108154296875,
        ],
    },
    {
        "token_ids": [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13],
        "logprobs": [
            -0.4164676070213318,
            -0.650071918964386,
            -1.2497835159301758,
            -0.41373029351234436,
            -0.10615493357181549,
            -1.7584394216537476,
            -0.014708652161061764,
            -0.09583187848329544,
            -0.05152897536754608,
            -0.5075417757034302,
            -0.04731765389442444,
            -0.004497650545090437,
            -0.734763503074646,
            -0.006638852413743734,
            -0.03197822347283363,
            -0.02921150252223015,
        ],
    },
    {
        "token_ids": [537, 1101, 264, 29016, 8645, 714, 264, 27155, 17991, 315, 1246, 582, 3887, 11, 975, 11],
        "logprobs": [
            -2.253798246383667,
            -0.6618977785110474,
            -1.1329351663589478,
            -1.5547151565551758,
            -2.502274751663208,
            -0.6954820156097412,
            -0.16636334359645844,
            -0.9172338247299194,
            -0.8822890520095825,
            -0.6346368789672852,
            -1.1293692588806152,
            -0.355191707611084,
            -0.23319871723651886,
            -0.3448694050312042,
            -0.015147067606449127,
            -0.023382289335131645,
        ],
    },
]

QWEN3_PROMPTS_LONG_BASELINE = [
    {
        "token_ids": [4710, 1249, 11625, 419, 3491, 11, 582, 1184, 311, 990, 279, 7506, 315, 328, 1543, 323],
        "logprobs": [
            -1.1807091236114502,
            -0.7548902034759521,
            -0.06596270948648453,
            -0.21140751242637634,
            -0.17375628650188446,
            -0.012642711400985718,
            -0.8002182841300964,
            -0.5415360331535339,
            -0.005715575534850359,
            -1.0228453874588013,
            -0.49259668588638306,
            -0.5897566080093384,
            -0.0001705739414319396,
            -0.008742272853851318,
            -0.0008293526479974389,
            -0.20622815191745758,
        ],
    },
    {
        "token_ids": [4710, 1249, 11625, 419, 3491, 11, 582, 646, 990, 279, 2701, 5486, 25, 6771, 400, 47],
        "logprobs": [
            -0.6663188934326172,
            -0.31078171730041504,
            -0.1298152357339859,
            -0.17452840507030487,
            -0.14723201096057892,
            -0.011572140268981457,
            -0.5467657446861267,
            -1.1380974054336548,
            -0.8147172927856445,
            -0.2714247703552246,
            -1.2143880128860474,
            -0.84015291929245,
            -0.7829729318618774,
            -1.5653221607208252,
            -0.631846010684967,
            -1.7602534294128418,
        ],
    },
    {
        "token_ids": [4710, 1249, 11625, 419, 3491, 11, 582, 646, 990, 279, 2701, 5486, 25, 6771, 400, 1124],
        "logprobs": [
            -0.979120671749115,
            -0.74333256483078,
            -0.22203724086284637,
            -0.31942218542099,
            -0.2570137679576874,
            -0.012989195063710213,
            -0.939915657043457,
            -1.038073182106018,
            -0.4173763394355774,
            -0.17641115188598633,
            -1.119675874710083,
            -1.2035267353057861,
            -0.7646281719207764,
            -1.5680660009384155,
            -0.5041356086730957,
            -1.0269478559494019,
        ],
    },
]

DEEPSEEK_V2_LITE_PROMPTS_SHORT_BASELINE = [
    {
        "token_ids": [185, 40, 608, 245, 207, 17, 15, 1012, 1712, 12608, 11, 285, 304, 463, 803, 14079],
        "logprobs": [
            -4.621323108673096,
            -3.077335834503174,
            -1.3346226215362549,
            -0.9667544364929199,
            -2.628643274307251,
            -1.0353751182556152,
            -1.9108309745788574,
            -0.9746326208114624,
            -0.07258293032646179,
            -2.375293016433716,
            -1.7306381464004517,
            -1.7152436971664429,
            -0.5231161713600159,
            -1.2002876996994019,
            -0.945522665977478,
            -2.263141393661499,
        ],
    },
    {
        "token_ids": [245, 668, 779, 317, 441, 889, 245, 69524, 11, 548, 245, 42357, 11, 245, 39925, 11],
        "logprobs": [
            -1.9601335525512695,
            -2.9202699661254883,
            -0.935673713684082,
            -1.9938912391662598,
            -2.5300979614257812,
            -2.1406760215759277,
            -2.01202130317688,
            -2.9169065952301025,
            -0.5751016736030579,
            -0.5648833513259888,
            -1.2851604223251343,
            -2.817136764526367,
            -0.826416015625,
            -0.8048403263092041,
            -1.2173839807510376,
            -0.32657456398010254,
        ],
    },
    {
        "token_ids": [8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185, 549, 13829, 13451, 279],
        "logprobs": [
            -0.39279282093048096,
            -0.8086707592010498,
            -0.7125738859176636,
            -1.6490240097045898,
            -1.9644602537155151,
            -0.4904576539993286,
            -0.0832064300775528,
            -0.00726190535351634,
            -0.38277116417884827,
            -0.2940319776535034,
            -0.7399694323539734,
            -0.07564151287078857,
            -0.7905924320220947,
            -1.8350766897201538,
            -0.4076140522956848,
            -0.20184002816677094,
        ],
    },
    {
        "token_ids": [6464, 11, 285, 359, 487, 82, 889, 1872, 276, 752, 34993, 13, 1733, 20838, 11106, 276],
        "logprobs": [
            -2.4984869956970215,
            -0.7851194739341736,
            -0.6003906726837158,
            -1.4917504787445068,
            -0.9962607026100159,
            -0.004179196432232857,
            -1.5068310499191284,
            -0.4656505584716797,
            -0.0027805021964013577,
            -0.11925199627876282,
            -0.35579147934913635,
            -0.35492807626724243,
            -1.605405330657959,
            -1.38535737991333,
            -0.930091142654419,
            -0.004122450482100248,
        ],
    },
]

DEEPSEEK_V2_LITE_PROMPTS_LONG_BASELINE = [
    {
        "token_ids": [185, 185, 1679, 26430, 279, 16145, 285, 8204, 185, 185, 13483, 9890, 16982, 457, 17693, 829],
        "logprobs": [
            -0.24134111404418945,
            -0.06949862092733383,
            -2.196256637573242,
            -0.3235849142074585,
            -5.745722592109814e-05,
            -7.152554530875932e-07,
            -9.381330892210826e-05,
            -3.4570632578834193e-06,
            -0.0006878394051454961,
            -6.079655122448457e-06,
            -7.152554530875932e-07,
            -8.356221951544285e-05,
            -1.8358061424805783e-05,
            -1.2636104656849056e-05,
            -1.5497195136049413e-06,
            -2.0146166207268834e-05,
        ],
    },
    {
        "token_ids": [185, 185, 1679, 26430, 279, 16145, 285, 8204, 185, 185, 13483, 9890, 16982, 457, 17693, 829],
        "logprobs": [
            -0.27419501543045044,
            -0.08238636702299118,
            -2.0612075328826904,
            -1.0054221153259277,
            -4.947062916471623e-05,
            -4.768370445162873e-07,
            -2.1457441107486375e-05,
            -3.933898824470816e-06,
            -0.00018630675913300365,
            -1.1920920996999484e-06,
            -4.172316494077677e-06,
            -0.00010048838157672435,
            -1.0847986231965479e-05,
            -1.1920928244535389e-07,
            -1.1920920996999484e-06,
            -1.4185804502631072e-05,
        ],
    },
    {
        "token_ids": [185, 185, 1679, 26430, 279, 16145, 285, 8204, 185, 185, 13483, 9890, 16982, 457, 17693, 829],
        "logprobs": [
            -0.14973750710487366,
            -0.09690935909748077,
            -1.8772021532058716,
            -1.6962311267852783,
            -0.0001429217227268964,
            -8.344646289515367e-07,
            -7.617183291586116e-05,
            -3.814689989667386e-06,
            -0.0010556369088590145,
            -6.318072337307967e-06,
            -1.1920920996999484e-06,
            -0.00035553809721022844,
            -2.729855441430118e-05,
            -2.3841855067985307e-07,
            -1.4305104514278355e-06,
            -4.410733708937187e-06,
        ],
    },
]

CASE_QWEN_ACLGRAPH = {
    "model": QWEN3,
    "quantization": None,
    "prompts": {"short": PROMPTS_SHORT, "long": PROMPTS_LONG},
    "compilation_config": {"max_cudagraph_capture_size": 24, "cudagraph_mode": "FULL"},
    "tensor_parallel_size": 4,
    "data_parallel_size": 1,
    "enable_expert_parallel": False,
    "golden_answers": {"short": QWEN3_PROMPTS_SHORT_BASELINE, "long": QWEN3_PROMPTS_LONG_BASELINE},
    "baseline_capture_mem": 0.20,
    "capture_mem_tolerance": 1.3,
}

CASE_DS_ACLGRAPH = {
    "model": DEEPSEEK_V2_LITE,
    "quantization": "ascend",
    "prompts": {"short": PROMPTS_SHORT, "long": PROMPTS_LONG},
    "compilation_config": {"max_cudagraph_capture_size": 24, "cudagraph_mode": "FULL_AND_PIECEWISE"},
    "tensor_parallel_size": 2,
    "data_parallel_size": 2,
    "enable_expert_parallel": True,
    "golden_answers": {
        "short": DEEPSEEK_V2_LITE_PROMPTS_SHORT_BASELINE,
        "long": DEEPSEEK_V2_LITE_PROMPTS_LONG_BASELINE,
    },
    "baseline_capture_mem": 0.68,
    "capture_mem_tolerance": 1.5,
}

CASE_DS_ACLGRAPH_ENPU = {
    **CASE_DS_ACLGRAPH,
    "env_vars": {"ENPU_ENABLE": "true"},
}

# inherit from tests/e2e/pull_request/utils.py::compare_logprobs
ATOL = 0.0689

_SAMPLING_PARAMS = SamplingParams(
    max_tokens=16,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    logprobs=20,
)


def _install_spies(metrics: dict[str, Any] | None):
    """Installs thread-safe spies on NPU methods to track invocation counts."""
    if metrics is None:
        return contextlib.nullcontext()

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    def make_spy(cls, method_name, capture_mem):
        original = getattr(cls, method_name)

        def spy(self, *args, **kwargs):
            mem_before = torch.npu.mem_get_info()[0]  # free memory
            result = original(self, *args, **kwargs)
            mem_after = torch.npu.mem_get_info()[0]
            with capture_mem["call_counts"].get_lock():
                capture_mem["call_counts"].value += 1
                capture_mem["mem_before_capture"].value += mem_before
                capture_mem["mem_after_capture"].value += mem_after
            return result

        return spy

    stack = contextlib.ExitStack()
    hooks = [
        (NPUModelRunner, "capture_model", metrics["capture_mem"]),
    ]

    for cls, method, metric in hooks:
        stack.enter_context(patch.object(cls, method, make_spy(cls, method, metric)))

    return stack


def _check_prefill_token(baseline, comp_ids, comp_logprobs, prompt_idx: int, atol: float) -> None:
    """Token 0 is produced by the prefill pass; both models see identical input,
    so the chosen token *must* be the same and its logprob must match within atol."""
    base_token_id = baseline["token_ids"][0]
    comp_token_id = comp_ids[0]
    assert base_token_id == comp_token_id, (
        f"Prefill token mismatch at prompt {prompt_idx}: baseline={base_token_id}, compiled={comp_token_id}"
    )
    base_logprob = baseline["logprobs"][0]
    comp_logprob = comp_logprobs[0][comp_token_id]
    assert abs(base_logprob - comp_logprob) <= atol, (
        f"Prefill logprob mismatch at prompt {prompt_idx}: "
        f"baseline={base_logprob:.4f}, compiled={comp_logprob:.4f}, "
        f"diff={abs(base_logprob - comp_logprob):.4f} > atol={atol}"
    )


def _check_decode_token(baseline, comp_ids, comp_logprobs, token_idx: int, prompt_idx: int, decode_atol: float) -> None:
    """Tokens 1-2 come from decode passes.  When the two models pick different
    tokens the context has already diverged, so we cannot compare logprobs of
    the chosen tokens directly.  Instead we do a cross-lookup: find the
    baseline's chosen token inside compiled's top-K distribution (and vice
    versa) and assert that the assigned log-probability is close.  This
    confirms that the compiled model's distribution is numerically consistent
    with the baseline's even when the argmax differs by a tiny margin.
    """
    base_token_id = baseline["token_ids"][token_idx]
    comp_token_id = comp_ids[token_idx]
    base_logprob = baseline["logprobs"][token_idx]
    comp_topk = comp_logprobs[token_idx]
    if base_token_id == comp_token_id:
        # Happy path: same token, direct logprob comparison.
        diff = abs(base_logprob - comp_topk[comp_token_id])
        assert diff <= decode_atol, (
            f"Decode logprob mismatch at prompt {prompt_idx}, token {token_idx}: "
            f"baseline={base_logprob:.4f}, "
            f"compiled={comp_topk[comp_token_id]:.4f}, "
            f"diff={diff:.4f} > decode_atol={decode_atol}"
        )
        return

    # Tokens differ – cross-lookup in each model's top-K distribution.
    comp_logprob = comp_topk[comp_token_id]
    # Check: what log-probability did compiled assign to baseline's token?
    assert base_token_id in comp_topk, (
        f"Decode token mismatch at prompt {prompt_idx}, token {token_idx}: "
        f"baseline chose token {base_token_id} (logprob={base_logprob:.4f}) but "
        f"compiled chose token {comp_token_id} (logprob={comp_logprob:.4f}) and "
        f"baseline's token does not appear in compiled's top-{_SAMPLING_PARAMS.logprobs} distribution"
    )
    comp_logprob_of_base_token = comp_topk[base_token_id]
    diff = abs(base_logprob - comp_logprob_of_base_token)
    assert diff <= decode_atol, (
        f"Decode distribution mismatch at prompt {prompt_idx}, token {token_idx}: "
        f"baseline chose token {base_token_id} with logprob={base_logprob:.4f}; "
        f"compiled assigned logprob={comp_logprob_of_base_token:.4f} to that token, "
        f"diff={diff:.4f} > decode_atol={decode_atol} "
        f"(compiled chose token {comp_token_id} with logprob={comp_logprob:.4f})"
    )


def _run_worker_process(
    rank: int,
    local_rank: int,
    world_size: int,
    cur_case: dict,
    master_ip: str,
    master_port: int,
    result_queue: multiprocessing.Queue,
    metrics: dict[str, Any] | None = None,
):
    """Main entry point for the worker process."""
    os.environ.update(
        {
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(local_rank),
            "VLLM_DP_SIZE": str(world_size),
            "VLLM_DP_MASTER_IP": master_ip,
            "VLLM_DP_MASTER_PORT": str(master_port),
        }
    )

    for key, value in cur_case.get("env_vars", {}).items():
        os.environ[key] = str(value)

    # Apply hooks and run inference
    with _install_spies(metrics):
        short_prompts = cur_case["prompts"]["short"]
        chunk_size = len(short_prompts) // world_size
        short_start_idx = rank * chunk_size
        short_end_idx = short_start_idx + chunk_size if rank < world_size - 1 else len(short_prompts)
        local_short_prompts = short_prompts[short_start_idx:short_end_idx]

        long_prompts = cur_case["prompts"]["long"]
        chunk_size = len(long_prompts) // world_size
        long_start_idx = rank * chunk_size
        long_end_idx = long_start_idx + chunk_size if rank < world_size - 1 else len(long_prompts)
        local_long_prompts = long_prompts[long_start_idx:long_end_idx]

        llm = LLM(
            model=cur_case["model"],
            max_model_len=1024,
            compilation_config=cur_case["compilation_config"],
            quantization=cur_case["quantization"],
            tensor_parallel_size=cur_case["tensor_parallel_size"],
            enable_expert_parallel=cur_case["enable_expert_parallel"],
            trust_remote_code=True,
        )

        compiled_outputs_short = llm.generate(local_short_prompts, _SAMPLING_PARAMS)
        compiled_outputs_long = llm.generate(local_long_prompts, _SAMPLING_PARAMS)

        def extract_outputs(outputs):
            extracted = []
            for out in outputs:
                gen = out.outputs[0]
                extracted.append(
                    {
                        "text": gen.text,
                        "token_ids": list(gen.token_ids),
                        "logprobs": [
                            {token_id: lp.logprob for token_id, lp in step_logprobs.items()}
                            for step_logprobs in gen.logprobs
                        ]
                        if gen.logprobs
                        else None,
                    }
                )
            return extracted

        result_data = {
            "rank": rank,
            "short": {"prompt_idx": short_start_idx, "outputs": extract_outputs(compiled_outputs_short)},
            "long": {"prompt_idx": long_start_idx, "outputs": extract_outputs(compiled_outputs_long)},
        }
        result_queue.put(result_data)


def _exit():
    from vllm_ascend.ascend_config import clear_ascend_config

    clear_ascend_config()
    cleanup_dist_env_and_memory()


def check_accuracy(baselines, result, atol, decode_atol):
    for idx, comp_out in enumerate(result["outputs"]):
        prompt_idx = result["prompt_idx"] + idx
        baseline = baselines[prompt_idx]
        comp_ids = comp_out["token_ids"]
        comp_logprobs = comp_out["logprobs"]

        assert comp_logprobs is not None, f"logprobs not returned for prompt {prompt_idx}"
        assert len(baseline["token_ids"]) == len(comp_ids) == _SAMPLING_PARAMS.max_tokens, (
            f"Expected {_SAMPLING_PARAMS.max_tokens} tokens for prompt {prompt_idx}, "
            f"got baseline={len(baseline['token_ids'])}, compiled={len(comp_ids)}"
        )
        _check_prefill_token(baseline, comp_ids, comp_logprobs, prompt_idx, atol)
        for token_idx in range(1, _SAMPLING_PARAMS.max_tokens):
            _check_decode_token(baseline, comp_ids, comp_logprobs, token_idx, prompt_idx, decode_atol)


def check_capture_mem(capture_mem, baseline_capture_mem=0.2, capture_mem_tolerance=1.3):
    assert capture_mem["call_counts"].value != 0, (
        f"capture_model was not called during test. capture_called({capture_mem['call_counts'].value})"
    )

    print("capture_called =", capture_mem["call_counts"].value)
    print("capture_mem_before =", capture_mem["mem_before_capture"].value)
    print("capture_mem_after =", capture_mem["mem_after_capture"].value)

    mem_used_by_capture = (
        capture_mem["mem_before_capture"].value - capture_mem["mem_after_capture"].value
    ) / capture_mem["call_counts"].value
    # Empirical observation: capturing ACL graphs for Qwen3-0.6B uses ~0.20 GiB of NPU memory.
    # DeepSeek-V2-Lite-W8A8 uses ~0.68 GiB of NPU memory
    # a 1.3x tolerance is applied to account for runtime variance.
    max_capture_mem_gib = baseline_capture_mem * capture_mem_tolerance
    max_mem_expected = max_capture_mem_gib * (1024**3)
    assert mem_used_by_capture < max_mem_expected, (
        f"capture_model used more memory than expected. "
        f"Used: {mem_used_by_capture / (1024**3):.2f} GiB, "
        f"Expected: < {max_capture_mem_gib:.2f} GiB"
    )


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("cur_case", [CASE_QWEN_ACLGRAPH, CASE_DS_ACLGRAPH, CASE_DS_ACLGRAPH_ENPU])
def test_aclgraph(cur_case: dict, monkeypatch: pytest.MonkeyPatch):
    # Counter doesn't work in default "spawn" mode
    metrics = None
    if "DeepSeek-V2-Lite-W8A8" in cur_case["model"]:
        # TODO(shihan-lin168): remove this env after set_device issue is resolved
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    else:
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
        metrics = {
            "capture_mem": {
                # Monitor key metrics of mem
                "call_counts": multiprocessing.Value("i", 0),
                "mem_before_capture": multiprocessing.Value("q", 0),
                "mem_after_capture": multiprocessing.Value("q", 0),
            },
        }

    port = get_open_port()

    # Create a queue to collect results from both processes
    result_queue: multiprocessing.Queue[dict] = multiprocessing.Queue()

    # Launch workers
    workers = []
    for rank in range(cur_case["data_parallel_size"]):
        p = multiprocessing.Process(
            target=_run_worker_process,
            args=(rank, rank, cur_case["data_parallel_size"], cur_case, "127.0.0.1", port, result_queue, metrics),
        )
        p.start()
        workers.append(p)

    all_dp_results = []

    # get results
    for _ in range(cur_case["data_parallel_size"]):
        try:
            result = result_queue.get(timeout=180)
            all_dp_results.append(result)
        except queue.Empty:
            print("Error: Timeout waiting for worker results. A worker might have crashed.")
            break

    # Supervision loop
    for p in workers:
        p.join(timeout=30)
        if p.exitcode != 0:
            for k in workers:
                if k.is_alive():
                    k.kill()
                    p.join(timeout=5)
            raise RuntimeError(f"Worker {p.pid} failed with exit code {p.exitcode}")
    _exit()
    assert len(all_dp_results) == cur_case["data_parallel_size"], f"Expected 2 results, got {len(all_dp_results)}"

    # check graph memory
    if metrics is not None:
        check_capture_mem(metrics["capture_mem"], cur_case["baseline_capture_mem"], cur_case["capture_mem_tolerance"])

    # check accuracy
    decode_atol = 2 * ATOL
    for result in all_dp_results:
        check_accuracy(cur_case["golden_answers"]["short"], result["short"], ATOL, decode_atol)
        check_accuracy(cur_case["golden_answers"]["long"], result["long"], ATOL, decode_atol)
