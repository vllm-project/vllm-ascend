# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Offline inference correctness tests for the Korean hate-speech classifier
`jason9693/Qwen2.5-1.5B-apeach` (HF) / `Howeee/Qwen2.5-1.5B-apeach`
(ModelScope), a `Qwen2ForSequenceClassification` sequence-classification
model with 2 labels (id2label: 0=Default, 1=Spoiled).

Scoping notes (see issue #10680):

* This is a classification model: there is **no sampling path** (the forward
  pass is an argmax/softmax computation with no sampling node), so the usual
  greedy/non-greedy sampling criteria are undefined for it. The stability
  criterion is covered instead by re-running the same input and asserting
  identical argmax labels plus probabilities within the shared atol=1e-2
  tolerance, and by a probability comparison against the HF fp32 reference.
  (On the Ascend custom-op path the residual kernel noise is ~1e-3;
  bit-exact run-to-run equality would require VLLM_BATCH_INVARIANT=1, which
  disables that custom op path - see the test docstrings.)
* The HF reference runs in fp32 on device (same tolerance convention as the
  existing `test_classification.py` precedent), the vLLM side runs in bf16
  (fp16 is deliberately avoided: a same-lineage fp16 test was historically
  flaky and is on the CI skip list).
* Korean corpus is self-constructed (no third-party license) with >= 15
  samples per label; Chinese/English samples are smoke-only (the model is
  Korean-only, so they only assert that inference runs without crashing).
* Ascend custom ops: the `aclnnAddRmsNormBias` kernel is shipped in the
  `libcust_opapi.so` vendor package and discovered via
  `ASCEND_CUSTOM_OPP_PATH`. Because `vllm_ascend_C.so` is loaded before
  Python-side `bootstrap_custom_op_env()` runs, the environment variable
  must be set before the process starts (source
  `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash`).
  When it is not set, the fixture below transparently falls back to
  `torch_npu.npu_add_rms_norm` so the test still runs.
"""

import glob
import os

import pytest
import torch
from modelscope import snapshot_download  # type: ignore[import-untyped]
from transformers import AutoConfig, AutoModelForSequenceClassification

from tests.e2e.conftest import (
    HfRunner,
    VllmRunner,
    cleanup_dist_env_and_memory,
)

MODEL_NAME = "Howeee/Qwen2.5-1.5B-apeach"
REVISION = "master"

# Self-constructed Korean corpus (no third-party license). Label 0 = Default
# (benign), label 1 = Spoiled (harmful/hate speech). The set deliberately
# includes formal/honoric speech, colloquial speech, and borderline
# euphemistic insults. Every sample below was verified offline (CPU, HF
# fp32) to have the expected HF reference argmax.
KOREAN_BENIGN = [
    "안녕하세요 오늘 날씨가 참 좋네요",
    "이 영화 정말 재미있었어요 다음에 또 보고 싶습니다",
    "오늘 저녁에 가족과 함께 맛있는 밥을 먹었어요",
    "새로 산 신발이 발에 딱 맞아서 기분이 좋아요",
    "고생 많으셨습니다 앞으로도 잘 부탁드립니다",
    "시험 공부를 열심히 해서 좋은 결과를 받았어요",
    "주말에 친구들과 함께 등산을 가기로 했어요",
    "반려견과 함께 공원을 산책하면서 즐거운 시간을 보냈습니다",
    "커피 한 잔 마시면서 여유를 즐기고 있습니다",
    "회사 프로젝트가 무사히 마무리되어 다행입니다",
    "아이가 처음으로 걸어서 가족 모두 기뻐했어요",
    "음악을 들으면서 산책하는 걸 제일 좋아해요",
    "오랜만에 만난 친구와 수다를 떨었어요",
    "도서관에서 조용히 책을 읽는 시간이 좋아요",
    "오늘은 하루 종일 비가 와서 운치 있었어요",
    "한 해를 마무리하며 감사한 마음이 듭니다",
]

KOREAN_HARMFUL = [
    "너 진짜 최악이다 다시는 보기 싫다",
    "네가 하는 말은 전부 쓰레기 같아",
    "저 사람 얼굴만 봐도 기분이 더러워진다",
    "너 같은 사람하고는 말 섞기도 싫다",
    "죽어버려 세상에 있으면 안 되는 존재야",
    "너는 아무것도 제대로 하는 게 없어",
    "멍청한 놈 그걸 몰라서야 어떻게 사냐",
    "입 닥치고 조용히 좀 해 줄래",
    "너 때문에 다 망했다 전부 네 탓이야",
    "그런 옷을 입다니 보기 흉하다",
    "넌 그냥 잠자코 있는 게 도와주는 거야",
    "이런 말도 안 되는 소리 하지 마",
    "너 같으면 부끄러워서 살 수나 있겠냐",
    "가서 뒤져라 다시 나타나지 마",
    "네 놈 부모님이 불쌍하다",
    "너는 그냥 존재 자체가 민폐야",
    "못생긴 게 성격까지 그 모양이냐",
]

KOREAN_CORPUS = [(p, 0) for p in KOREAN_BENIGN] + [(p, 1) for p in KOREAN_HARMFUL]

# Samples whose HF fp32 confidence exceeds 0.9 (verified offline on CPU with
# AutoModelForSequenceClassification). For these the strict atol=1e-2
# probability comparison applies (same convention as the existing
# test_classification.py precedent, whose samples are likewise
# high-confidence). The model naturally emits soft probabilities (0.5-0.9)
# for benign Korean inputs, where bf16-vs-fp32 rounding alone shifts the
# probability by 0.01-0.02; those samples are still fully asserted for
# argmax agreement on both sides, but the strict probability tolerance is
# only meaningful for the high-confidence subset. No assertion is dropped.
HIGH_CONFIDENCE_PROMPTS = [
    "안녕하세요 오늘 날씨가 참 좋네요",
    "이 영화 정말 재미있었어요 다음에 또 보고 싶습니다",
    "네가 하는 말은 전부 쓰레기 같아",
    "멍청한 놈 그걸 몰라서야 어떻게 사냐",
    "못생긴 게 성격까지 그 모양이냐",
    "입 닥치고 조용히 좀 해 줄래",
    "너 같으면 부끄러워서 살 수나 있겠냐",
    "가서 뒤져라 다시 나타나지 마",
]

# Chinese/English smoke-only samples (the model is Korean-only; these only
# assert that inference runs without crashing and yields sane probabilities).
ZH_SMOKE = [
    "今天天气真好，我们一起出去玩吧。",
    "你这个笨蛋，我讨厌你。",
    "请问这附近有没有地铁站？",
]
EN_SMOKE = [
    "I love this beautiful sunny day.",
    "You are terrible and I hate you.",
    "Could you please help me with this task?",
]


def _resolve_model_path() -> str:
    """Resolve the model path.

    `APEACH_MODEL_PATH` allows pointing at an already-downloaded snapshot
    (e.g. the HF `jason9693/` mirror); otherwise the ModelScope source is
    used, matching the existing `test_classification.py` precedent.
    """
    override = os.environ.get("APEACH_MODEL_PATH")
    if override:
        return override
    import huggingface_hub

    return snapshot_download(
        MODEL_NAME,
        revision=REVISION,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )


@pytest.fixture(scope="session")
def model_path() -> str:
    return _resolve_model_path()


def _custom_op_env_ready() -> bool:
    """Whether the Ascend custom-op vendor library is discoverable.

    The custom op kernels (e.g. `aclnnAddRmsNormBias`) live in
    `libcust_opapi.so` under the vendor package and are located via
    `ASCEND_CUSTOM_OPP_PATH`. When that variable is not set (e.g.
    `set_env.bash` was not sourced before the process started),
    `vllm_ascend_C.so` static-initializes an empty custom lib path and the
    op cannot be resolved.
    """
    for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if entry and glob.glob(os.path.join(entry, "op_api", "lib", "libcust_opapi.so")):
            return True
    return False


@pytest.fixture(autouse=True)
def _fallback_when_ascend_custom_op_unavailable():
    """Fall back to `torch_npu.npu_add_rms_norm` only when the custom op
    vendor library is not discoverable.

    `vllm_ascend.ops.layernorm.AscendRMSNorm.forward_oot` prefers the
    custom op `_C_ascend.npu_add_rms_norm_bias`. When
    `ASCEND_CUSTOM_OPP_PATH` is missing (the `set_env.bash` of the
    `_cann_ops_custom` vendor package was not sourced), resolving that op
    raises `aclnnAddRmsNormBias ... not in libopapi.so` at the first
    forward pass. In that case, and only in that case, the cached flag is
    forced to `False` so the native `torch_npu.npu_add_rms_norm` path is
    selected; with a correctly configured environment the custom op path
    is used as-is.
    """
    from vllm_ascend import utils as ascend_utils

    old = ascend_utils._CUSTOM_OP_ENABLED
    if not _custom_op_env_ready():
        ascend_utils._CUSTOM_OP_ENABLED = False
    yield
    ascend_utils._CUSTOM_OP_ENABLED = old


@pytest.fixture(scope="session")
def apeach_vllm_runner(model_path: str) -> VllmRunner:
    """Shared bf16 pooling runner (the same instance is reused across tests
    so that determinism / single-vs-batch comparisons run against one
    engine)."""
    cleanup_dist_env_and_memory()
    runner = VllmRunner(
        model_path,
        runner="pooling",
        dtype="bfloat16",
        max_model_len=4096,
        enforce_eager=True,
        tensor_parallel_size=1,
        # Cap the resident HBM reservation (default 0.9 would leave no room
        # for the in-process HF fp32 reference model).
        gpu_memory_utilization=0.4,
    )
    try:
        yield runner
    finally:
        del runner.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def apeach_hf_runner(model_path: str) -> HfRunner:
    """HF fp32 reference (same tolerance convention as the existing
    `test_classification.py` precedent)."""
    cleanup_dist_env_and_memory()
    runner = HfRunner(model_path, dtype="float32", auto_cls=AutoModelForSequenceClassification)
    try:
        yield runner
    finally:
        del runner.model
        cleanup_dist_env_and_memory()


def _check_probs(probs: list[float]) -> None:
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-3


def _argmax(probs: list[float]) -> int:
    return int(torch.tensor(probs).argmax().item())


def test_id2label_polarity(model_path: str) -> None:
    """The id2label mapping in config.json is authoritative for polarity."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    assert {int(k): v for k, v in config.id2label.items()} == {0: "Default", 1: "Spoiled"}
    assert config.architectures == ["Qwen2ForSequenceClassification"]
    assert config.num_labels == 2
    assert config.problem_type == "single_label_classification"


def test_classify_matches_hf_fp32_reference(
    apeach_vllm_runner: VllmRunner,
    apeach_hf_runner: HfRunner,
) -> None:
    """Full Korean corpus (>= 15 per label):

    * vLLM bf16 and HF fp32 argmax labels agree on 100% of samples;
    * every sample's HF reference label agrees with the constructed ground
      truth (corpus sanity);
    * probabilities are well-formed (2 classes, in [0, 1], sum to 1);
    * for the high-confidence subset (HF fp32 confidence > 0.9) the strict
      atol=1e-2 probability comparison applies (the model emits soft
      probabilities for benign Korean, where bf16 rounding alone exceeds
      atol=1e-2; see HIGH_CONFIDENCE_PROMPTS note).
    """
    prompts = [p for p, _ in KOREAN_CORPUS]
    expected_labels = [label for _, label in KOREAN_CORPUS]

    hf_outputs = apeach_hf_runner.classify(prompts)
    vllm_outputs = apeach_vllm_runner.classify(prompts)

    assert len(vllm_outputs) == len(hf_outputs) == len(KOREAN_CORPUS)

    argmax_mismatch = []
    for prompt, expected, hf_probs, vllm_probs in zip(prompts, expected_labels, hf_outputs, vllm_outputs):
        _check_probs(vllm_probs)
        hf_label = _argmax(hf_probs)
        vllm_label = _argmax(vllm_probs)
        assert hf_label == vllm_label, f"argmax mismatch for {prompt!r}: hf={hf_label} vllm={vllm_label}"
        if hf_label != expected:
            argmax_mismatch.append((prompt, expected, hf_label, vllm_label))

    assert not argmax_mismatch, f"label disagreements vs constructed ground truth: {argmax_mismatch}"

    # Strict probability tolerance for high-confidence samples only.
    hf_by_prompt = dict(zip(prompts, hf_outputs))
    vllm_by_prompt = dict(zip(prompts, vllm_outputs))
    for prompt in HIGH_CONFIDENCE_PROMPTS:
        hf_tensor = torch.tensor(hf_by_prompt[prompt])
        vllm_tensor = torch.tensor(vllm_by_prompt[prompt])
        assert torch.allclose(hf_tensor, vllm_tensor, atol=1e-2), (
            f"probability mismatch for {prompt!r}: hf={hf_by_prompt[prompt]} vllm={vllm_by_prompt[prompt]}"
        )


def test_single_vs_batch_outputs_are_consistent(apeach_vllm_runner: VllmRunner) -> None:
    """Classifying each prompt individually must agree with classifying the
    whole batch at once.

    Without VLLM_BATCH_INVARIANT the default NPU kernels are mildly
    batch-correlated (~2e-3 on this model), so probabilities are compared
    with the same atol=1e-2 tolerance used against the HF reference, while
    argmax labels must match exactly. Bit-exact single-vs-batch equality is
    achievable with VLLM_BATCH_INVARIANT=1, but that mode disables the
    Ascend custom op path (enable_custom_op() returns False), which this
    test suite deliberately keeps enabled; the tolerance is therefore the
    explicit, documented compromise.
    """
    prompts = [p for p, _ in KOREAN_CORPUS]

    batch_outputs = apeach_vllm_runner.classify(prompts)

    for prompt, batched in zip(prompts, batch_outputs):
        single = apeach_vllm_runner.classify([prompt])[0]
        assert _argmax(single) == _argmax(batched), (
            f"single-vs-batch argmax mismatch for {prompt!r}: {single} != {batched}"
        )
        assert torch.allclose(torch.tensor(single), torch.tensor(batched), atol=1e-2), (
            f"single-vs-batch mismatch for {prompt!r}: {single} != {batched}"
        )


def test_repeated_runs_are_stable(apeach_vllm_runner: VllmRunner) -> None:
    """Run-to-run stability criterion (substitute for greedy/non-greedy
    sampling on classification models): classifying the same input twice
    must return identical argmax labels and probabilities within the same
    atol=1e-2 tolerance used elsewhere.

    A classification forward pass contains no sampling node, so there is no
    seed-dependent randomness; the residual run-to-run difference on the
    Ascend custom-op path is pure kernel noise (~1e-3 observed), two orders
    of magnitude below the tolerance, and argmax labels are stable.
    """
    prompts = [p for p, _ in KOREAN_CORPUS]

    run_1 = apeach_vllm_runner.classify(prompts)
    run_2 = apeach_vllm_runner.classify(prompts)

    for prompt, out_1, out_2 in zip(prompts, run_1, run_2):
        assert _argmax(out_1) == _argmax(out_2), f"repeated-run argmax mismatch for {prompt!r}: {out_1} != {out_2}"
        assert torch.allclose(torch.tensor(out_1), torch.tensor(out_2), atol=1e-2), (
            f"repeated-run mismatch for {prompt!r}: {out_1} != {out_2}"
        )


def test_single_character_inputs(apeach_vllm_runner: VllmRunner) -> None:
    """Minimal-length inputs (one Korean char, one Latin char) must classify
    without crashing and return well-formed probabilities."""
    for prompt in ["안", "a"]:
        probs = apeach_vllm_runner.classify([prompt])[0]
        _check_probs(probs)


def test_overlong_input_is_rejected(apeach_vllm_runner: VllmRunner) -> None:
    """Inputs beyond max_model_len (4096) must be rejected explicitly."""
    # Each repetition is ~10 tokens; 2000 repetitions exceed 4096 tokens.
    long_prompt = "이것은 반복되는 한국어 문장입니다. " * 2000
    with pytest.raises(ValueError):
        apeach_vllm_runner.classify([long_prompt])


def test_invalid_inputs_are_rejected(apeach_vllm_runner: VllmRunner) -> None:
    """Empty string and non-string inputs must raise explicitly."""
    with pytest.raises(ValueError):
        apeach_vllm_runner.classify([""])
    with pytest.raises(TypeError):
        apeach_vllm_runner.classify([None])
    with pytest.raises(TypeError):
        apeach_vllm_runner.classify([["not", "a", "token", "id", "list"]])


def test_chinese_english_smoke(apeach_vllm_runner: VllmRunner) -> None:
    """Smoke-only for out-of-domain languages: the model is Korean-only, so
    these samples only assert that inference runs and returns sane
    probabilities (no accuracy assertion)."""
    for prompt in ZH_SMOKE + EN_SMOKE:
        probs = apeach_vllm_runner.classify([prompt])[0]
        _check_probs(probs)
