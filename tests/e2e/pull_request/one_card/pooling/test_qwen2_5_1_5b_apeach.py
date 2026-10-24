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
  identical argmax labels plus probabilities within the documented
  tolerance, and by a probability comparison against the HF fp32 reference.
  Measured on this corpus (33 samples, NPU, repeated sweeps on both
  published copies of the same weights): run-to-run variation is bit-exact
  (32/33 samples; ~1e-3 for the rest), single-vs-batch variation ~2e-3,
  and end-to-end bf16-vs-fp32 variation is <= 4e-3 for high-confidence
  samples but 2.0e-2 for one soft-probability sample (p ~ 0.69, where
  |dp| ~ p(1-p)|dz| amplifies the logit error ~4.7x). Tolerances are
  therefore tiered: atol=1e-2 on the high-confidence subset, atol=2.5e-2
  on the whole corpus, with exact argmax agreement everywhere.
* The HF reference runs in fp32 on device (same tolerance convention as the
  existing `test_classification.py` precedent), the vLLM side runs in bf16
  (fp16 is deliberately avoided: a same-lineage fp16 test was historically
  flaky and is on the CI skip list).
* Relation to `pooling/test_classification.py`: that file covers the same
  checkpoint but is skipped wholesale (`Model Runner V1 is outside the
  default CI path`), so it never executes on the default CI path. This file
  is the V1-agnostic coverage of the model that does run there.
* Korean corpus is self-constructed (no third-party license) with >= 15
  samples per label; Chinese/English samples are smoke-only (the model is
  Korean-only, so they only assert that inference runs without crashing).
* Environment requirement (primary): the `aclnnAddRmsNormBias` custom op
  kernel is shipped in the `libcust_opapi.so` vendor package and discovered
  via `ASCEND_CUSTOM_OPP_PATH`. Because `vllm_ascend_C.so` is loaded before
  Python-side `bootstrap_custom_op_env()` runs, the variable must be set
  before the process starts. Source the vendor environment script before
  running pytest (this is the PR-validated configuration; it keeps the
  custom op path active):

      . /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

  Fallback (used by CI, where the vendor script is not sourced):
  `VLLM_BATCH_INVARIANT=1` makes `enable_custom_op()` return `False`,
  selecting `torch_npu.npu_add_rms_norm` instead. When the vendor library
  is not discoverable the autouse fixture below exports
  `VLLM_BATCH_INVARIANT=1` in `os.environ` *before* the engine starts, so
  the EngineCore subprocess inherits it and the whole run uses the
  torch_npu native op path; with the vendor environment sourced the custom
  op path stays active.
"""

import glob
import os
from collections.abc import Generator

import pytest
import torch
from modelscope import snapshot_download  # type: ignore[import-untyped]
from transformers import AutoConfig, AutoModelForSequenceClassification
from vllm.exceptions import VLLMValidationError

from tests.e2e.conftest import (
    HfRunner,
    VllmRunner,
    cleanup_dist_env_and_memory,
    wait_until_npu_memory_free,
)

MODEL_NAME = "Howeee/Qwen2.5-1.5B-apeach"
# Pinned ModelScope commit (master as of 2026-10-24). The HF mirror
# jason9693/Qwen2.5-1.5B-apeach is the same checkpoint (commit 615164d4);
# the expected labels / confidences in this file are only valid for it.
REVISION = "68aef1ee42bff4db3c262492002c807ffd495edf"

# Self-constructed Korean corpus (no third-party license). Label 0 = Default
# (benign), label 1 = Spoiled (harmful/hate speech). The set deliberately
# includes formal/honorific speech, colloquial speech, and borderline
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

# Samples whose HF fp32 confidence is high enough (re-verified at runtime
# in test_classify_matches_hf_fp32_reference, > 0.9) that they sit far from
# the 0.5 decision boundary: the measured end-to-end bf16-vs-fp32 delta
# (<= 4e-3 for this subset) cannot flip their argmax. They carry the strict
# atol=1e-2 probability assertion; the rest of the corpus is compared with
# the looser atol=2.5e-2 documented in that test.
#
# "가서 뒤져라 다시 나타나지 마" is deliberately absent: its HF fp32
# confidence is only 0.9015 (measured), which would leave ~2e-3 of margin
# to the 0.9 anchor. It is still covered by the general probability and
# argmax assertions.
HIGH_CONFIDENCE_PROMPTS = [
    "안녕하세요 오늘 날씨가 참 좋네요",
    "이 영화 정말 재미있었어요 다음에 또 보고 싶습니다",
    "네가 하는 말은 전부 쓰레기 같아",
    "멍청한 놈 그걸 몰라서야 어떻게 사냐",
    "못생긴 게 성격까지 그 모양이냐",
    "입 닥치고 조용히 좀 해 줄래",
    "너 같으면 부끄러워서 살 수나 있겠냐",
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
    used, matching the existing `test_classification.py` precedent. The
    ModelScope revision is pinned to the commit this corpus was written
    against, so a later re-upload cannot silently change the reference
    probabilities.
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


@pytest.fixture(scope="session", autouse=True)
def _fallback_when_ascend_custom_op_unavailable() -> Generator[None, None, None]:
    """Fall back to `torch_npu.npu_add_rms_norm` only when the custom op
    vendor library is not discoverable.

    `vllm_ascend.ops.layernorm.AscendRMSNorm.forward_oot` prefers the
    custom op `_C_ascend.npu_add_rms_norm_bias`. When
    `ASCEND_CUSTOM_OPP_PATH` is missing (the `set_env.bash` of the
    `_cann_ops_custom` vendor package was not sourced), resolving that op
    raises `aclnnAddRmsNormBias ... not in libopapi.so` at the first
    forward pass. In that case, and only in that case, `VLLM_BATCH_INVARIANT`
    is exported so the native `torch_npu.npu_add_rms_norm` path is
    selected; with a correctly configured environment the custom op path
    is used as-is.

    The environment variable - not just the `_CUSTOM_OP_ENABLED` cache - is
    what matters: the engine runs in an `EngineCore` subprocess, which only
    inherits the parent's environment. This fixture is session-scoped and
    autouse, and `apeach_vllm_runner` depends on it explicitly, so it is
    always set up before the engine is created.
    """
    from vllm_ascend import utils as ascend_utils

    old_flag = ascend_utils._CUSTOM_OP_ENABLED
    old_env = os.environ.get("VLLM_BATCH_INVARIANT")
    if not _custom_op_env_ready():
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        ascend_utils._CUSTOM_OP_ENABLED = False
    yield
    ascend_utils._CUSTOM_OP_ENABLED = old_flag
    if old_env is None:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)
    else:
        os.environ["VLLM_BATCH_INVARIANT"] = old_env


@pytest.fixture(scope="session")
@wait_until_npu_memory_free(target_free_percentage=0.7)
def _npu_memory_free() -> None:
    """Wait for HBM left over by previous tests to be released before the
    engine reserves its pool (same guard as `test_classification.py`)."""


@pytest.fixture(scope="session")
def apeach_vllm_runner(
    model_path: str,
    _npu_memory_free: None,
    _fallback_when_ascend_custom_op_unavailable: None,
) -> Generator[VllmRunner, None, None]:
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
        # The classify adapter's LAST-token pooling is only an implicit
        # default (get_default_seq_pooling_type falls back to "LAST" via
        # getattr); pin it explicitly so the HF reference comparison does
        # not rely on that fallback.
        pooler_config={"pooling_type": "LAST"},
    )
    try:
        yield runner
    finally:
        # VllmRunner.__exit__ shuts down the EngineCore process, drops the
        # model and clears the ascend config / distributed state; going
        # through it (rather than a bare `del`) keeps the engine child from
        # outliving the test session.
        runner.__exit__(None, None, None)


@pytest.fixture(scope="session")
def apeach_hf_runner(model_path: str) -> Generator[HfRunner, None, None]:
    """HF fp32 reference (same tolerance convention as the existing
    `test_classification.py` precedent)."""
    cleanup_dist_env_and_memory()
    runner = HfRunner(model_path, dtype="float32", auto_cls=AutoModelForSequenceClassification)
    try:
        yield runner
    finally:
        runner.__exit__(None, None, None)


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
    * probabilities agree with the fp32 reference for **all 33 samples**
      within atol=2.5e-2, and the `HIGH_CONFIDENCE_PROMPTS` subset agrees
      within atol=1e-2. Measured on this corpus (NPU, bf16 vs fp32, repeated
      sweeps): the max |delta| over the subset is 4.0e-3, while the whole
      corpus is dominated by one soft-probability sample at 1.98e-2
      (HF fp32 p=0.6907, vLLM 0.7105; deterministic, bit-identical across
      runs and independent of batching). The tiered tolerance keeps the
      strict check where it is meaningful and still bounds the soft samples
      (~25% headroom over the measured max);
    * the `HIGH_CONFIDENCE_PROMPTS` samples are re-checked to stay far from
      the decision boundary (HF fp32 confidence above ~0.9 at runtime), so
      the measured delta provably cannot flip their argmax.
    """
    prompts = [p for p, _ in KOREAN_CORPUS]
    expected_labels = [label for _, label in KOREAN_CORPUS]

    hf_outputs = apeach_hf_runner.classify(prompts)
    vllm_outputs = apeach_vllm_runner.classify(prompts)

    assert len(vllm_outputs) == len(hf_outputs) == len(KOREAN_CORPUS)

    argmax_mismatch = []
    for prompt, expected, hf_probs, vllm_probs in zip(prompts, expected_labels, hf_outputs, vllm_outputs):
        _check_probs(vllm_probs)
        # HfRunner.classify is annotated `list[str]` in conftest but actually
        # returns per-prompt probability lists (`list[list[float]]`).
        hf_label = _argmax(hf_probs)  # type: ignore[arg-type]
        vllm_label = _argmax(vllm_probs)
        assert hf_label == vllm_label, f"argmax mismatch for {prompt!r}: hf={hf_label} vllm={vllm_label}"
        # Whole-corpus tolerance: 2.5e-2, i.e. ~25% headroom over the
        # measured max (1.98e-2, one soft-probability sample). bf16 error in
        # the logits (~5e-3 over 28 layers) is amplified by 1/(p(1-p)) ~ 4.7
        # in probability space for p ~ 0.7, so 1e-2 is not attainable here.
        assert torch.allclose(torch.tensor(hf_probs), torch.tensor(vllm_probs), atol=2.5e-2), (
            f"probability mismatch for {prompt!r}: hf={hf_probs} vllm={vllm_probs}"
        )
        if hf_label != expected:
            argmax_mismatch.append((prompt, expected, hf_label, vllm_label))

    assert not argmax_mismatch, f"label disagreements vs constructed ground truth: {argmax_mismatch}"

    # High-confidence subset: strict atol=1e-2 plus a margin guard. The
    # samples must stay far from the 0.5 decision boundary (measured bf16-vs-
    # fp32 delta here is <= 4e-3), and the confidence is re-computed at
    # runtime rather than hardcoded; the -2e-3 slack absorbs device-to-device
    # fp32 variation of the HF reference (lowest measured: 0.9047, benign).
    hf_by_prompt = dict(zip(prompts, hf_outputs))
    vllm_by_prompt = dict(zip(prompts, vllm_outputs))
    for prompt in HIGH_CONFIDENCE_PROMPTS:
        # Same conftest annotation mismatch as above: entries are probability
        # lists, not the `str` the annotation claims.
        hf_ref = hf_by_prompt.get(prompt)
        assert hf_ref is not None, f"stale HIGH_CONFIDENCE_PROMPTS entry {prompt!r}: not part of KOREAN_CORPUS"
        vllm_ref = vllm_by_prompt.get(prompt)
        assert vllm_ref is not None, f"missing vllm output for {prompt!r}"
        hf_tensor = torch.tensor(hf_ref)
        hf_confidence = hf_tensor.max().item()
        assert hf_confidence > 0.9 - 2e-3, (
            f"stale HIGH_CONFIDENCE_PROMPTS entry {prompt!r}: HF fp32 confidence {hf_confidence:.4f} too close to 0.5"
        )
        assert torch.allclose(hf_tensor, torch.tensor(vllm_ref), atol=1e-2), (
            f"probability mismatch for high-confidence {prompt!r}: hf={hf_ref} vllm={vllm_ref}"
        )


def test_single_vs_batch_outputs_are_consistent(apeach_vllm_runner: VllmRunner) -> None:
    """Classifying each prompt individually must agree with classifying the
    whole batch at once.

    On the custom op path the default NPU kernels are mildly
    batch-correlated (~2e-3 measured on this model), so probabilities are
    compared with the same atol=1e-2 tolerance used against the HF
    reference, while argmax labels must match exactly. Under the fallback
    configuration (VLLM_BATCH_INVARIANT=1, which is what CI uses because
    the vendor environment is not sourced there) single-vs-batch results
    are bit-exact; the tolerance is kept so the assertion is identical on
    both configurations.
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
    must return identical argmax labels and probabilities within the
    atol=1e-2 tolerance used by the other stability checks.

    A classification forward pass contains no sampling node, so there is no
    seed-dependent randomness; the residual run-to-run difference is pure
    kernel noise (~1e-3 observed), an order of magnitude below the
    tolerance, and argmax labels are stable.
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
    """Inputs beyond max_model_len (4096) must be rejected explicitly.

    The exception type follows the vLLM prompt-validation implementation:
    on the 0.26 local pin an overlong prompt raises a plain `ValueError`;
    from 0.31 the engine wraps validation failures in `VLLMValidationError`,
    whose base class changed from `ValueError` to `VLLMClientError`, so it
    no longer matches `ValueError`. Accepting both keeps the test valid on
    either pin.
    """
    # Each repetition is ~10 tokens; 2000 repetitions exceed 4096 tokens.
    long_prompt = "이것은 반복되는 한국어 문장입니다. " * 2000
    with pytest.raises((ValueError, VLLMValidationError)):
        apeach_vllm_runner.classify([long_prompt])


def test_invalid_inputs_are_rejected(apeach_vllm_runner: VllmRunner) -> None:
    """Empty string and non-string inputs must raise explicitly.

    The empty-string case has the same version caveat as
    `test_overlong_input_is_rejected`; non-string inputs are rejected by
    the classify adapter itself and raise `TypeError` on both pins.
    """
    with pytest.raises((ValueError, VLLMValidationError)):
        apeach_vllm_runner.classify([""])
    with pytest.raises(TypeError):
        apeach_vllm_runner.classify([None])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        apeach_vllm_runner.classify([["not", "a", "token", "id", "list"]])  # type: ignore[list-item]


def test_chinese_english_smoke(apeach_vllm_runner: VllmRunner) -> None:
    """Smoke-only for out-of-domain languages: the model is Korean-only, so
    these samples only assert that inference runs and returns sane
    probabilities (no accuracy assertion)."""
    for prompt in ZH_SMOKE + EN_SMOKE:
        probs = apeach_vllm_runner.classify([prompt])[0]
        _check_probs(probs)
