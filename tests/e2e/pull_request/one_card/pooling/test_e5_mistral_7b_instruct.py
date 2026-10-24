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
"""Offline inference correctness tests for intfloat/e5-mistral-7b-instruct.

The model is a MistralModel (no LM head) embedding model that uses
last-token pooling with L2 normalization and recommends at most 4096
tokens per input. On vLLM 0.26 the embedding pooler head applies
L2 normalization by default (the `normalize` field was removed from
PoolerConfig), so only `pooling_type=LAST` needs to be requested. Query inputs must carry the asymmetric instruction
prefix while passages stay bare text (model card).

Assertion set (issue #10679):
  1. output shape (n, 4096);
  2. element-wise |delta| <= 2e-2 after both sides are L2-normalized
     (cosine alone cannot catch normalization bugs);
  3. cosine >= 1 - 1e-3 (via check_embeddings_close);
  4. top-1 ranking over a fixed query x passage set matches the HF reference,
     and every query retrieves its intended passage (40/40);
  5. token-id alignment between vLLM and the HF tokenizer (BOS and the
     last real token; neither side appends </s>);
  6. > max_model_len raises a validation error (no silent truncation); the
     empty / whitespace-only behavior is documented: vLLM 0.26 tokenizes it to
     a single BOS token and pools from it, vLLM >= 0.31 rejects it;
  7. non-string inputs (scalar, bytes, dict, mixed list) are rejected with
     TypeError / ValueError instead of returning an embedding.

Corpus licensing: the 40 query-passage pairs below are written for this test
and are the only text sent to the model. No MS MARCO (or any other
third-party) corpus is downloaded or redistributed; the file itself stays
under the repository's Apache-2.0 license.
"""

import contextlib
import hashlib
import os
import tempfile
import time
from queue import Empty
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tests.e2e.conftest import VllmRunner
from tests.e2e.utils import check_embeddings_close

try:
    # vLLM >= 0.31 wraps prompt-validation failures in VLLMValidationError,
    # which no longer derives from ValueError there, while older pins only
    # raise ValueError. Probe the symbol instead of importing it blindly.
    from vllm.exceptions import VLLMValidationError
except ImportError:  # pragma: no cover - depends on the vLLM pin
    VLLMValidationError = None  # type: ignore[assignment,misc]

#: Exceptions that count as "input rejected" by the prompt-validation tests.
EXPECTED_VALIDATION_ERRORS: tuple[type[BaseException], ...] = (ValueError,)
if VLLMValidationError is not None:
    EXPECTED_VALIDATION_ERRORS = (ValueError, VLLMValidationError)

MODEL = "intfloat/e5-mistral-7b-instruct"
# Point this at a local copy of the weight to skip the hub lookup entirely
# (CI pre-downloads the model and may export it; air-gapped hosts can too).
MODEL_PATH_ENV = "E5_MISTRAL_MODEL_PATH"

# Official instruction template (model card): queries only, passages bare.
DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"
QUERY_PREFIX = f"Instruct: {DEFAULT_TASK}\nQuery: "

HIDDEN_SIZE = 4096
MAX_MODEL_LEN = 4096

# Element-wise tolerance after both sides are L2-normalized (bf16 vs fp32).
ELEMENTWISE_TOL = 2e-2
# Cosine tolerance used by check_embeddings_close (cosine >= 1 - COSINE_TOL).
COSINE_TOL = 1e-3
# Single-input (n=1) embedding vs the same text's row in the batched run: same
# weights and dtype, but the n=1 and n=80 kernels reduce in a different order,
# so a small bf16 difference is expected (measured ~1.5e-3 on this corpus).
SINGLE_INPUT_TOL = 5e-3

# Wall-clock budget for the spawned HF reference worker (7B fp32 load + forward).
HF_REFERENCE_TIMEOUT_S = 600

# Config fingerprint of the vLLM engine, folded into every _CACHE key: a
# dtype/pooler/max_model_len change must not reuse stale cached embeddings.
ENGINE_FINGERPRINT = f"dtype=bfloat16,pooler=LAST,max_model_len={MAX_MODEL_LEN}"

# Long English input, guaranteed to exceed MAX_MODEL_LEN tokens.
OVERSIZE_TEXT = "retrieval passage about renewable energy sources and storage. " * 1200

# Self-constructed bilingual corpus (no third-party text is redistributed).
# Each entry is (query, relevant passage); queries carry the instruction
# prefix, passages stay bare. Chinese rows are observation rows: the model
# card recommends English for best accuracy.
CORPUS: list[tuple[str, str]] = [
    # ---- English ----
    (
        "how much protein should a female eat daily",
        "The CDC recommends that women aged 19 to 70 consume about 46 grams of protein per day "
        "on average, though needs increase during pregnancy or intense marathon training.",
    ),
    (
        "what is the definition of a summit in geography",
        "In geography, a summit is the highest point of a mountain or hill, the top of its "
        "surface above the surrounding slopes.",
    ),
    (
        "how does photosynthesis convert light into energy",
        "Photosynthesis is the process by which green plants use sunlight, water and carbon "
        "dioxide to produce glucose and release oxygen.",
    ),
    (
        "why do volcanoes erupt",
        "Volcanoes erupt when magma, gas and ash from beneath the Earth's crust rise through "
        "vents in the surface, driven by pressure and buoyancy.",
    ),
    (
        "what causes tides in the ocean",
        "Ocean tides are caused mainly by the gravitational pull of the moon and the sun on "
        "the rotating Earth, producing two high and two low tides each day.",
    ),
    (
        "how do vaccines train the immune system",
        "Vaccines introduce a harmless piece or marker of a pathogen so the immune system can "
        "recognize it and mount a faster response upon future exposure.",
    ),
    (
        "what is a black hole",
        "A black hole is a region of spacetime where gravity is so strong that nothing, not "
        "even light, can escape once it crosses the event horizon.",
    ),
    (
        "why is the Great Barrier Reef important",
        "The Great Barrier Reef off Australia is the world's largest coral reef system, hosting "
        "thousands of marine species and protecting coastlines from storms.",
    ),
    (
        "how much caffeine is in a cup of coffee",
        "A typical 240-milliliter cup of brewed coffee contains about 95 milligrams of caffeine, "
        "though the amount varies with bean type and brewing method.",
    ),
    (
        "what is blockchain technology used for",
        "Blockchain is a distributed ledger technology that records transactions in linked, "
        "tamper-resistant blocks, used for cryptocurrencies, supply chain tracking and digital "
        "identity.",
    ),
    (
        "who was socrates and what did he teach",
        "Socrates was an ancient Greek philosopher who taught by asking probing questions, a "
        "method later called the Socratic method, and emphasized self-knowledge and virtue.",
    ),
    (
        "how does the human heart pump blood",
        "The human heart pumps blood through four chambers: the right side sends blood to the "
        "lungs to pick up oxygen, and the left side pumps it to the rest of the body.",
    ),
    (
        "what makes lightning happen during a thunderstorm",
        "Lightning happens when electrical charges separate within a storm cloud and discharge "
        "between clouds or to the ground, heating the air into a bright plasma channel.",
    ),
    (
        "why do leaves change color in autumn",
        "Leaves change color in autumn because chlorophyll breaks down as days shorten, "
        "revealing yellow and orange carotenoid pigments that were present all along.",
    ),
    (
        "what is the function of the ozone layer",
        "The ozone layer in the stratosphere absorbs most of the sun's harmful ultraviolet "
        "radiation, protecting living organisms from DNA damage and skin cancer.",
    ),
    (
        "how do bees make honey",
        "Bees make honey by collecting nectar, storing it in the hive, and fanning it with "
        "their wings until much of the water evaporates and it thickens into honey.",
    ),
    (
        "what is inflation in economics",
        "Inflation is the rate at which the general level of prices for goods and services "
        "rises, eroding purchasing power over time.",
    ),
    (
        "how do earthquakes release energy",
        "Earthquakes release energy when stress accumulated along geological faults suddenly "
        "slips, sending seismic waves through the Earth's crust.",
    ),
    (
        "what languages are spoken in brazil",
        "The official language of Brazil is Portuguese, brought during colonization, and the "
        "country also recognizes numerous indigenous languages.",
    ),
    (
        "how does the international space station stay in orbit",
        "The International Space Station stays in orbit by traveling at about 7.66 kilometers "
        "per second, so its forward motion balances Earth's gravitational pull.",
    ),
    # ---- Chinese ----
    (
        "中国最长的河流是哪条",
        "长江是中国最长的河流，全长约6300公里，发源于青藏高原，自西向东流入东海。",
    ),
    (
        "光合作用主要在植物哪个部位进行",
        "光合作用主要在植物的叶片中进行，叶绿体利用光能把二氧化碳和水转化成有机物并释放氧气。",
    ),
    (
        "熊猫主要吃什么食物",
        "大熊猫主要以竹子为食，每天要花十几个小时进食，也会偶尔吃一些小动物或果实。",
    ),
    (
        "长城修建的主要目的是什么",
        "长城最初修建的目的是防御北方游牧民族的入侵，后来逐渐成为贸易管理和边疆控制的屏障。",
    ),
    (
        "为什么天空是蓝色的",
        "天空呈现蓝色是因为阳光进入大气层后发生瑞利散射，波长较短的蓝光比红光更容易被散射。",
    ),
    (
        "高铁和普通铁路有什么区别",
        "高铁使用专用的无砟轨道和电力动车组，运行速度可达每小时350公里，而普通铁路一般低于160公里。",
    ),
    (
        "二十四节气有什么作用",
        "二十四节气反映太阳运行周期和季节变化，用于指导传统农业的播种、收获和日常生活安排。",
    ),
    (
        "口罩为什么能防护呼吸道传染病",
        "口罩通过纤维层拦截飞沫和颗粒物，减少病原体经呼吸道传播的风险，密合性好的口罩防护效果更佳。",
    ),
    (
        "京剧有哪些主要行当",
        "京剧的主要行当有生、旦、净、丑，分别扮演不同性别、年龄、性格和身份的舞台人物。",
    ),
    (
        "移动支付在中国普及的原因",
        "移动支付在中国普及得益于智能手机普及、二维码标准统一和平台补贴，使小额交易变得便捷高效。",
    ),
    (
        "台风是如何形成的",
        "台风是热带气旋在温暖洋面上发展而成的，海面温度高于26.5摄氏度、水汽充沛时容易生成并逐渐增强。",
    ),
    (
        "垃圾分类有什么好处",
        "垃圾分类可以提高资源回收利用率，减少填埋和焚烧带来的污染，降低生活垃圾处理的总体成本。",
    ),
    (
        "为什么煮饺子时要加水",
        "煮饺子时加水能降低锅内的温度，防止沸水翻滚导致饺子皮破裂，让饺子受热更均匀。",
    ),
    (
        "5G网络比4G快在哪里",
        "5G网络使用更高的频段和大规模天线技术，峰值速率比4G快数十倍，延迟也更低。",
    ),
    (
        "什么是碳中和",
        "碳中和是指在一定时期内通过节能减排、植树造林等方式抵消二氧化碳排放，实现净零排放。",
    ),
    (
        "长江三角洲包括哪些城市",
        "长江三角洲主要包括上海、江苏南部、浙江北部和安徽部分地区，是中国经济最活跃的区域之一。",
    ),
    (
        "茶叶主要有哪几种发酵程度",
        "茶叶按发酵程度分为不发酵的绿茶、半发酵的乌龙茶和全发酵的红茶，此外还有后发酵的黑茶。",
    ),
    (
        "为什么太阳能热水器能加热水",
        "太阳能热水器通过集热板吸收太阳辐射能，将热量传递给水管中的水，从而把水加热到使用温度。",
    ),
    (
        "兵马俑有什么历史价值",
        "兵马俑是秦始皇陵的陪葬坑，展现了秦代军事、雕塑和铸造工艺水平，为了解秦代历史提供了实物证据。",
    ),
    (
        "电子发票和纸质发票有什么不同",
        "电子发票以电子数据形式存在，开具、存储和查验都通过网络完成，与纸质发票具有同等法律效力。",
    ),
]

# Cache for expensive artifacts (7B model forward / engine start) so the
# assertion set can share them on single-core CPU machines.
# NOTE: "vllm_tokenizer" holds the tokenizer of an engine that has already
# been destroyed by VllmRunner.__exit__; only its CPU-side tokenizer methods
# (encode/decode) are safe to reuse in later tests.
# NOTE: the cache lives in process memory and therefore assumes this file runs
# in a single process, serially. Under pytest-xdist every worker is a separate
# interpreter with its own empty cache, so the engines would be started once
# per worker for the same texts -- do not rely on _CACHE for cross-worker reuse.
_CACHE: dict[str, Any] = {}


def _cache_key(prefix: str, texts: list[str]) -> str:
    """Collision-resistant cache key: engine fingerprint + full-content sha256."""
    digest = hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}::{ENGINE_FINGERPRINT}::{len(texts)}::{digest}"


def instruct_query(query: str) -> str:
    """Official query template: the instruction prefix is added on the query side only."""
    return QUERY_PREFIX + query


def _hf_last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Official last-token pooling from the e5-mistral model card."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _model_path() -> str:
    """Resolve the model directory.

    E5_MISTRAL_MODEL_PATH wins when it is set, so CI (or an air-gapped host)
    can point at a pre-downloaded copy. Otherwise the weight is resolved
    through ModelScope, going offline only when the environment asks for it:
    CI exports HF_HUB_OFFLINE=1 and relies on the model cache populated from
    .github/workflows/misc/model_dataset_list.json, so a model missing from
    that list fails here instead of silently reaching out to the network.
    """
    override = os.environ.get(MODEL_PATH_ENV)
    if override:
        path = os.path.expanduser(override)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{MODEL_PATH_ENV}={path!r} is not a directory")
        return path

    from modelscope import snapshot_download  # type: ignore[import-untyped]

    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
    return snapshot_download(MODEL, local_files_only=offline)


def _worker_stderr(log_path: str) -> str:
    """Return the tail of a spawn worker's stderr log (best effort)."""
    try:
        with open(log_path, errors="replace") as handle:
            tail = handle.read()[-4000:]
    except OSError:
        return "<stderr unavailable>"
    return tail.strip() or "<empty>"


def _hf_reference_worker(texts: list[str], model_name: str, queue: Any, log_path: str) -> None:
    """Worker body for the spawned HF reference process (see _hf_reference_embeddings)."""
    # Send the OS-level stderr (Python tracebacks as well as torch/CANN
    # messages) to log_path, so the parent can attach it when the worker dies
    # without putting a result on the queue.
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    model = model.to(device).eval()

    all_features: list[np.ndarray] = []
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                # No truncation: the reference must see the same tokens as vLLM
                # (vLLM raises on oversize input instead of truncating).
                truncation=False,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            features = _hf_last_token_pool(hidden, encoded["attention_mask"])
            features = F.normalize(features, p=2, dim=1)
            all_features.append(features.float().cpu().numpy())

    queue.put(np.concatenate(all_features, axis=0))


def _hf_reference_embeddings(texts: list[str]) -> np.ndarray:
    """HF reference: AutoModel + official last_token_pool + L2 norm (float32).

    The reference runs in a spawned subprocess: initializing NPU in the main
    pytest process would break later VllmRunner engine cores, which are
    forked ("Cannot re-initialize NPU in forked subprocess").
    """
    key = _cache_key("hf", texts)
    if key in _CACHE:
        return _CACHE[key]

    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    log_fd, log_path = tempfile.mkstemp(prefix="e5_mistral_hf_ref_", suffix=".log")
    os.close(log_fd)
    proc = ctx.Process(target=_hf_reference_worker, args=(texts, _model_path(), queue, log_path))
    proc.start()

    embeddings: np.ndarray | None = None
    try:
        # Poll with a short timeout instead of one long blocking get(): if the
        # worker dies (OOM, CANN abort) the test must fail right away, with the
        # worker's stderr attached.
        deadline = time.monotonic() + HF_REFERENCE_TIMEOUT_S
        while embeddings is None:
            try:
                embeddings = queue.get(timeout=1)
            except Empty:
                if not proc.is_alive():
                    proc.join()
                    raise RuntimeError(
                        f"HF reference worker died (exitcode={proc.exitcode}) before "
                        f"delivering embeddings. stderr:\n{_worker_stderr(log_path)}"
                    ) from None
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"HF reference worker returned no embeddings within "
                        f"{HF_REFERENCE_TIMEOUT_S}s (still alive). stderr:\n{_worker_stderr(log_path)}"
                    ) from None

        proc.join(timeout=HF_REFERENCE_TIMEOUT_S)
        if proc.is_alive():
            raise RuntimeError("HF reference worker did not exit after delivering embeddings")
        if proc.exitcode != 0:
            raise RuntimeError(
                f"HF reference worker exited abnormally: exitcode={proc.exitcode}. stderr:\n{_worker_stderr(log_path)}"
            )
    except BaseException:
        proc.terminate()
        proc.join(timeout=30)
        raise
    finally:
        queue.close()
        queue.join_thread()
        proc.close()
        with contextlib.suppress(OSError):
            os.unlink(log_path)

    assert embeddings.shape == (len(texts), HIDDEN_SIZE)
    _CACHE[key] = embeddings
    return embeddings


def _vllm_embeddings(texts: list[str]) -> np.ndarray:
    """vLLM embeddings via the pooling runner with the LAST + normalize override."""
    key = _cache_key("vllm", texts)
    if key in _CACHE:
        return _CACHE[key]

    with VllmRunner(
        _model_path(),
        runner="pooling",
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        pooler_config={"pooling_type": "LAST"},  # v0.26+: embed head L2-normalizes by default
    ) as vllm_runner:
        outputs = vllm_runner.embed(texts)
        tokenizer = vllm_runner.model.get_tokenizer()

    embeddings = np.asarray(outputs, dtype=np.float32)
    # No explicit cleanup_dist_env_and_memory() here: VllmRunner.__exit__
    # (tests/e2e/conftest.py) already shuts down the engine core and cleans up.
    _CACHE["vllm_tokenizer"] = tokenizer
    _CACHE[key] = embeddings
    return embeddings


def _vllm_tokenizer() -> Any:
    """vLLM tokenizer, loaded once via a VllmRunner and cached."""
    if "vllm_tokenizer" not in _CACHE:
        _vllm_embeddings([instruct_query(CORPUS[0][0])])
    return _CACHE["vllm_tokenizer"]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="embedding",
    feature="pooling",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_e5_mistral_embedding_shape():
    """Output shape must be (n, 4096)."""
    texts = [instruct_query(q) for q, _ in CORPUS] + [p for _, p in CORPUS]
    embeddings = _vllm_embeddings(texts)
    assert embeddings.shape == (len(texts), HIDDEN_SIZE), (
        f"expected shape ({len(texts)}, {HIDDEN_SIZE}), got {embeddings.shape}"
    )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="embedding",
    feature="pooling",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_e5_mistral_embedding_close():
    """Element-wise |delta| after both sides L2-normalize, plus cosine check."""
    queries = [instruct_query(q) for q, _ in CORPUS]
    passages = [p for _, p in CORPUS]

    vllm_out = _vllm_embeddings(queries + passages)
    hf_out = _hf_reference_embeddings(queries + passages)

    # Explicit normalization check: the pooler must L2-normalize on its own.
    # Without this, a missing normalization on the vLLM side would be masked
    # by the _l2_normalize calls below (the re-normalization would hide it).
    vllm_norms = np.linalg.norm(vllm_out, axis=1)
    assert np.allclose(vllm_norms, 1.0, atol=1e-3), (
        f"vLLM outputs are not L2-normalized: norms in [{vllm_norms.min()}, {vllm_norms.max()}]"
    )
    hf_norms = np.linalg.norm(hf_out, axis=1)
    assert np.allclose(hf_norms, 1.0, atol=1e-3), (
        f"HF outputs are not L2-normalized: norms in [{hf_norms.min()}, {hf_norms.max()}]"
    )

    vllm_norm = _l2_normalize(vllm_out)
    hf_norm = _l2_normalize(hf_out)

    delta = np.abs(vllm_norm - hf_norm)
    max_delta = float(delta.max())
    assert max_delta <= ELEMENTWISE_TOL, (
        f"max element-wise |delta| after L2 normalization = {max_delta:.4f} > {ELEMENTWISE_TOL}"
    )

    check_embeddings_close(
        embeddings_0_lst=hf_norm.tolist(),
        embeddings_1_lst=vllm_norm.tolist(),
        name_0="hf",
        name_1="vllm",
        tol=COSINE_TOL,
    )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="embedding",
    feature="pooling",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_e5_mistral_single_input_consistency():
    """Each text embedded alone must match its row in the batched result.

    Covers the single-input code path: every solo call below passes exactly one
    prompt, so the engine runs the n=1 path, and the returned vector must agree
    with the corresponding row of the batched run (batching and padding must
    not change the pooled vector).
    """
    queries = [instruct_query(q) for q, _ in CORPUS]
    passages = [p for _, p in CORPUS]
    batch = _vllm_embeddings(queries + passages)

    # One English query, its passage, and one Chinese query (last corpus row).
    solo_texts = [queries[0], passages[0], queries[-1]]
    # Corresponding rows in the batched run: queries[0] -> 0,
    # passages[0] -> len(queries), queries[-1] -> len(queries) - 1.
    ref_rows = [0, len(queries), len(queries) - 1]

    for text, ref_row in zip(solo_texts, ref_rows):
        solo = np.asarray(_vllm_embeddings([text]), dtype=np.float32)
        assert solo.shape == (1, HIDDEN_SIZE), (
            f"single-input embedding for {text[:40]!r} has shape {solo.shape}, expected (1, {HIDDEN_SIZE})"
        )
        solo_vec = solo[0]

        norm = float(np.linalg.norm(solo_vec))
        assert abs(norm - 1.0) <= 1e-3, f"single-input output for {text[:40]!r} is not L2-normalized: norm={norm}"

        delta = float(np.abs(solo_vec - batch[ref_row]).max())
        assert delta <= SINGLE_INPUT_TOL, (
            f"single-input embedding diverges from batched row {ref_row} for {text[:40]!r}: "
            f"max |delta| = {delta:.4f} > {SINGLE_INPUT_TOL}"
        )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="embedding",
    feature="pooling",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_e5_mistral_top1_ranking():
    """Top-1 over a fixed query x passage set: HF agreement and 40/40 intended pairs."""
    queries = [instruct_query(q) for q, _ in CORPUS]
    passages = [p for _, p in CORPUS]

    vllm_out = _vllm_embeddings(queries + passages)
    hf_out = _hf_reference_embeddings(queries + passages)
    vllm_norm = _l2_normalize(vllm_out)
    hf_norm = _l2_normalize(hf_out)

    vllm_q, vllm_p = vllm_norm[: len(queries)], vllm_norm[len(queries) :]
    hf_q, hf_p = hf_norm[: len(queries)], hf_norm[len(queries) :]

    vllm_scores = vllm_q @ vllm_p.T
    hf_scores = hf_q @ hf_p.T

    vllm_top1 = vllm_scores.argmax(axis=1)
    hf_top1 = hf_scores.argmax(axis=1)
    mismatches = [(i, int(hf_top1[i]), int(vllm_top1[i])) for i in range(len(queries)) if hf_top1[i] != vllm_top1[i]]
    assert not mismatches, f"top-1 ranking mismatches (query_idx, hf_top1, vllm_top1): {mismatches}"

    # The corpus is 40 one-to-one (query, relevant passage) pairs, so the
    # intended top-1 of query i is passage i: vLLM must hit 40/40 intended pairs.
    intended = np.arange(len(queries))
    off_intended = np.flatnonzero(vllm_top1 != intended)
    assert not len(off_intended), (
        f"vLLM top-1 missed the intended pair for {len(off_intended)}/{len(queries)} queries: "
        f"[(query_idx, got)] = {[(int(i), int(vllm_top1[i])) for i in off_intended]}"
    )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="embedding",
    feature="pooling",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_e5_mistral_token_id_alignment():
    """vLLM and HF tokenizers must agree on token ids (BOS and the last real token)."""
    hf_tokenizer = AutoTokenizer.from_pretrained(_model_path())
    vllm_tokenizer = _vllm_tokenizer()

    # English rows plus Chinese rows (the corpus is 20 English + 20 Chinese
    # entries) so the alignment also covers multilingual tokenization.
    sampled = CORPUS[:4] + CORPUS[20:24]
    texts = [instruct_query(q) for q, _ in sampled] + [p for _, p in sampled]

    for text in texts:
        vllm_ids = vllm_tokenizer.encode(text, add_special_tokens=True)
        hf_ids = hf_tokenizer.encode(text, add_special_tokens=True)
        assert vllm_ids == hf_ids, (
            f"token id mismatch for {text[:40]!r}: "
            f"vllm[:8]={vllm_ids[:8]} hf[:8]={hf_ids[:8]} "
            f"vllm[-4:]={vllm_ids[-4:]} hf[-4:]={hf_ids[-4:]}"
        )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="embedding",
    feature="pooling",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_e5_mistral_invalid_inputs():
    """Negative / boundary input behavior.

    - Inputs beyond max_model_len are rejected (no silent truncation). The
      exception type follows the vLLM pin: a plain ValueError on 0.26, a
      VLLMValidationError on >= 0.31 (see EXPECTED_VALIDATION_ERRORS).
    - Empty / whitespace-only input: issue #10679 asks for an explicit error.
      Measured on vLLM 0.26 (offline LLM.embed and /v1/embeddings alike): the
      string tokenizes to a single BOS token and the engine returns a valid
      (1, 4096) embedding pooled from it instead of raising. vLLM >= 0.31
      rejects it during prompt validation, so both outcomes are accepted;
      what must not happen is a crash of another kind or a wrong shape.
    """
    with VllmRunner(
        _model_path(),
        runner="pooling",
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        pooler_config={"pooling_type": "LAST"},  # v0.26+: embed head L2-normalizes by default
    ) as runner:
        with pytest.raises(EXPECTED_VALIDATION_ERRORS, match="(?i)too long|maximum context|longer than"):
            runner.embed([OVERSIZE_TEXT])

        for empty_text in ("", "   \n\t  "):
            try:
                outputs = runner.embed([empty_text])
            except EXPECTED_VALIDATION_ERRORS:
                # vLLM >= 0.31 rejects empty prompts in validation; 0.26
                # tokenizes them to a single BOS token and pools from it.
                continue
            assert len(outputs) == 1
            assert len(outputs[0]) == HIDDEN_SIZE


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="embedding",
    feature="pooling",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_e5_mistral_non_string_inputs():
    """Non-string inputs must be rejected, never silently embedded.

    Measured on vLLM 0.26 + Ascend (offline LLM.embed): a scalar or a mixed
    prompt list fails while the prompt is tokenized (TypeError: object of type
    'int' has no len()), while bytes / dict inputs are mistaken for prompt
    embeddings and rejected because --enable-prompt-embeds is not set. Both
    are input-validation errors, so the assertion only pins the error class,
    not the wording.
    """
    with VllmRunner(
        _model_path(),
        runner="pooling",
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        pooler_config={"pooling_type": "LAST"},  # v0.26+: embed head L2-normalizes by default
    ) as runner:
        rejected: tuple[type[BaseException], ...] = (TypeError,) + EXPECTED_VALIDATION_ERRORS
        for bad_input in (123, None, b"hello", {"input": "hello"}, ["valid text", 123]):
            with pytest.raises(rejected):
                runner.embed([bad_input])
