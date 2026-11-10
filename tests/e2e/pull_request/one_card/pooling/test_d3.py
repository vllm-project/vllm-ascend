# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import json
from collections.abc import Iterator
from typing import Any

import huggingface_hub
import pytest
import requests
import torch
from modelscope import snapshot_download  # type: ignore[import-untyped]
from transformers import AutoModel, modeling_utils
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

# Probe for the feature instead of a version: native D3 (vLLM #61565, #61778) is in neither v0.31.0 nor v0.32.0.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm.model_executor.models.d3") is None,
    reason="D3 needs a vLLM revision with native D3 support",
)

D3_REPO = "vllm-sr/d3-edge"
# Allow for numerical differences between the Ascend and reference runtimes.
SOFTMAX_ATOL = 0.03
# These tests only send text, so media inputs are switched off. Keep every D3 run on one setting: the torch.compile
# cache key of vLLM ignores `limit_mm_per_prompt`, and a graph compiled with the other setting crashes graph-mode
# startup when it is reused.
NO_MEDIA = {"image": 0, "video": 0}
STATE = "A customer reports a damaged delivery and asks for a replacement."
QUESTIONS: dict[str, dict[str, Any]] = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle the request?",
        "criteria": {
            "returns": "Refunds and replacements",
            "billing": "Payments and invoices",
            "technical": "Product setup and faults",
        },
    },
    "receipt": {
        "type": "noul",
        "instructions": "Does the customer ask for a replacement?",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Routine", "Soon", "Today"],
    },
}


@contextlib.contextmanager
def _without_cpu_device_map() -> Iterator[None]:
    """Load the checkpoint's reference runtime on the CPU without `accelerate`.

    The runtime always loads its backbone with `device_map={"": <device>}`, and transformers needs `accelerate`
    for any `device_map`. On the CPU that map is only the default placement, so it is dropped there.
    """
    check_device_map = modeling_utils.check_and_set_device_map

    def without_cpu_map(device_map):
        if device_map == {"": "cpu"}:
            device_map = None
        return check_device_map(device_map)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modeling_utils, "check_and_set_device_map", without_cpu_map)
        yield


@pytest.fixture(scope="module")
def model_name():
    yield snapshot_download(D3_REPO, local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE)


@pytest.fixture(scope="module")
def d3_server(model_name):
    # Shared by both tests below: starting the engine is the expensive part of this
    # file, and both tests only need HTTP access to the same running server.
    host = "127.0.0.1"
    port = get_open_port()
    server_args = [
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        "4096",
        "--limit-mm-per-prompt",
        json.dumps(NO_MEDIA),
    ]
    with RemoteOpenAIServer(model_name, server_args, server_host=host, server_port=port, auto_port=False) as server:
        yield server


def test_d3_pooling_matches_checkpoint_runtime(model_name, d3_server):
    with _without_cpu_device_map():
        reference_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device="cpu")
    runtime = reference_model.runtime
    prepared = runtime.prepare(STATE, QUESTIONS)
    runnable_keys = prepared.runnable
    option_counts = [len(prepared.questions[key].keys) for key in runnable_keys]
    prompt_ids = [prepared.sequences[key] for key in runnable_keys]
    reference_logits = list(runtime.logits(prompt_ids, option_counts).cpu())
    del reference_model, runtime

    response = requests.post(
        d3_server.url_for("classify"),
        json={"model": model_name, "input": prompt_ids},
        timeout=120,
    )
    assert response.status_code == 200, response.text
    outputs = response.json()["data"]

    assert len(outputs) == len(option_counts)
    for reference, output, option_count in zip(reference_logits, outputs, option_counts):
        ascend_logits = torch.tensor(output["probs"])
        assert ascend_logits.shape == (255,)
        assert torch.isfinite(ascend_logits).all()
        torch.testing.assert_close(
            torch.softmax(ascend_logits[:option_count].float(), dim=-1),
            torch.softmax(reference[:option_count].float(), dim=-1),
            atol=SOFTMAX_ATOL,
            rtol=0,
        )


def test_d3_systemone_endpoint(model_name, d3_server):
    request = {
        "model": model_name,
        "state": STATE,
        "questions": QUESTIONS,
    }
    response = requests.post(
        d3_server.url_for("v1", "systemone"),
        json=request,
        timeout=600,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "structured_decision"
    assert set(payload["answers"]) == set(QUESTIONS)

    choice = payload["answers"]["route"]
    assert choice["type"] == "choice"
    assert choice["choice"] in QUESTIONS["route"]["criteria"]
    assert set(choice["probabilities"]) == set(QUESTIONS["route"]["criteria"])

    noul = payload["answers"]["receipt"]
    assert noul["type"] == "noul"
    assert set(noul["probabilities"]) == {"yes", "no"}
    assert noul["noul"] == pytest.approx(noul["probabilities"]["yes"])

    score = payload["answers"]["urgency"]
    assert score["type"] == "score"
    assert set(score["probabilities"]) == {"0", "1", "2"}

    for answer in payload["answers"].values():
        probabilities = list(answer["probabilities"].values())
        assert all(torch.isfinite(torch.tensor(probabilities)))
        assert all(probability >= 0 for probability in probabilities)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)
        assert 0 <= answer["confidence"] <= 1
