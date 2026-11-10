# SPDX-License-Identifier: Apache-2.0

import base64
import importlib.util
import io

import huggingface_hub
import pytest
import requests
from modelscope import snapshot_download  # type: ignore[import-untyped]
from PIL import Image
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm.model_executor.models.clef_flash") is None,
    reason="CLEF Flash needs upstream joint-schema model support",
)

MODEL = "Cloudflare/clef-flash"


@pytest.fixture(scope="module")
def clef_server():
    host = "127.0.0.1"
    port = get_open_port()
    checkpoint = snapshot_download(MODEL, local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE)
    args = ["--host", host, "--port", str(port), "--max-model-len", "4096", "--served-model-name", MODEL]
    with RemoteOpenAIServer(checkpoint, args, server_host=host, server_port=port, auto_port=False) as server:
        yield server


@pytest.mark.parametrize("image_color", [None, "red", "blue"], ids=["text", "red-image", "blue-image"])
def test_clef_flash_joint_schema(clef_server, image_color):
    request = {
        "model": MODEL,
        "state": "A customer reports a damaged delivery and requests a replacement.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle the request?",
                "criteria": {"returns": "Refunds and replacements", "billing": "Payments and invoices"},
            },
            "replacement": {"type": "noul", "instructions": "Is a replacement requested?"},
        },
    }
    expected = {"route": "returns"}
    if image_color:
        buffer = io.BytesIO()
        Image.new("RGB", (128, 128), image_color).save(buffer, format="PNG")
        request["state"] = "Inspect the image and determine its predominant color."
        request["questions"] = {
            "color": {
                "type": "choice",
                "instructions": "What is the predominant color in the image?",
                "criteria": {"red": "Red", "blue": "Blue"},
            }
        }
        request["images"] = ["data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")]
        expected = {"color": image_color}

    response = requests.post(clef_server.url_for("v1", "systemone"), json=request, timeout=600)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "structured_decision"
    assert payload["usage"]["output_tokens"] == 0
    assert set(payload["answers"]) == set(request["questions"])
    for key, answer in payload["answers"].items():
        probabilities = answer["probabilities"]
        options = {"yes", "no"} if answer["type"] == "noul" else set(request["questions"][key]["criteria"])
        assert set(probabilities) == options
        assert all(0 <= p <= 1 for p in probabilities.values())
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    for key, choice in expected.items():
        assert payload["answers"][key]["choice"] == choice
