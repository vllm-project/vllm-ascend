# SPDX-License-Identifier: Apache-2.0
"""Loopback-only, fixed-image HTTP streaming smoke with synthetic reduced weights."""

import argparse
import base64
import hashlib
import io
import json
import os
import shlex
from pathlib import Path

from PIL import Image
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.ci.glm53flash_collect import LOADER
from tools.ci.glm53flash_launch import effective_settings, serve_command


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings, _, env = effective_settings("text_tp4")
    settings.update(
        load_format=LOADER,
        dtype="bfloat16",
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={"max_image_tokens": 256},
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]},
    )
    env.update(VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_USE_V2_MODEL_RUNNER="0")
    port = get_open_port()
    command = serve_command(args.model, settings, port)
    command[2] = "tools.ci.glm53flash_serve"
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    stream = io.BytesIO()
    Image.new("RGB", (56, 56), (80, 120, 200)).save(stream, format="PNG")
    raw = stream.getvalue()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]
    with RemoteOpenAIServer(
        args.model,
        shlex.join(command),
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        env_dict={**os.environ, **env},
        max_wait_seconds=480,
    ) as server:
        client = server.get_client()
        completion = client.chat.completions.create(
            model="glm", messages=messages, max_tokens=8, temperature=0, stream=True
        )
        chunks = [chunk.model_dump() for chunk in completion]
        assert chunks and any(c["choices"] and c["choices"][0].get("finish_reason") for c in chunks)
        (output / "result.json").write_text(
            json.dumps({"status": "PASS", "image_sha256": hashlib.sha256(raw).hexdigest(), "chunks": chunks}, indent=2)
        )
        print("FLASH_IMAGE_STREAM_PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
