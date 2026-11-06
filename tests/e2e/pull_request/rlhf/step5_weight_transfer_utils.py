# SPDX-License-Identifier: Apache-2.0
"""Step5 startup-vs-live oracle, adapted from PR #18634 (not its model matrix).

The reference reads the original checkpoint, independently of the transport
source. A small, complete BF16 fixture is required; no silent layer slicing.
"""

import base64
import gc
import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import psutil
import pytest
import requests
import torch
from PIL import Image
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    WeightUpdateModelCase,
    assert_weight_update_matches_reference,
    register_engines_once,
)
from tests.step5_weight_source import Step5CheckpointSource, is_vision_weight

MODEL = "step5-weight-transfer-test"
TIMEOUT = 600
MEMORY_UTILIZATION = 0.45


def checkpoint(value, multimodal):
    path = Path(value).resolve()
    config = json.loads((path / "config.json").read_text())
    arch = "Step5ForConditionalGeneration" if multimodal else "Step5ForCausalLM"
    assert arch in config["architectures"], f"Expected {arch}, got {config['architectures']}"
    text = config.get("text_config", config)
    assert not config.get("quantization_config") and not text.get("quantization_config"), "BF16 fixture required"
    metadata = Step5CheckpointSource(path).metadata()
    names = [m.name for m in metadata]
    for fragment in (".moe.router_bias", ".moe.gate.weight", ".share_expert.", ".moe.gate_proj.weight"):
        assert any(fragment in n for n in names), f"Fixture does not exercise {fragment}"
    assert any(n.endswith("lm_head.weight") for n in names)
    assert any(m.dtype == torch.bfloat16 for m in metadata)
    assert all(m.dtype in (torch.bfloat16, torch.float32, torch.int64, torch.int32, torch.bool) for m in metadata)
    if multimodal:
        assert any("vision_model." in n for n in names)
        assert any("vit_large_projector." in n for n in names)
    return path


def post(server, route, *, payload=None, params=None):
    response = requests.post(server.url_for(route), json=payload, params=params, timeout=TIMEOUT)
    response.raise_for_status()


class BackgroundPost(threading.Thread):
    def __init__(self, server, route, payload):
        super().__init__(daemon=True)
        self.server, self.route, self.payload = server, route, payload
        self.error = None

    def run(self):
        try:
            post(self.server, self.route, payload=self.payload)
        except Exception as error:
            self.error = error

    def finish(self):
        self.join(TIMEOUT + 5)
        assert not self.is_alive(), f"/{self.route} timed out"
        if self.error is not None:
            raise RuntimeError(f"/{self.route} failed") from self.error


@contextmanager
def server_for(path, backend, monkeypatch, multimodal):
    # Both servers use identical allocations, graph mode and processors.
    deadline = time.monotonic() + TIMEOUT
    while True:
        free, total = torch.npu.mem_get_info(0)
        if free >= total * MEMORY_UTILIZATION + 2 * 2**30:
            break
        assert time.monotonic() < deadline, "Previous server did not release HBM"
        time.sleep(2)
    port = get_open_port()
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        MODEL,
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--enable-sleep-mode",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        str(MEMORY_UTILIZATION),
        "--distributed-executor-backend",
        "mp",
        "--tensor-parallel-size",
        "1",
        "--no-enable-prefix-caching",
        "--additional-config",
        '{"weight_nz_mode": 0}',
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}',
    ]
    if multimodal:
        args += ["--mm-processor-cache-gb", "0"]
    if backend:
        args += ["--load-format", "dummy", "--weight-transfer-config", json.dumps({"backend": backend})]
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with RemoteOpenAIServer(
        str(path),
        vllm_serve_args=args,
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        env_dict={
            "VLLM_SERVER_DEV_MODE": "1",
            "ASCEND_RT_VISIBLE_DEVICES": "0",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        },
    ) as server:
        yield server


def signature(server, multimodal):
    client = server.get_client().with_options(timeout=TIMEOUT, max_retries=0)
    result = []
    for prompt in (
        ("Describe what you see.", "What are the main colors and shapes?")
        if multimodal
        else (
            "The capital of France is",
            "Explain why the sky is blue:",
        )
    ):
        if multimodal:
            image = Image.new("RGB", (224, 224), "red")
            image.paste("blue", (112, 0, 224, 224))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}],
                    }
                ],
                temperature=0,
                seed=0,
                max_tokens=8,
                logprobs=True,
                top_logprobs=1,
            )
            choice = response.choices[0]
            assert choice.logprobs and choice.logprobs.content
            tokens = tuple((t.token, tuple(t.bytes or [])) for t in choice.logprobs.content)
            probs = tuple(t.logprob for t in choice.logprobs.content)
        else:
            response = client.completions.create(
                model=MODEL, prompt=prompt, temperature=0, seed=0, max_tokens=8, logprobs=1
            )
            choice = response.choices[0]
            assert choice.logprobs and choice.logprobs.token_logprobs
            tokens = tuple(choice.logprobs.tokens)
            probs = tuple(choice.logprobs.token_logprobs)
        assert tokens and len(tokens) == len(probs)
        assert all(p is not None and math.isfinite(p) for p in probs)
        result.append((tokens, probs))
    return result


def run_oracle(request, backend, multimodal=False):
    option = "--step5-vl-checkpoint" if multimodal else "--step5-text-checkpoint"
    value = request.config.getoption(option)
    if not value:
        pytest.skip(f"Provide {option} with a complete single-card Step5 BF16 fixture")
    path = checkpoint(value, multimodal)
    assert not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), (
        "Run in an isolated allocated container with identity device numbering (unset ASCEND_RT_VISIBLE_DEVICES)"
    )
    required = 1 if backend == "npu_ipc" else 2
    if torch.npu.device_count() < required:
        pytest.skip(f"Requires {required} allocated NPUs")
    # Native HCCL calls can block even after the HTTP thread fails. Bound the
    # entire lane, and clean up only this test's newly created process group.
    command = [
        sys.executable,
        "-m",
        "tests.e2e.pull_request.rlhf.step5_weight_transfer_utils",
        str(path),
        backend,
        str(int(multimodal)),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        assert process.wait(timeout=3600) == 0, "Step5 oracle subprocess failed; see server/test logs"
    finally:
        # Some server executors create their own sessions. Retain their process
        # identities before terminating the lane, rather than matching by name.
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        for child in children:
            with suppress(psutil.NoSuchProcess):
                child.kill()
        psutil.wait_procs(children, timeout=30)
        process.wait(timeout=30)


def _run_lane(path, monkeypatch, backend, multimodal):
    device = 0 if backend == "npu_ipc" else 1
    torch.npu.set_device(device)
    with server_for(path, None, monkeypatch, multimodal) as reference:
        expected = signature(reference, multimodal)
        assert signature(reference, multimodal) == expected, "Reference is not deterministic"

    with server_for(path, backend, monkeypatch, multimodal) as live:
        dummy = signature(live, multimodal)
        assert dummy != expected, "Dummy signature cannot observe the payload"
        source = Step5CheckpointSource(path, f"npu:{device}")
        update = ipc_updater(live, source) if backend == "npu_ipc" else hccl_updater(live, source)
        updated = []
        for round_index in range(2):
            update(False)
            updated.append(signature(live, multimodal))
            assert updated[-1] == expected, f"Live update {round_index + 1} != startup(W)"
        assert_weight_update_matches_reference(
            dummy,
            expected,
            updated[0],
            updated[1],
            WeightUpdateModelCase(id="step5-vl" if multimodal else "step5-text", model=str(path), hf_overrides={}),
        )
        if multimodal:
            # Same language weights, zeroed vision/projector: a skipped visual
            # update or stale encoder cache must not pass unnoticed.
            update(True)
            assert signature(live, True) != expected, "Image probe is insensitive to visual weights"
            update(False)
            assert signature(live, True) == expected, "Visual weights did not recover"


def ipc_updater(server, source):
    from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
    from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
        NPUIPCTrainerInitInfo,
    )

    class Payload(WeightSource):
        zero_vision = False

        def metadata(self):
            return [ParamMeta(m.name, m.dtype, m.shape) for m in source.metadata()]

        def __iter__(self):
            for name, value in source:
                yield name, torch.zeros_like(value) if self.zero_vision and is_vision_weight(name) else value

    payload = Payload()
    register_engines_once()
    engine = WeightTransferTrainerFactory.trainer_init(
        NPUIPCTrainerInitInfo(rank=0, packed=False),
        client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
        source=payload,
    )

    def update(zero_vision):
        payload.zero_vision = zero_vision
        post(server, "pause")
        post(server, "sleep", params={"level": 2})
        post(server, "wake_up", params={"tags": ["weights"]})
        engine.send_weights()
        gc.collect()
        torch.npu.empty_cache()
        post(server, "wake_up", params={"tags": ["kv_cache"]})

    return update


def hccl_updater(server, source):
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )

    init = dict(master_address=get_ip(), master_port=get_open_port(), world_size=2)
    thread = BackgroundPost(server, "init_weight_transfer_engine", {"init_info": {**init, "rank_offset": 1}})
    thread.start()
    group = HCCLWeightTransferEngine.trainer_init(init)
    thread.finish()
    metadata = source.metadata()

    def update(zero_vision):
        assert not zero_vision
        post(server, "pause")
        post(server, "start_weight_update")
        thread = BackgroundPost(
            server,
            "update_weights",
            {
                "update_info": {
                    "names": [m.name for m in metadata],
                    "shapes": [list(m.shape) for m in metadata],
                    "dtype_names": [str(m.dtype).split(".")[-1] for m in metadata],
                    "packed": False,
                }
            },
        )
        thread.start()
        HCCLWeightTransferEngine.trainer_send_weights(
            iter(source), HCCLTrainerSendWeightsArgs(group=group, packed=False)
        )
        torch.npu.synchronize()
        thread.finish()
        post(server, "finish_weight_update")
        post(server, "reset_prefix_cache")
        post(server, "resume")

    return update


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        _run_lane(Path(sys.argv[1]), patch, sys.argv[2], bool(int(sys.argv[3])))
