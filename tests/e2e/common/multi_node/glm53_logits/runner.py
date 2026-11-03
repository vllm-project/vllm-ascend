# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Two-pod nightly adapter for the measured GLM-5.3 logits cases."""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import aiohttp
import yaml

from tests.e2e.common.multi_node.utils import get_net_interface, resolve_cluster_ips, resolve_current_node_index

from .config import build
from .gate import GOLDEN, check_capture, load_golden, validate_features, validate_model

SERVER_MODULE = "tests.e2e.common.multi_node.glm53_logits.logits_server"
PORTS = {"P": 18701, "D": 18702}


def server_env(role, case, ip):
    env = dict(os.environ)
    interface = get_net_interface(ip)
    env.update(
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_HOST_IP=ip,
        HCCL_IF_IP=ip,
        HCCL_SOCKET_IFNAME=interface,
        GLOO_SOCKET_IFNAME=interface,
        TP_SOCKET_IFNAME=interface,
        HCCL_CONNECT_TIMEOUT="180",
        HCCL_EXEC_TIMEOUT="180",
        HCCL_BUFFSIZE="1024",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="1",
        ASCEND_LOCAL_COMM_RES='{"version":"1.3"}',
        ASCEND_GLOBAL_RESOURCE_CONFIG='{"HCCL":{"device_mode":"HCCS"}}',
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        TASK_QUEUE_ENABLE="1",
        VLLM_NO_USAGE_STATS="1",
    )
    env.pop("VLLM_PP_LAYER_PARTITION", None)
    if role == "P" and case != "disable_pp":
        env["VLLM_PP_LAYER_PARTITION"] = "4,4"
    return env


async def call(session, url, data=None):
    method = session.get if data is None else session.post
    async with method(url, **({"json": data} if data is not None else {})) as response:
        response.raise_for_status()
        return await response.json()


async def wait_ready(session, urls, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Local logits server exited; inspect server.log")
        try:
            health = {}
            for role, url in urls.items():
                health[role] = await asyncio.wait_for(call(session, url + "/health"), timeout=5)
            return health
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(2)
    raise TimeoutError("Timed out waiting for both logits servers")


async def capture_and_compare(session, urls, case, manifest, corpus, output, artifact_root=GOLDEN):
    # The same immutable trace is used on every candidate; generated output IDs
    # are alignment checks, not the numerical oracle. Compare pre-processor scores.
    metrics = []
    entries = {(e["sample"], e["role"]): e for e in manifest["cases"][case]}
    for item in corpus:
        request_id = f"{case}-{item['name']}-{uuid.uuid4().hex}"
        common = dict(prompt_ids=item["prompt_ids"], trace=item["trace"], max_tokens=8)
        prefill = dict(
            common,
            id=request_id + "-p",
            trace=item["trace"][:1],
            max_tokens=1,
            kv_transfer_params=dict(
                do_remote_decode=True,
                do_remote_prefill=False,
                remote_engine_id=None,
                remote_block_ids=None,
                remote_host=None,
                remote_port=None,
            ),
        )
        p_result = await call(session, urls["P"] + "/capture", prefill)
        transfer = p_result.get("kv_transfer_params")
        if not transfer or not transfer.get("do_remote_prefill") or not transfer.get("remote_block_ids"):
            raise RuntimeError("Missing P-to-D KV handoff; this is not a PD test")
        d_result = await call(
            session, urls["D"] + "/capture", dict(common, id=request_id + "-d", kv_transfer_params=transfer)
        )
        for role, result in [("P", p_result), ("D", d_result)]:
            if result["logits_mode"] != "raw_logits":
                raise ValueError("Expected raw logits")
            path = output / f"{item['name']}-{role}.npz"
            async with session.get(urls[role] + "/artifacts/" + Path(result["file"]).name) as response:
                response.raise_for_status()
                path.write_bytes(await response.read())
            path.with_suffix(".json").write_text(json.dumps(result, indent=2))
            metrics.append(check_capture(path, entries[item["name"], role], manifest, root=artifact_root))
    return metrics


def stop_process_group(process):
    # Each server has its own session. Only terminate processes created here.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


async def run(raw):
    case = raw["logits_regression"]["case"]
    artifact_root = Path(raw["logits_regression"]["baseline_path"])
    manifest, corpus = load_golden(case, artifact_root=artifact_root)
    model = Path(raw["model"])
    validate_model(model, manifest)
    ips = resolve_cluster_ips(raw, 2)
    index = resolve_current_node_index(ips)
    if index not in (0, 1):
        raise ValueError("GLM-5.3 logits requires exactly two nodes")
    role = ("P", "D")[index]
    timeout = int(raw["logits_regression"].get("timeout_seconds", 1800))
    output = Path(os.environ.get("LOG_PREFIX", "./logits_results")) / "glm53_logits" / case / role / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    config = build(case, role, model)
    config_path = output / "engine.json"
    config_path.write_text(json.dumps(config, indent=2))
    urls = {r: f"http://{ip}:{PORTS[r]}" for r, ip in zip(("P", "D"), ips)}
    report = dict(case=case, role=role, baseline_id=manifest["baseline_id"], passed=False)
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                SERVER_MODULE,
                "--config",
                str(config_path),
                "--port",
                str(PORTS[role]),
                "--output-dir",
                str(output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=server_env(role, case, ips[index]),
        )
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                if role == "D":
                    # Leader sends the final outcome only after all artifacts were fetched.
                    deadline = time.monotonic() + 2 * timeout
                    while process.poll() is None and time.monotonic() < deadline:
                        await asyncio.sleep(2)
                    if process.poll() != 0:
                        raise RuntimeError("Decode server failed or leader timed out; inspect server.log")
                    report.update(json.loads((output / "completion.json").read_text()))
                    if report.get("passed") is not True:
                        raise RuntimeError("Leader reported a regression")
                else:
                    try:
                        health = await wait_ready(session, urls, process, timeout)
                        for r, data in health.items():
                            validate_features(data, case, r)
                            if data["config"] != build(case, r, model):
                                raise ValueError(f"Unexpected server configuration for {r}")
                        report["health"] = health
                        report["metrics"] = await capture_and_compare(
                            session, urls, case, manifest, corpus, output, artifact_root
                        )
                        report["passed"] = all(row["passed"] for row in report["metrics"])
                    except Exception as error:
                        report["error"] = str(error)
                        raise
                    finally:
                        try:
                            await asyncio.wait_for(call(session, urls["D"] + "/finish", report), timeout=10)
                        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                            report["finish_error"] = str(error)
                            report["passed"] = False
                    if not report["passed"]:
                        raise AssertionError(f"{case}: logits regression failed; see {output / 'report.json'}")
        finally:
            (output / "report.json").write_text(json.dumps(report, indent=2))
            stop_process_group(process)
    print(json.dumps({"case": case, "role": role, "passed": True, "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    asyncio.run(run(yaml.safe_load(args.config.read_text())))
