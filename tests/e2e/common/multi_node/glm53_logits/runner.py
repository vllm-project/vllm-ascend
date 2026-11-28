# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Raw-logit checks using main's external-DP vllm serve launcher."""

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import aiohttp
import yaml

from .gate import GOLDEN, check_capture, load_golden, validate_features, validate_model


async def call(session, url, data=None):
    method = session.get if data is None else session.post
    async with method(url, **({"json": data} if data is not None else {})) as response:
        response.raise_for_status()
        return await response.json()


async def capture_and_compare(session, urls, case, manifest, corpus, output, artifact_root=GOLDEN):
    # The same immutable trace is used on every candidate; generated output IDs
    # are alignment checks, not the numerical oracle. Compare pre-processor scores.
    metrics = []
    entries = {(e["sample"], e["role"]): e for e in manifest["cases"][case]}
    for sample_index, item in enumerate(corpus):
        decoders = urls["D"] if isinstance(urls["D"], list) else [urls["D"]]
        selected = {"P": urls["P"], "D": decoders[sample_index % len(decoders)]}
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
        p_result = await call(session, selected["P"] + "/capture", prefill)
        transfer = p_result.get("kv_transfer_params")
        if not transfer or not transfer.get("do_remote_prefill") or not transfer.get("remote_block_ids"):
            raise RuntimeError("Missing P-to-D KV handoff; this is not a PD test")
        d_result = await call(
            session, selected["D"] + "/capture", dict(common, id=request_id + "-d", kv_transfer_params=transfer)
        )
        for role, result in [("P", p_result), ("D", d_result)]:
            if result["logits_mode"] != "raw_logits":
                raise ValueError("Expected raw logits")
            path = output / f"{item['name']}-{role}.npz"
            async with session.get(selected[role] + "/artifacts/" + Path(result["file"]).name) as response:
                response.raise_for_status()
                path.write_bytes(await response.read())
            path.with_suffix(".json").write_text(json.dumps(result, indent=2))
            metrics.append(check_capture(path, entries[item["name"], role], manifest, root=artifact_root))
    return metrics


async def collect_health(session, urls, processes, timeout):
    endpoints = {"P": [urls["P"]], "D": urls["D"]}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p, _, _ in processes):
            raise RuntimeError("Local vllm serve process exited; inspect rank logs")
        try:
            health = {}
            for role, addresses in endpoints.items():
                features = []
                for address in addresses:
                    data = await asyncio.wait_for(call(session, address + "/health"), 30)
                    features.extend(data["features"])
                health[role] = {"features": features}
            return health
        except (TimeoutError, aiohttp.ClientError):
            await asyncio.sleep(2)
    raise TimeoutError("Timed out waiting for worker feature probes")


async def wait_completion(session, addresses, processes, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p, _, _ in processes):
            raise RuntimeError("Decode vllm serve process exited")
        outcomes = [await call(session, url + "/completion") for url in addresses]
        if all("passed" in result for result in outcomes):
            if not all(result["passed"] is True for result in outcomes):
                raise RuntimeError("Leader reported a logits regression")
            return outcomes[0]
        await asyncio.sleep(2)
    raise TimeoutError("Leader did not deliver completion to every decode rank")


async def run(config_path):
    # Lazy import keeps numerical/protocol tests independent of launcher packages.
    from tests.e2e.common.multi_node.external_dp.external_dp_config import (
        ExternalDPConfigLoader,
        RankResolver,
        resolve_current_node_index,
    )
    from tests.e2e.common.multi_node.external_dp.runtime import (
        ExternalDPProxyLauncher,
        ExternalDPServerManager,
        build_all_server_commands,
        wait_ranks_ready,
    )

    raw = yaml.safe_load(Path(config_path).read_text())
    case = raw["logits_regression"]["case"]
    artifact_root = Path(raw["logits_regression"]["baseline_path"])
    manifest, corpus = load_golden(case, artifact_root=artifact_root)
    validate_model(Path(raw["model"]), manifest)
    config = ExternalDPConfigLoader.from_yaml(str(config_path))
    ranks = RankResolver(config).resolve()
    index = resolve_current_node_index(config)
    if config.num_nodes != 2 or index not in (0, 1):
        raise ValueError("GLM-5.3 logits requires exactly two nodes")
    role = ("P", "D")[index]
    timeout = int(raw["logits_regression"].get("timeout_seconds", 3600))
    output = Path(os.environ.get("LOG_PREFIX", "./logits_results")) / "glm53_logits" / case / role / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    commands = build_all_server_commands(config, ranks)
    (output / "commands.json").write_text(json.dumps([c.display_cmd for c in commands], indent=2))
    prefills = [r for r in ranks if r.role == "prefiller"]
    decodes = [r for r in ranks if r.role == "decoder"]
    if len(prefills) != 1 or len(decodes) != 4:
        raise ValueError("Expected one prefill API and four external decode DP APIs")
    urls = {
        "P": f"http://{prefills[0].host}:{prefills[0].port}/glm53_logits",
        "D": [f"http://{r.host}:{r.port}/glm53_logits" for r in decodes],
    }
    report = dict(case=case, role=role, baseline_id=manifest["baseline_id"], passed=False)
    try:
        with ExternalDPServerManager(
            config=config,
            ranks=ranks,
            current_node_index=index,
            log_root=output,
            extra_envs={"LOG_PREFIX": str(output)},
        ) as manager:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                if role == "D":
                    report.update(await wait_completion(session, urls["D"], manager.rank_processes, 2 * timeout))
                    report["role"] = role
                else:
                    try:
                        wait_ranks_ready(ranks, timeout=timeout, rank_processes=manager.rank_processes)
                        with ExternalDPProxyLauncher(
                            config=config, ranks=ranks, current_node_index=index, log_root=output
                        ) as proxy:
                            proxy.wait_ready()
                            health = await collect_health(session, urls, manager.rank_processes, timeout)
                            for r, data in health.items():
                                validate_features(data, case, r)
                            report["health"] = health
                            report["metrics"] = await capture_and_compare(
                                session, urls, case, manifest, corpus, output, artifact_root
                            )
                            report["passed"] = all(row["passed"] for row in report["metrics"])
                    except Exception as error:
                        report["error"] = str(error)
                        raise
                    finally:
                        for address in urls["D"]:
                            try:
                                await asyncio.wait_for(call(session, address + "/finish", report), timeout=10)
                            except (TimeoutError, aiohttp.ClientError) as error:
                                report.setdefault("finish_errors", []).append(str(error))
                                report["passed"] = False
                    if not report["passed"]:
                        raise AssertionError(f"{case}: logits regression failed; see {output / 'report.json'}")
    except Exception as error:
        report.update(passed=False, error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"case": case, "role": role, "passed": True, "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.config))
