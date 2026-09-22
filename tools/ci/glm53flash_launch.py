# SPDX-License-Identifier: Apache-2.0
"""Shared effective settings and dry-run-first full-model service launcher."""

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEPLOYMENT = Path(__file__).with_name("glm53flash_deployment.json")


def effective_settings(profile):
    spec = json.loads(DEPLOYMENT.read_text())
    result = copy.deepcopy(spec["community"])
    overrides = spec["profiles"][profile]
    for name, value in overrides.items():
        if value is None:
            result.pop(name, None)
        else:
            result[name] = value
    changes = {
        name: {"community": spec["community"].get(name), "effective": result.get(name)}
        for name in overrides
        if result.get(name) != spec["community"].get(name)
    }
    return result, changes, spec["environment"]


def serve_command(model, settings, port):
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "glm",
        "--api-server-count",
        "1",
        "--trust-remote-code",
    ]
    for name, value in settings.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            command.extend((flag, json.dumps(value) if isinstance(value, (dict, list)) else str(value)))
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=("text_tp4", "nightly_tp16"), default="nightly_tp16")
    parser.add_argument("--port", type=int, default=18053)
    parser.add_argument("--mtp-off", action="store_true", help="Paired full-model performance control")
    parser.add_argument("--execute", action="store_true", help="Default prints command only")
    args = parser.parse_args()
    settings, changes, environment = effective_settings(args.profile)
    if args.mtp_off:
        changes["speculative_config"] = {"community": settings.pop("speculative_config", None), "effective": None}
    command = serve_command(args.model, settings, args.port)
    print(json.dumps({"command": command, "environment": environment, "changes": changes}, indent=2))
    print(shlex.join(command))
    if args.execute:
        raise SystemExit(subprocess.call(command, env={**os.environ, **environment}))
