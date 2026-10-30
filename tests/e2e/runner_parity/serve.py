# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Launch one arm with the shared, explicit full-model configuration."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("v1", "v2"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/glm51_a5_tp8.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    env = dict(os.environ, **config["env"])
    env.update(config["runner_env"][args.runner])
    command = ["vllm", "serve", args.model]
    for key, value in config["args"].items():
        if isinstance(value, bool):
            command.append("--" + ("" if value else "no-") + key)
        else:
            command.extend(["--" + key, json.dumps(value) if isinstance(value, dict) else str(value)])
    print(json.dumps({"command": command, "runner": args.runner, "frozen_stack": config["frozen_stack"]}), flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
