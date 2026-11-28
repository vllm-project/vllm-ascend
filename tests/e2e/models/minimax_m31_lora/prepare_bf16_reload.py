# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Create a separate, measurably changed BF16 checkpoint for reload probes."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--control", required=True, help="One unwrapped checkpoint parameter, e.g. the first layer norm"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    targets = json.loads(args.targets.read_text())
    names = {target["logical"].removeprefix("model.") + ".weight" for target in targets}
    names.add(args.control)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    changed = []
    for source in args.model.iterdir():
        if source.is_file() and source.suffix != ".safetensors":
            shutil.copy2(source, args.output / source.name)
    for source in sorted(args.model.glob("*.safetensors")):
        with safe_open(source, framework="pt", device="cpu") as data:
            tensors = {}
            for name in data.keys():  # noqa: SIM118 -- safe_open is not iterable.
                value = data.get_tensor(name)
                if name in names:
                    assert value.dtype == torch.bfloat16, (name, value.dtype)
                    value = value * 0.5
                    changed.append(name)
                tensors[name] = value
            save_file(tensors, str(args.output / source.name), metadata=data.metadata())
        del tensors
    assert set(changed) == names, sorted(names - set(changed))
    (args.output / "reload-probe.json").write_text(json.dumps({"scale": 0.5, "changed": changed}, indent=2) + "\n")


if __name__ == "__main__":
    main()
