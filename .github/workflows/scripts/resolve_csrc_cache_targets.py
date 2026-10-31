#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import argparse
import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).with_name("csrc_cache_targets.json")


def load_targets(config_path: Path = _CONFIG_PATH) -> dict[str, dict]:
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _extract_image_tag(image: str) -> str:
    image_name, separator, image_tag = image.rpartition(":")
    if not separator or not image_name or not image_tag or "/" in image_tag or "@" in image_name:
        raise ValueError(f"csrc cache image must use a tag: {image!r}")
    return image_tag


def resolve_targets(target_ids: list[str] | None, config_path: Path = _CONFIG_PATH) -> list[dict]:
    configured = load_targets(config_path)
    selected_ids = list(configured) if target_ids is None else target_ids
    targets = []
    for target_id in selected_ids:
        target = {"id": target_id, **configured[target_id]}
        target["image_tag"] = _extract_image_tag(target["image"])
        targets.append(target)
    return targets


def resolve_nightly_targets(image: str, soc_version: str, config_path: Path = _CONFIG_PATH) -> list[dict]:
    configured = list(load_targets(config_path).values())
    matching = [
        target["soc_version"]
        for target in configured
        if soc_version.lower() in (target["device"].lower(), target["soc_version"].lower())
    ]
    if not matching:
        raise ValueError(f"Unknown nightly SOC: {soc_version!r}")
    soc = matching[0]
    targets = []
    for architecture in dict.fromkeys(target["cache_arch"] for target in configured if target["soc_version"] == soc):
        template = next(target for target in configured if target["cache_arch"] == architecture)
        targets.append(
            {
                "id": f"nightly-{soc}-{architecture}",
                "runner": template["runner"],
                "cache_arch": architecture,
                "image": image,
                "image_tag": "nightly",
                "soc_version": soc,
                "os": "ubuntu",
                "nightly": True,
            }
        )
    return targets


def write_outputs(targets: list[dict]) -> None:
    outputs = {
        "targets": json.dumps(targets, separators=(",", ":")),
        "has_targets": str(bool(targets)).lower(),
    }
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            for name, value in outputs.items():
                output.write(f"{name}={value}\n")
    else:
        for name, value in outputs.items():
            print(f"{name}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve csrc cache target IDs.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Resolve every configured target.")
    selection.add_argument("--target-ids", help="JSON array of target IDs.")
    parser.add_argument("--nightly-image", default="")
    parser.add_argument("--soc-version", default="")
    args = parser.parse_args()

    target_ids = None if args.all else json.loads(args.target_ids)
    if args.nightly_image:
        if target_ids:
            parser.error("Nightly image cannot be combined with configured target IDs")
        write_outputs(resolve_nightly_targets(args.nightly_image, args.soc_version))
    else:
        write_outputs(resolve_targets(target_ids))


if __name__ == "__main__":
    main()
