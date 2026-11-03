# SPDX-License-Identifier: Apache-2.0
"""Launch one arm of the saved A5 C8/MTP3 single-D comparison."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def validate_checkpoint(model: str, contract: dict) -> None:
    description = Path(model) / "quant_model_description.json"
    if not description.is_file():
        raise ValueError(f"The A5 reference requires a local MXFP checkpoint with {description.name}.")
    actual = json.loads(description.read_text())
    expected = {
        "model_quant_type": contract["model_quant_type"],
        "group_size": contract["group_size"],
        "model.layers.0.self_attn.q_a_proj.weight": contract["attention_projection_quant"],
    }
    mismatches = {key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"Checkpoint quantization differs from the A5 reference (actual, expected): {mismatches}")


def arguments_for_profile(config: dict, arm: str, profile: str) -> list[str]:
    argv = list(config[arm])
    if profile == "a5_c8_main":
        index = argv.index("--additional-config")
        additional = json.loads(argv[index + 1])
        # Current Ascend selects Indexer C8 layers from checkpoint metadata.
        additional.pop("sparse_li_c8_layer_policy", None)
        # Match the reviewed MXFP Prolog path's original final NZ operands.
        additional["weight_nz_mode"] = 1
        argv[index + 1] = json.dumps(additional)
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("on", "off"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8029)
    parser.add_argument("--host-ip", help="Actual data-network IP for this host, when required")
    parser.add_argument("--profile", choices=("a5_c8_main",), default="a5_c8_main")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("reference_configuration.json").read_text())
    argv = arguments_for_profile(config, args.arm, args.profile)
    for flag, value in [("--model", args.model), ("--host", args.host), ("--port", str(args.port))]:
        argv[argv.index(flag) + 1] = value
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *argv]
    if args.print_command:
        print(json.dumps(command, indent=2))
        return
    validate_checkpoint(args.model, config["checkpoint_contract"])
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ASCEND_HISPARSE_PROFILE"] = args.profile
    if args.host_ip:
        os.environ["VLLM_HOST_IP"] = args.host_ip
    from vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile import optimized_profile_enabled

    optimized_profile_enabled()
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
