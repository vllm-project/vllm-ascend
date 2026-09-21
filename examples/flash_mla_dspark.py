# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference launcher for the external FlashMLA MRV2 candidate.

See docs/flash_mla_mrv2.md. --dry-run imports no accelerator packages and
starts no service. Actual launches require a prepared A5 NPU environment.
"""

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import shlex
import sys
from pathlib import Path


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--num-speculative-tokens", type=positive_int)
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--tp", type=positive_int, default=4)
    parser.add_argument("--draft-tp", type=positive_int, default=4)
    parser.add_argument("--max-model-len", type=positive_int, default=4096)
    parser.add_argument("--max-num-seqs", type=positive_int, default=4)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8000)
    parser.add_argument("--served-model-name", default="flashmla-dspark")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "serve_args", nargs=argparse.REMAINDER, help="checkpoint-specific vLLM arguments after -- (e.g. quantization)"
    )
    args = parser.parse_args()
    if not args.target_only and (not args.draft_model or args.num_speculative_tokens is None):
        parser.error("DSpark requires --draft-model and the checkpoint's --num-speculative-tokens")
    if args.target_only and (args.draft_model or args.num_speculative_tokens is not None):
        parser.error("omit draft arguments for --target-only")
    if args.port > 65535:
        parser.error("--port must be at most 65535")
    return args


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.target_model,
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        str(args.tp),
        "--pipeline-parallel-size",
        "1",
        "--decode-context-parallel-size",
        "1",
        "--prefill-context-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.mode == "eager":
        command.append("--enforce-eager")
    else:
        command += ["--compilation-config", json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"})]
    if not args.target_only:
        command += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "dspark",
                    "model": args.draft_model,
                    "num_speculative_tokens": args.num_speculative_tokens,
                    "draft_tensor_parallel_size": args.draft_tp,
                    "enforce_eager": args.mode == "eager",
                }
            ),
        ]
    extra = args.serve_args[1:] if args.serve_args[:1] == ["--"] else args.serve_args
    # Keep the selected eager/graph, dtype and topology contract unambiguous.
    reserved = {value for value in command if value.startswith("--")}
    reserved.update(
        {
            "--enforce-eager",
            "--no-enforce-eager",
            "--compilation-config",
            "--speculative-config",
            "-tp",
            "-pp",
            "-dcp",
            "-pcp",
            "-O",
            "-cc",
            "-sc",
            "--spec-method",
            "--num-speculative-tokens",
        }
    )
    if any(value.split("=", 1)[0].split(".", 1)[0].replace("_", "-") in reserved for value in extra):
        raise ValueError("extra serve arguments must not override the launcher's mode, topology or core settings")
    return command + extra


def preflight() -> None:
    """Inspect the installed Python entry points and Meta support, without NPU tensor execution."""
    import torch
    import torch_npu  # noqa: F401

    for name in ("vllm", "vllm_ascend", "cann_ops_transformer.ops"):
        module = importlib.import_module(name)
        print(f"{name}: {module.__file__}", flush=True)
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
        try:
            print(f"{name} version: {importlib.metadata.version(name)}", flush=True)
        except importlib.metadata.PackageNotFoundError:
            print(f"{name} version: unavailable; inspect the source checkout", flush=True)
    for distribution in importlib.metadata.packages_distributions().get("cann_ops_transformer", []):
        print(f"operator distribution: {distribution} {importlib.metadata.version(distribution)}", flush=True)

    ops = importlib.import_module("cann_ops_transformer.ops")
    for name in ("flash_mla_with_kvcache_metadata", "flash_mla_with_kvcache"):
        function = getattr(ops, name)
        if not callable(function):
            raise TypeError(f"cann_ops_transformer.ops.{name} is not callable")
        try:
            source = inspect.getfile(function)
        except TypeError:
            source = "native binding; inspect the installed package schema"
        print(f"external entry: {name}; module={getattr(function, '__module__', None)}; source={source}", flush=True)

    # H=8 here probes Meta registration only; model heads are never changed.
    schedule = ops.flash_mla_with_kvcache_metadata(
        torch.empty(2, dtype=torch.int32, device="meta"),
        8,
        1,
        cu_seqlens_q=torch.empty(3, dtype=torch.int32, device="meta"),
        seqused_q=torch.empty(2, dtype=torch.int32, device="meta"),
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        head_dim_qk=576,
        head_dim_v=512,
        mask_mode=0,
        layout_q="TND",
    )
    if not isinstance(schedule, torch.Tensor) or schedule.device.type != "meta":
        raise TypeError("external metadata operator must return a Meta tensor for buffer sizing")
    print(f"external Meta schedule: shape={tuple(schedule.shape)}, dtype={schedule.dtype}", flush=True)
    print("Python entry/Meta check only: NPU dispatch, BBND strides and numerical correctness still need validation.")


def main() -> None:
    args = parse_args()
    try:
        command = build_command(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    environment = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_ASCEND_ENABLE_FLASH_MLA": "1",
        "ASCEND_RT_VISIBLE_DEVICES": args.devices,
    }
    os.environ.update(environment)
    print(f"working directory: {Path.cwd()}")
    for name, value in environment.items():
        print(f"export {name}={shlex.quote(value)}")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    preflight()
    if not args.preflight_only:
        sys.stdout.flush()
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
