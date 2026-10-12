#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run on the failing Linux/NPU host; does not modify the vLLM installation.

Each test uses a fresh process. The sole environment difference between A
and B is the selected repository vendor in ASCEND_CUSTOM_OPP_PATH.
Each probe tests a small single-sequence case, not the model's actual inputs.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="/data/w50063966/vllm-ascend")
    p.add_argument("--device", type=int, default=0, help="Logical NPU device index")
    p.add_argument("--timeout", type=int, default=300, help="Seconds per subprocess, including JIT")
    p.add_argument("--output", help="New report directory; default: unique directory in /tmp")
    p.add_argument(
        "--probe",
        choices=("decode-2d", "decode-3d", "prefill-2d", "prefill-3d"),
        default="prefill-2d",
        help="Which input contract to exercise",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16", help="Default: bf16")
    p.add_argument("--child", choices=("A_without_repo_opp", "B_with_repo_opp"), help=argparse.SUPPRESS)
    p.add_argument("--result-file", help=argparse.SUPPRESS)
    return p


def real(path):
    return str(Path(path).expanduser().resolve())


def vendor_path(args):
    return Path(args.repo).resolve() / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"


def normalized_opp_entries(value):
    return [real(entry) for entry in (value or "").split(os.pathsep) if entry]


def dump_libraries():
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6:
            path = fields[5]
            if any(s in path.lower() for s in ("libopapi", "libcust", "optiling", "op_proto", "causal_conv1d")):
                paths.add(path)
    print("LOADED_LIBRARIES (mapped libraries; not proof of the selected operator provider):")
    for path in sorted(paths):
        print(" ", path)
    return sorted(paths)


def dump_metadata(vendor, package_file):
    # Inspect only conventional Ascend 950 metadata directories, not the whole disk.
    providers = [vendor]
    providers.extend(Path(entry) for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if entry)
    opp_roots = []
    if os.environ.get("ASCEND_OPP_PATH"):
        opp_roots.append(Path(os.environ["ASCEND_OPP_PATH"]))
    for parent in Path(package_file).resolve().parents:
        if parent.name == "python":
            opp_roots.append(parent.parent / "opp")
            break
    for opp in opp_roots:
        providers.append(opp / "built-in")
        providers.extend(path for path in (opp / "vendors").glob("*") if path.is_dir())
    seen = set()
    records = []
    print("INSTALLED_METADATA (presence does not prove runtime selection):")
    for provider in providers:
        root = provider / "op_impl/ai_core/tbe/config"
        if not root.is_dir():
            continue
        for path in sorted(root.glob("ascend950*/*ops-info.json")):
            if real(path) in seen:
                continue
            seen.add(real(path))
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict):
                    continue
                for op_name in ("CausalConv1d", "VllmCausalConv1d"):
                    op = data.get(op_name)
                    if op is None:
                        continue
                    print("  FILE:", path)
                    print("  OP:", op_name)
                    print(json.dumps(op, ensure_ascii=False, indent=2))
                    outputs = {
                        key: value for key, value in op.items() if key.startswith("output") and key[6:].isdigit()
                    }
                    records.append({"provider": str(provider), "path": str(path), "op_name": op_name, "outputs": outputs})
            except (OSError, ValueError) as exc:
                print("  metadata read error:", path, str(exc))
    return records


def short_error(exc):
    lines = str(exc).splitlines()
    for line in lines:
        if "Cannot find any bin" in line:
            return line.strip()[:240]
    return next((line.strip()[:240] for line in lines if line.strip()), type(exc).__name__)


def child(args):
    result = {"case": args.child, "probe": args.probe, "dtype": args.dtype, "status": "FAIL", "stage": "import"}
    result["opp_before_import"] = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    expected_entries = normalized_opp_entries(result["opp_before_import"])
    try:
        print("PYTHON:", sys.executable)
        print("CASE:", args.child, args.probe, args.dtype)
        for name in (
            "ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "ASCEND_RT_VISIBLE_DEVICES"
        ):
            print(name + ":", os.environ.get(name, "<unset>"))
        from importlib import import_module

        import torch
        import torch_npu

        # Initialize the NPU backend before loading its operator registrations.
        import cann_ops_transformer
        import cann_ops_transformer.ops  # Register the same operator namespace used by Kimi.

        print("TORCH:", torch.__version__, "TORCH_NPU:", getattr(torch_npu, "__version__", "unknown"))
        is_prefill = args.probe.startswith("prefill")
        operator_name = "causal_conv1d_fn" if is_prefill else "causal_conv1d_update"
        operator = getattr(torch.ops.cann_ops_transformer, operator_name)
        try:
            wrapper = import_module(f"cann_ops_transformer.ops.mamba.causal_conv1d.{operator_name}")
            print("WRAPPER:", wrapper.__file__)
            result["wrapper"] = wrapper.__file__
        except Exception as exc:
            print("Could not inspect wrapper path:", exc)
        try:
            result["schema"] = str(operator.default._schema)
            print("SCHEMA:", result["schema"])
        except Exception as exc:
            print("Could not inspect schema:", exc)
        result["opp_before_call"] = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        print("OPP_AFTER_IMPORT:", result["opp_before_call"])
        result["stage"] = "environment_control"
        current_entries = normalized_opp_entries(result["opp_before_call"])
        contains_vendor = real(vendor_path(args)) in current_entries
        if current_entries != expected_entries or contains_vendor != args.child.startswith("B_"):
            raise RuntimeError("An import changed the tested OPP setting; this A/B case is not isolated")
        try:
            result["registrations"] = dump_metadata(vendor_path(args), cann_ops_transformer.__file__)
        except Exception as exc:
            print("Could not inspect conventional metadata locations:", exc)

        result["stage"] = "allocate"
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        # Width=4, one real sequence, cache row 1 (row 0 is the null slot).
        length = 4 if is_prefill else 1
        x_shape = (1, length, 128) if args.probe.endswith("3d") else (length, 128)
        state = torch.zeros((2, 3, 128), dtype=dtype, device=device)
        tensors = {
            "x": torch.ones(x_shape, dtype=dtype, device=device),
            "weight": torch.full((4, 128), 0.25, dtype=dtype, device=device),
            "query_start_loc": torch.tensor([0, length], dtype=torch.int32, device=device),
        }
        indices = torch.tensor([1], dtype=torch.int32, device=device)
        if is_prefill:
            tensors.update(
                conv_states=state,
                cache_indices=indices,
                has_initial_state=torch.tensor([0], dtype=torch.int32, device=device),
            )
        else:
            tensors.update(
                conv_state=state,
                conv_state_indices=indices,
                num_accepted_tokens=(
                    torch.tensor([1], dtype=torch.int32, device=device) if args.probe == "decode-2d" else None
                ),
            )
        for name, tensor in tensors.items():
            if tensor is None:
                print("INPUT:", name, "None")
                continue
            try:
                fmt = torch_npu.get_npu_format(tensor)
            except Exception:
                fmt = "unavailable"
            print(
                "INPUT:", name, "shape=", list(tensor.shape), "dtype=", str(tensor.dtype),
                "stride=", list(tensor.stride()), "format=", fmt, "device=", str(tensor.device),
            )
        torch.npu.synchronize()
        result["stage"] = "operator"
        y = operator(**tensors, bias=None, activation="silu")
        result["stage"] = "synchronize"
        torch.npu.synchronize()
        result["stage"] = "result_check"
        y_cpu = y.detach().float().cpu()
        state_cpu = state.detach().float().cpu()
        expected = torch.nn.functional.silu(torch.arange(1, length + 1, dtype=torch.float32) * 0.25)
        expected = expected[:, None].expand(length, 128).reshape(x_shape)
        y_ok = tuple(y_cpu.shape) == x_shape and bool(torch.allclose(y_cpu, expected, atol=2e-3, rtol=2e-2))
        expected_state = torch.zeros((2, 3, 128))
        expected_state[1, -min(length, 3):, :] = 1
        state_ok = bool(torch.equal(state_cpu, expected_state))
        result.update(y_ok=y_ok, state_ok=state_ok)
        if not (y_ok and state_ok):
            raise RuntimeError(f"Operator executed but result check failed: y_ok={y_ok}, state_ok={state_ok}")
        result.update(status="PASS", stage="done")
    except Exception as exc:
        result["error"] = short_error(exc)
        traceback.print_exc()
    finally:
        result["opp_after_call"] = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        print("OPP_FINAL:", result["opp_after_call"])
        before_call_entries = normalized_opp_entries(result.get("opp_before_call"))
        final_entries = normalized_opp_entries(result["opp_after_call"])
        final_contains_vendor = real(vendor_path(args)) in final_entries
        result["control_valid"] = (
            "opp_before_call" in result
            and before_call_entries == expected_entries
            and final_entries == expected_entries
            and final_contains_vendor == args.child.startswith("B_")
        )
        if final_entries != expected_entries or (
            "opp_before_call" in result and not result["control_valid"]
        ):
            result["original_stage"] = result["stage"]
            result["original_error"] = result.get("error")
            result.update(
                status="INCOMPLETE",
                stage="environment_control",
                error="Could not verify unchanged OPP order across imports and the call; inspect the log",
            )
        try:
            result["mapped_libraries"] = dump_libraries()
        except Exception as exc:
            print("Could not inspect mapped libraries:", exc)
        Path(args.result_file).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("RESULT:", json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


def parent(args):
    if sys.platform != "linux":
        raise SystemExit("Run this script on the failing Linux/NPU server, not the local PC.")
    vendor = vendor_path(args)
    if not vendor.is_dir():
        raise SystemExit(f"Repository vendor directory not found: {vendor}")
    if args.output:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="kimi-conv-diag-"))
    entries = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    remaining = [entry for entry in entries if entry and real(entry) != real(vendor)]
    print("Reports:", output, flush=True)
    print(
        "A removes only the selected repository OPP entry; B prepends it. Other environment variables stay unchanged.",
        flush=True,
    )
    results = []
    for dtype in (args.dtype,):
        for mode in ("A_without_repo_opp", "B_with_repo_opp"):
            env = os.environ.copy()
            opp_entries = remaining if mode.startswith("A_") else [str(vendor), *remaining]
            if opp_entries:
                env["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(opp_entries)
            else:
                env.pop("ASCEND_CUSTOM_OPP_PATH", None)
            name = f"{mode}_{args.probe}_{dtype}"
            result_file = output / f"{name}.json"
            log_file = output / f"{name}.log"
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()), "--child", mode,
                "--probe", args.probe, "--dtype", dtype, "--repo", str(Path(args.repo).resolve()),
                "--device", str(args.device), "--result-file", str(result_file),
            ]
            failure = None
            returncode = None
            with log_file.open("w") as log:
                try:
                    process = subprocess.run(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                    )
                    returncode = process.returncode
                    if returncode and not result_file.exists():
                        failure = f"Process exited with code {returncode}; inspect log"
                except subprocess.TimeoutExpired:
                    failure = f"TIMEOUT after {args.timeout}s (may include JIT compilation)"
            if failure:
                result = {"case": mode, "probe": args.probe, "dtype": dtype, "status": "INCOMPLETE", "error": failure}
            else:
                if result_file.exists():
                    result = json.loads(result_file.read_text())
                else:
                    result = {
                        "case": mode, "probe": args.probe, "dtype": dtype,
                        "status": "INCOMPLETE", "error": "Missing result file",
                    }
            result["returncode"] = returncode
            if returncode and result["status"] == "PASS":
                result.update(
                    status="INCOMPLETE",
                    stage="child_exit",
                    error=f"Operator passed but subprocess exited with code {returncode}; inspect log",
                )
            results.append(result)
            print(
                f"{name}: {result['status']} stage={result.get('stage', 'unknown')} {result.get('error', '')}",
                flush=True,
            )
    summary = {
        "repo_vendor": str(vendor),
        "original_custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Summary:", output / "summary.json")
    print("Only a small operator probe was run; no model or installed operator files were changed.")
    return 0


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(child(arguments) if arguments.child else parent(arguments))
