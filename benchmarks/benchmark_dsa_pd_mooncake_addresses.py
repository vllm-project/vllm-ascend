# SPDX-License-Identifier: Apache-2.0
"""Compare exact list outputs including NumPy array construction and tolist.

Load the production helper without importing the serving stack. NumPy variants
are numeric prototypes only, using checked synthetic values within uint64.
They are not installed in the transfer path by this benchmark.
"""

import argparse
import ast
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


def load_helper(root, ref=None):
    relative_path = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py"
    path = root / relative_path
    source = (
        subprocess.check_output(["git", "show", f"{ref}:{relative_path}"], cwd=root, text=True)
        if ref
        else path.read_text()
    )
    tree = ast.parse(source)
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets)
    ]
    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_block_transfer_addresses"
    ]
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MooncakePullRecvingThread"
    )
    helper = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_append_block_transfer_addresses"
    )
    helper.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *constants,
            *builders,
            helper,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[helper.name]


def list_extend(local, remote, lb, rb, ls, rs, size, src, dst, lengths, li=0, ri=0, targets=None, target=None):
    if size <= 0:
        return
    src.extend([lb + group[0] * ls + li for group in local])
    dst.extend([rb + group[0] * rs + ri for group in remote])
    lengths.extend([size * len(group) for group in local])
    if targets is not None:
        targets.extend([target] * len(local))


def numpy_arrays(local, remote, lb, rb, ls, rs, size, src, dst, lengths, li=0, ri=0, targets=None, target=None):
    if size <= 0:
        return
    n = len(local)
    a = np.fromiter((group[0] for group in local), dtype=np.uint64, count=n)
    b = np.fromiter((group[0] for group in remote), dtype=np.uint64, count=n)
    counts = np.fromiter((len(group) for group in local), dtype=np.uint64, count=n)
    src.extend((a * ls + lb + li).tolist())
    dst.extend((b * rs + rb + ri).tolist())
    lengths.extend((counts * size).tolist())
    if targets is not None:
        targets.extend([target] * n)


def numpy_records(local, remote, lb, rb, ls, rs, size, src, dst, lengths, li=0, ri=0, targets=None, target=None):
    if size <= 0:
        return
    dtype = np.dtype([("local", np.uint64), ("remote", np.uint64), ("blocks", np.uint64)])
    data = np.fromiter(((a[0], b[0], len(a)) for a, b in zip(local, remote)), dtype=dtype, count=len(local))
    src.extend((data["local"] * ls + lb + li).tolist())
    dst.extend((data["remote"] * rs + rb + ri).tolist())
    lengths.extend((data["blocks"] * size).tolist())
    if targets is not None:
        targets.extend([target] * len(local))


def calibrate(operation):
    iterations = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            operation()
        if time.perf_counter_ns() - start >= 5_000_000:
            return iterations
        iterations *= 2


def measure_variants(operations, samples):
    iterations = {name: calibrate(operation) for name, operation in operations.items()}
    values = {name: [] for name in operations}
    for sample in range(samples):
        order = list(operations)
        order = order[sample % len(order) :] + order[: sample % len(order)]
        for name in order:
            operation, count = operations[name], iterations[name]
            start = time.perf_counter_ns()
            for _ in range(count):
                operation()
            values[name].append((time.perf_counter_ns() - start) / count / 1000)
    return {
        name: {"median_us": statistics.median(data), "samples_us": data, "iterations": iterations[name]}
        for name, data in values.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--baseline-ref", default="95e59fd53d18650f59feaca5f0c2fa0bfc3d0c65")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    production = load_helper(root)
    variants = {
        "baseline": load_helper(root, args.baseline_ref),
        "candidate": production,
        "list_extend": list_extend,
        "numpy_arrays": numpy_arrays,
        "numpy_records": numpy_records,
    }
    results = []
    for count in [1, 4, 16, 33, 128, 512, 2048]:
        local = [[i * 2] for i in range(count)]
        remote = [[i * 3] for i in range(count)]
        if count == 1:
            local, remote = [list(range(33))], [list(range(64, 97))]
        target = object()

        def run(fn, local=local, remote=remote, target=target):
            src, dst, lengths, targets = [], [], [], []
            fn(local, remote, 1 << 40, 1 << 42, 131072, 131072, 131072, src, dst, lengths, 16, 32, targets, target)
            return src, dst, lengths, targets

        expected = run(production)
        for name, fn in variants.items():
            assert run(fn) == expected, name
        results.append(
            {
                "groups": count,
                "variants": measure_variants(
                    {name: lambda fn=fn, run=run: run(fn) for name, fn in variants.items()}, args.samples
                ),
            }
        )
    result = {
        "samples": args.samples,
        "results": results,
        "outputs_equal": True,
        "baseline_ref": args.baseline_ref,
        "scope": (
            "Address helper only, includes fresh outputs, array construction, and tolist; "
            "excludes native binding conversion."
        ),
        "limitations": (
            "Numeric prototypes do not include uint64 overflow fallback checks and are not production replacements."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
