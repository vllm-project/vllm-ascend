# SPDX-License-Identifier: Apache-2.0
"""Validate emitted A3 QLI dtype registration before running NPU tests.

Accept either a generated aic-*-ops-info.json or a captured command artifact
whose stdout contains the operator name followed by its JSON registration.
This checks the generated artifact, rather than searching OpDef source text.
"""

import argparse
import json
from pathlib import Path

OP_NAME = "VllmAscendQuantLightningIndexerV2"


def read_entry(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "stdout" in data:
        prefix = OP_NAME + " "
        for line in data["stdout"].splitlines():
            if line.startswith(prefix):
                return json.loads(line[len(prefix) :])
        raise ValueError(f"{path}: command artifact has no {OP_NAME} registration")
    return data[OP_NAME]


def check_entry(entry):
    def types(field):
        return {value.strip().replace("float32", "float") for value in entry[field]["dtype"].split(",")}

    expected = {
        "input0": {"int8"},
        "input1": {"int8"},
        "input2": {"float16"},
        "input3": {"float16"},
        "input4": {"float16"},
        "output0": {"int32"},
        "output1": {"bfloat16", "float"},
        "output2": {"int32"},
    }
    for field, required in expected.items():
        actual = types(field)
        if actual != required:
            raise ValueError(
                f"{field} {entry[field]['name']}: expected dtypes {sorted(required)}, got {sorted(actual)}"
            )
        formats = {value.strip() for value in entry[field]["format"].split(",")}
        if formats != {"ND"}:
            raise ValueError(f"{field}: expected ND format, got {sorted(formats)}")
    if entry["output1"]["name"] != "sparse_values":
        raise ValueError("output1 must register sparse_values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.registration:
        try:
            check_entry(read_entry(path))
        except (KeyError, ValueError) as error:
            raise SystemExit(f"{path}: FAIL: {error}") from error
        print(f"{path}: PASS: A3 legacy BF16 and native FP32 score registrations")


if __name__ == "__main__":
    main()
