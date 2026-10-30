# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Compare raw offline precision arms; runtime/ownership validation is separate."""

import argparse
import json
from pathlib import Path

from compare import compare_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--upstream-utils", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_precision(args.v1, args.v2, args.upstream_utils)
    result["strict_generated_ids_equal"] = sum(row["generated_token_ids_equal"] for row in result["rows"])
    result["validation_scope"] = (
        "Raw precision only; same runtime/configuration and NPU ownership require separate evidence."
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case_count", "passed", "failed", "strict_generated_ids_equal")}))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
