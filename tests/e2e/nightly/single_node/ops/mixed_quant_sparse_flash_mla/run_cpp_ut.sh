#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../../../../.." && pwd)"
build_dir=${1:?Usage: run_cpp_ut.sh /absolute/ut-build /absolute/deps}
deps_dir=${2:?Usage: run_cpp_ut.sh /absolute/ut-build /absolute/deps}
: "${ASCEND_HOME_PATH:?Source the CANN environment before running this script}"
mkdir -p "$build_dir" "$deps_dir"
build_dir="$(cd "$build_dir" && pwd)"
deps_dir="$(cd "$deps_dir" && pwd)"
if [ "$build_dir" = "$repo/csrc/build" ]; then
    echo "Use an independent UT build directory while compiling custom kernels." >&2
    exit 2
fi
logs="$(dirname "$build_dir")/logs"
mkdir -p "$logs"
export CPATH="$repo/csrc/third_party/catlass/include${CPATH:+:$CPATH}"
cmake -S "$repo/csrc" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_MODE=FALSE \
    -DASCEND_COMPUTE_UNIT=ascend950 \
    -DASCEND_OP_NAME=mixed_quant_sparse_flash_mla \
    -DVENDOR_NAME=custom -DCANN_3RD_LIB_PATH="$deps_dir" \
    -DENABLE_TEST=ON -DOP_HOST_UT=ON -DOP_API_UT=ON \
    -DENABLE_UT_EXEC=OFF -DENABLE_ASAN=OFF \
    -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=OFF -DENABLE_AICPU=OFF \
    2>&1 | tee "$logs/ut-configure.log"
cmake --build "$build_dir" --target transformer_op_host_ut transformer_op_api_ut -j4 \
    2>&1 | tee "$logs/ut-build.log"

export BUILD_PATH="$build_dir"
framework="$build_dir/tests/ut/framework_normal"
export LD_LIBRARY_PATH="$build_dir:$framework/op_host:$framework/op_api:$framework/op_api/op_api_ut_common/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$framework/op_api/transformer_op_api_ut" > "$logs/ut-api-ldd.txt"
readelf -d "$framework/op_api/libopapi_transformer_ut.so" > "$logs/ut-api-link.txt"
for kind in host api; do
    status=0
    "$framework/op_$kind/transformer_op_${kind}_ut" \
        --gtest_output="xml:$logs/ut-$kind.xml" > "$logs/ut-$kind.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$logs/ut-$kind.exit"
    cat "$logs/ut-$kind.log"
    if [ "$status" -ne 0 ]; then
        exit "$status"
    fi
done
python3 - "$logs" <<'PY'
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

logs = Path(sys.argv[1])
results = {}
for name, expected in (("host", 36), ("api", 6)):
    root = ET.parse(logs / f"ut-{name}.xml").getroot()
    assert int(root.get("tests", 0)) == expected, root.attrib
    assert all(int(root.get(key, 0)) == 0 for key in ("failures", "disabled", "errors")), root.attrib
    cases = root.findall(".//testcase")
    assert len(cases) == expected
    assert all(case.get("status") == "run" and case.get("result") == "completed" for case in cases)
    results[name] = {"passed": expected, "time_seconds": root.get("time"), "xml": str(logs / f"ut-{name}.xml")}
summary = {"total_passed": 42, "original_cases": 32, "rope0_cases": 10, "results": results}
(logs / "ut-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
