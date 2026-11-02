#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root=$(dirname -- "$repo")
if [[ ${1:-} == --inside ]]; then
    artifacts=$2
    expression=$3
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TRITON_INTERPRET=0 PYTHONPATH="$(dirname -- "$repo")/vllm:$repo"
    python3 - <<'PY'
import importlib.metadata as metadata
import platform
import sys
import torch
import torch_npu
assert torch.npu.is_available() and torch.npu.device_count() == 1
print('NPU:', torch.npu.get_device_name(0))
print('Python:', sys.executable, platform.python_version())
for name in ('torch', 'torch-npu', 'triton-ascend', 'pytest'):
    print(name, metadata.version(name))
PY
    command=(python3 -m pytest --confcutdir=tests/e2e/pull_request/one_card
             tests/e2e/pull_request/one_card/test_hisparse_ops_npu.py
             tests/e2e/pull_request/one_card/test_hisparse_invalidate_written_slots.py
             -k "$expression" -x -vv --tb=short --basetemp="$artifacts/tmp" --junitxml="$artifacts/results.xml")
    printf '%q ' "${command[@]}" > "$artifacts/command.txt"
    printf '\n' >> "$artifacts/command.txt"
    exec timeout --kill-after=5s 720s "${command[@]}"
fi
if [[ $# -lt 2 || ! $1 =~ ^[0-9]+$ ]]; then
    echo 'Usage: bash tools/run_hisparse_npu_tests.sh PHYSICAL_DEVICE RUNNING_CONTAINER [pytest-expression]' >&2
    exit 2
fi
physical=$1
container=$2
expression=${3-}
stamp=$(date +%Y%m%d-%H%M%S)
artifacts="$root/.artifacts/hisparse-npu-ut-$stamp"
mkdir -p "$artifacts"
trap 'npu-smi info > "$artifacts/npu-after.txt"; echo "Artifacts: $artifacts"' EXIT
npu-smi info > "$artifacts/npu-before.txt"
python3 - "$artifacts/npu-before.txt" "$physical" <<'PY'
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text().split('Process id', 1)[-1]
active = {(int(a), int(b)) for a, b, pid in re.findall(r'\|\s*(\d+)\s+(\d+)\s+(\d+)\s+\S', text)}
physical = int(sys.argv[2])
if (physical // 2, physical % 2) in active:
    raise SystemExit('Selected NPU has a compute process; preserving existing work')
PY
echo "HiSparse NPU UT: physical device $physical; runtime $container preserved; maximum 13 minutes."
docker inspect --format '{{.Name}} {{.Image}} {{.State.Status}}' "$container" > "$artifacts/runtime.txt"
git -C "$repo" rev-parse HEAD > "$artifacts/ascend-head.txt"
git -C "$root/vllm" rev-parse HEAD > "$artifacts/vllm-head.txt"
git -C "$repo" status --short > "$artifacts/ascend-status.txt"
git -C "$root/vllm" status --porcelain > "$artifacts/vllm-status.txt"
sha256sum "$repo/vllm_ascend/ops/triton/hisparse.py" \
    "$repo/tests/e2e/pull_request/one_card/test_hisparse_ops_npu.py" \
    "$repo/tests/e2e/pull_request/one_card/test_hisparse_ops.py" \
    "$repo/tests/e2e/pull_request/one_card/test_hisparse_invalidate_written_slots.py" \
    "$repo/tests/e2e/pull_request/one_card/hisparse_reference.py" \
    "$repo/tools/run_hisparse_npu_tests.sh" "$root/vllm/csrc/hisparse_kernels.cu" \
    "$root/vllm/csrc/torch_bindings.cpp" "$root/vllm/tests/v1/attention/test_sparse_mla_backends.py" \
    > "$artifacts/source-sha256.txt"
timeout --kill-after=5s 780s docker exec --workdir "$repo" --env "ASCEND_RT_VISIBLE_DEVICES=$physical" \
    --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 --env "ASCEND_WORK_PATH=$artifacts" \
    --env "TRITON_CACHE_DIR=$root/.artifacts/hisparse-npu-triton-cache" \
    --env "TRITON_DUMP_DIR=$artifacts/triton-dump" \
    "$container" timeout --kill-after=5s 750s bash "$repo/tools/run_hisparse_npu_tests.sh" \
    --inside "$artifacts" "$expression" 2>&1 | tee "$artifacts/pytest.log"
