#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tests=tests/e2e/pull_request/one_card

if [[ ${1:-} == --inside ]]; then
    artifacts=$2
    selection=$3
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TRITON_INTERPRET=1 TRITON_DEBUG=1 PYTHONPATH="$(dirname -- "$repo")/vllm:$repo"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python3 -m venv --system-site-packages --without-pip "$artifacts/venv"
    py="$artifacts/venv/bin/python"
    "$py" - <<'PY'
import ast
import importlib.metadata as metadata
from pathlib import Path
import platform
import shutil
import sys
print("Python:", sys.executable, platform.python_version())
for name in ("torch", "torch-npu", "triton-ascend", "pytest", "numpy"):
    print(name, metadata.version(name))
for name in ("vllm_ascend/ops/triton/hisparse.py",
             "tests/e2e/pull_request/one_card/test_hisparse_ops.py",
             "tests/e2e/pull_request/one_card/hisparse_reference.py"):
    ast.parse(Path(name).read_text(), filename=name)
    print("AST PASS", name)
print("Ruff:", shutil.which("ruff") or "NOT INSTALLED")
PY
    files=("$tests/test_hisparse_ops.py")
    expression='not npu_'
    case $selection in
        all) files+=("$tests/test_hisparse_invalidate_written_slots.py");;
        invalidation) files=("$tests/test_hisparse_invalidate_written_slots.py");;
        resolver|gather|regression) expression="$selection and not npu_";;
    esac
    cd "$repo"
    command=("$py" -m pytest --confcutdir="$tests" "${files[@]}" -k "$expression"
             -vv --tb=short --junitxml="$artifacts/results.xml")
    printf '%q ' "${command[@]}" > "$artifacts/command.txt"
    printf '\n' >> "$artifacts/command.txt"
    exec timeout --kill-after=5s 720s "${command[@]}"
fi

if [[ ${1:-} != cpu || ! ${2:-all} =~ ^(all|resolver|gather|regression|invalidation)$ ]]; then
    echo 'Usage: bash tools/run_hisparse_ops_tests.sh cpu [all|resolver|gather|regression|invalidation]' >&2
    exit 2
fi
selection=${2:-all}
image=sha256:8725ad72b9db273848f13894bad0e547a67567cccb2f8692b7e3588b2e959982
stamp=$(date +%Y%m%d-%H%M%S)
workspace=$(dirname -- "$repo")
artifacts="$workspace/.artifacts/hisparse-ops-$stamp-$selection-cpu"
mkdir -p "$artifacts"
container="codex-h00954440-$stamp-hisparse-ops-$$"
unit="codex-cleanup-$container"
created=0
started=$SECONDS
cleanup() {
    status=$?
    if [[ $created == 1 ]]; then
        systemctl stop "$unit.timer" >/dev/null 2>&1 || true
        docker rm -f "$container" > "$artifacts/cleanup.log" 2>&1 || status=1
        remaining=$(docker ps -a --filter "name=^/$container$" --format '{{.Names}}')
        printf 'Remaining task containers: %s\n' "$remaining" >> "$artifacts/cleanup.log"
        [[ -z $remaining ]] || status=1
    fi
    printf 'elapsed_seconds=%s\nexit_status=%s\n' "$((SECONDS - started))" "$status" > "$artifacts/timing.txt"
    echo "Artifacts: $artifacts"
    exit "$status"
}
trap cleanup EXIT
docker image inspect --format '{{.Id}} {{json .RepoDigests}}' "$image" > "$artifacts/image.txt"
git -C "$repo" rev-parse HEAD > "$artifacts/ascend-head.txt"
git -C "$workspace/vllm" rev-parse HEAD > "$artifacts/vllm-head.txt"
git -C "$workspace/vllm" status --porcelain > "$artifacts/vllm-status.txt"
git -C "$repo" status --short > "$artifacts/ascend-status.txt"
sha256sum "$repo/vllm_ascend/ops/triton/hisparse.py" "$repo/$tests/test_hisparse_ops.py" \
    "$repo/$tests/hisparse_reference.py" "$repo/$tests/test_hisparse_invalidate_written_slots.py" \
    "$repo/tools/run_hisparse_ops_tests.sh" "$workspace/vllm/csrc/hisparse_kernels.cu" \
    "$workspace/vllm/csrc/torch_bindings.cpp" > "$artifacts/source-sha256.txt"
bash -n "$repo/tools/run_hisparse_ops_tests.sh"
bash -n "$repo/tools/run_hisparse_invalidate_tests.sh"
for path in vllm_ascend/ops/triton/hisparse.py "$tests/test_hisparse_ops.py" "$tests/hisparse_reference.py" \
            tools/run_hisparse_ops_tests.sh; do
    set +e
    git -C "$repo" diff --no-index --check /dev/null "$path" >> "$artifacts/whitespace.txt" 2>&1
    result=$?
    set -e
    [[ $result -le 1 ]] || exit "$result"
done
test ! -s "$artifacts/whitespace.txt"
docker run -d --name "$container" --network none --cpus 2 --memory 8g --pids-limit 512 \
    --security-opt seccomp=unconfined --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    --mount "type=bind,src=$workspace,dst=$workspace" --workdir "$repo" \
    --entrypoint /bin/bash "$image" -lc 'sleep 900' > "$artifacts/container-id.txt"
created=1
systemd-run --unit "$unit" --on-active=900s --collect /usr/bin/docker rm -f "$container"
docker inspect --format '{{.Name}} devices={{json .HostConfig.Devices}} privileged={{.HostConfig.Privileged}} network={{.HostConfig.NetworkMode}}' \
    "$container" | tee "$artifacts/container-scope.txt"
docker exec "$container" bash "$repo/tools/run_hisparse_ops_tests.sh" --inside "$artifacts" "$selection" \
    2>&1 | tee "$artifacts/pytest.log"
