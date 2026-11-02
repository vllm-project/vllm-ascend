#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TEST_PATH=tests/e2e/pull_request/one_card/test_hisparse_invalidate_written_slots.py

if [[ ${1:-} == --inside ]]; then
    mode=$2
    artifacts=$3
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export PYTHONPATH="$(dirname -- "$REPO_ROOT")/vllm:$REPO_ROOT"
    python3 -m venv --system-site-packages --without-pip "$artifacts/venv"
    python_bin="$artifacts/venv/bin/python"
    "$python_bin" - <<'PY'
import importlib.metadata as metadata
import platform
import sys
import torch
import triton
print("Python:", sys.executable, platform.python_version())
print("torch:", torch.__version__, "triton:", triton.__version__)
for name in ("torch-npu", "triton-ascend", "pytest"):
    print(name + ":", metadata.version(name))
PY
    args=()
    if [[ $mode == cpu ]]; then
        export TRITON_INTERPRET=1
        args=(-k 'not npu_')
    else
        export TRITON_INTERPRET=0
        "$python_bin" - <<'PY'
import torch
import torch_npu
assert torch.npu.is_available(), "No released, usable NPU is visible"
assert torch.npu.device_count() == 1, "Expected exactly one released physical device"
print("Visible NPUs:", torch.npu.device_count())
print("NPU 0:", torch.npu.get_device_name(0))
PY
    fi
    cd "$REPO_ROOT"
    exec timeout --kill-after=5s 720s "$python_bin" -m pytest \
        --confcutdir=tests/e2e/pull_request/one_card "$TEST_PATH" \
        "${args[@]}" -vv --junitxml="$artifacts/results.xml"
fi

mode=${1:-}
physical_device=${2:-}
if [[ $mode != cpu && $mode != npu ]]; then
    echo "Usage: bash tools/run_hisparse_invalidate_tests.sh cpu" >&2
    echo "   or: bash tools/run_hisparse_invalidate_tests.sh npu PHYSICAL_DEVICE --device-released" >&2
    exit 2
fi
if [[ $mode == npu ]]; then
    if [[ ! $physical_device =~ ^[0-9]+$ || ${3:-} != --device-released ]]; then
        echo "NPU mode requires a physical device and explicit released-device confirmation." >&2
        exit 2
    fi
    test -c "/dev/davinci$physical_device"
fi

image=sha256:8725ad72b9db273848f13894bad0e547a67567cccb2f8692b7e3588b2e959982
stamp=$(date +%Y%m%d-%H%M%S)
workspace=$(dirname -- "$REPO_ROOT")
artifacts="$workspace/.artifacts/hisparse-invalidate-$stamp-$mode"
mkdir -p "$artifacts"
container="codex-h00954440-$stamp-hisparse-$mode-$$"
unit="codex-cleanup-$container"
created=0

cleanup() {
    if [[ $created == 1 ]]; then
        systemctl stop "$unit.timer" >/dev/null 2>&1 || true
        docker rm -f "$container" > "$artifacts/cleanup.log" 2>&1 || true
        docker ps -a --filter "name=^/$container$" --format '{{.Names}} {{.Status}}' \
            >> "$artifacts/cleanup.log"
        if [[ $mode == npu ]]; then
            npu-smi info > "$artifacts/npu-after.txt" 2>&1 || true
        fi
    fi
    echo "Artifacts: $artifacts"
}
trap cleanup EXIT

docker image inspect --format '{{.Id}} {{json .RepoDigests}}' "$image" > "$artifacts/image.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$artifacts/ascend-head.txt"
git -C "$REPO_ROOT" status --short > "$artifacts/ascend-status.txt"
sha256sum "$REPO_ROOT/vllm_ascend/ops/triton/hisparse.py" \
    "$REPO_ROOT/$TEST_PATH" "$REPO_ROOT/tools/run_hisparse_invalidate_tests.sh" \
    > "$artifacts/source-sha256.txt"

device_args=()
if [[ $mode == npu ]]; then
    npu-smi info | tee "$artifacts/npu-before.txt"
    wall "[codex]h00954440 $(date +%F) HiSparse invalidation UT; physical device $physical_device; no ports; container $container; maximum 15 minutes."
    device_args=(
        --device "/dev/davinci$physical_device"
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
        --mount type=bind,src=/usr/local/Ascend/driver/lib64,dst=/usr/local/Ascend/driver/lib64,readonly
        --mount type=bind,src=/usr/local/Ascend/driver/version.info,dst=/usr/local/Ascend/driver/version.info,readonly
        --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly
        --env "ASCEND_RT_VISIBLE_DEVICES=$physical_device"
    )
fi

docker run -d --name "$container" --network none --cpus 2 --memory 8g --pids-limit 512 \
    --security-opt seccomp=unconfined --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    --mount "type=bind,src=$workspace,dst=$workspace" --workdir "$REPO_ROOT" \
    "${device_args[@]}" --entrypoint /bin/bash "$image" -lc 'sleep 900' \
    > "$artifacts/container-id.txt"
created=1
if ! systemd-run --unit "$unit" --on-active=900s --collect /usr/bin/docker rm -f "$container"; then
    echo "Automatic cleanup registration failed; stopping the task container." >&2
    exit 1
fi
docker inspect --format '{{.Name}} devices={{json .HostConfig.Devices}} privileged={{.HostConfig.Privileged}} network={{.HostConfig.NetworkMode}}' \
    "$container" | tee "$artifacts/container-scope.txt"
docker exec "$container" bash "$REPO_ROOT/tools/run_hisparse_invalidate_tests.sh" \
    --inside "$mode" "$artifacts" 2>&1 | tee "$artifacts/pytest.log"
