#!/usr/bin/env bash
set -euo pipefail

baseline=$(realpath "$1")
evidence=$(realpath "$2")
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
version=9.2.0-beta.2
base_url=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.T3
[[ $(uname -m) == aarch64 ]]
[[ $(id -u) == 0 ]]

# This script is only for the disposable CI container, never for a host installation.
prefix=$(mktemp -d /opt/cann-920beta2-experiment.XXXXXX)
downloads=$(mktemp -d /tmp/cann-920beta2-downloads.XXXXXX)
npu-smi info >"$evidence/npu-before.txt"
python3 -m pip freeze >"$evidence/python-before.txt"
git -C "$baseline" rev-parse HEAD >"$evidence/baseline-sha.txt"

for package in toolkit A3-ops; do
    filename="Ascend-cann-${package}_${version}_linux-aarch64.run"
    curl --fail --location --retry 3 --connect-timeout 30 \
        --referer https://www.hiascend.com/ \
        "$base_url/$filename" --output "$downloads/$filename"
    sha256sum "$downloads/$filename" >>"$evidence/package-sha256.txt"
    bash "$downloads/$filename" --quiet --install --install-for-all --install-path="$prefix"
done

# Discover the actual version directory instead of inheriting the base image's latest symlink.
mapfile -t toolkit_infos < <(find "$prefix" -type f -name ascend_toolkit_install.info)
[[ ${#toolkit_infos[@]} == 1 ]]
cann_root=$(dirname "$(dirname "${toolkit_infos[0]}")")
mapfile -t ops_infos < <(find "$cann_root" -type f -name ascend_ops_install.info)
[[ ${#ops_infos[@]} == 1 ]]
for info in "${toolkit_infos[0]}" "${ops_infos[0]}"; do
    cat "$info" | tee -a "$evidence/cann-install-info.txt"
    grep -Eq '^version=9\.2\.0-beta\.2\r?$' "$info"
done
grep -Eq '^package_name=Ascend-cann-A3-ops\r?$' "${ops_infos[0]}"

# Remove inherited toolkit/OPP paths, retaining the driver and non-CANN dependencies.
for variable in PATH LD_LIBRARY_PATH PYTHONPATH; do
    value=${!variable:-}
    cleaned=''
    IFS=: read -ra entries <<<"$value"
    for entry in "${entries[@]}"; do
        case "$entry" in
            */Ascend/cann-*|*/Ascend/ascend-toolkit*|*/Ascend/opp*) continue ;;
        esac
        [[ -z "$entry" ]] || cleaned="${cleaned:+$cleaned:}$entry"
    done
    export "$variable=$cleaned"
done
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME
unset ASCEND_CUSTOM_OPP_PATH
set +u
source "$cann_root/set_env.sh"
set -u
[[ $(realpath "$ASCEND_OPP_PATH") == "$cann_root"/* ]]
printf 'cann_root=%s\nASCEND_HOME_PATH=%s\nASCEND_OPP_PATH=%s\nLD_LIBRARY_PATH=%s\n' \
    "$cann_root" "${ASCEND_HOME_PATH:-}" "$ASCEND_OPP_PATH" "$LD_LIBRARY_PATH" \
    >"$evidence/active-cann-paths.txt"

# Keep Python dependencies fixed. Fail explicitly if the shared image has drifted.
python3 - <<'PY'
import importlib.metadata
for package, expected in (("torch-npu", "2.10.0.post4"), ("vllm", "0.27.1")):
    actual = importlib.metadata.version(package)
    print(f"{package}={actual}", flush=True)
    assert actual == expected, f"Base image drift: expected {package}={expected}, got {actual}"
PY
MAX_JOBS=32 python3 -m pip install --no-deps --no-build-isolation -e "$baseline"
python3 -m pip freeze >"$evidence/python-after.txt"

export HF_HUB_OFFLINE=1 VLLM_USE_MODELSCOPE=True VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$baseline"
python3 "$scripts/probe.py" --baseline "$baseline" --evidence "$evidence"
npu-smi info >"$evidence/npu-after.txt"
