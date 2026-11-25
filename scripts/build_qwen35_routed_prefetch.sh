#!/usr/bin/env bash
set -euo pipefail

VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
BUILD_DIR="${BUILD_DIR:-${VLLM_ASCEND_ROOT}/build/qwen35-routed-prefetch}"

if [[ ! -d "${VLLM_ASCEND_ROOT}/vllm_ascend/ops" ]]; then
    echo "vLLM-Ascend source tree not found: ${VLLM_ASCEND_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${ASCEND_ENV}" ]]; then
    echo "Ascend environment script not found: ${ASCEND_ENV}" >&2
    exit 1
fi
if ! command -v pgrep >/dev/null; then
    echo "pgrep is required to check for running inference services" >&2
    exit 1
fi
if pgrep -f '[V]LLM::EngineCore|[V]LLM::Worker|[/]vllm serve' >/dev/null; then
    echo "Stop inference services before replacing the kernel binary." >&2
    exit 1
fi

source "${ASCEND_ENV}"
cmake -S "${VLLM_ASCEND_ROOT}/csrc/qwen35_weight_prefetch" \
    -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j2

extension_suffix="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")')"
if [[ -z "${extension_suffix}" ]]; then
    echo "Python did not report an extension suffix via sysconfig" >&2
    exit 1
fi
artifact="${BUILD_DIR}/lib/_qwen35_routed_prefetch${extension_suffix}"
if [[ ! -f "${artifact}" ]]; then
    echo "Expected extension artifact not found: ${artifact}" >&2
    exit 1
fi
install -D -m 0755 "${artifact}" "${VLLM_ASCEND_ROOT}/vllm_ascend/ops/$(basename "${artifact}")"
echo "Installed $(basename "${artifact}") into ${VLLM_ASCEND_ROOT}/vllm_ascend/ops"
