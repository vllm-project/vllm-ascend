#!/usr/bin/env bash
set -euo pipefail

KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${KERNEL_DIR}/../.." && pwd)"
OPS="dcut_recurrent_gated_delta_rule,dcut_causal_conv1d"
SOC="${SOC_VERSION:-ascend910b}"
BUILD_TORCH=1
TORCH_ONLY=0
JOBS="${MAX_JOBS:-8}"

usage() {
    cat <<'EOF'
Usage: bash dcut/kernel/build.sh [options]

Options:
  --ops=<op1,op2>       Build one or both D-Cut AscendC operators.
  --soc=<soc>           ascend910b, ascend910_93, or ascend950.
  --torch-only          Build only the Torch registration library.
  --skip-torch          Build only the AscendC custom-op package.
  -j<N>                 Parallel jobs for the Torch registration library.
EOF
}

for arg in "$@"; do
    case "${arg}" in
        --ops=*) OPS="${arg#*=}" ;;
        --soc=*) SOC="${arg#*=}" ;;
        --torch-only) TORCH_ONLY=1 ;;
        --skip-torch) BUILD_TORCH=0 ;;
        -j*) JOBS="${arg#-j}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; usage >&2; exit 2 ;;
    esac
done

case "${SOC}" in
    ascend910b*|ascend910_93*|ascend950*) ;;
    *) echo "D-Cut GDN kernels do not support SOC ${SOC}" >&2; exit 2 ;;
esac

IFS=',' read -r -a OP_NAMES <<< "${OPS}"
OVERLAYS=()
cleanup() {
    local overlay
    for overlay in "${OVERLAYS[@]:-}"; do
        if [[ -L "${overlay}" ]]; then
            rm -- "${overlay}"
        fi
    done
}
trap cleanup EXIT

if [[ "${TORCH_ONLY}" == "0" ]]; then
    for op_name in "${OP_NAMES[@]}"; do
        case "${op_name}" in
            dcut_recurrent_gated_delta_rule|dcut_causal_conv1d) ;;
            *) echo "Unknown D-Cut operator: ${op_name}" >&2; exit 2 ;;
        esac
        source_dir="${KERNEL_DIR}/${op_name}"
        overlay="${REPO_ROOT}/csrc/attention/${op_name}"
        if [[ -e "${overlay}" || -L "${overlay}" ]]; then
            echo "Refusing to replace existing path: ${overlay}" >&2
            exit 1
        fi
        ln -s "${source_dir}" "${overlay}"
        OVERLAYS+=("${overlay}")
    done

    (
        cd "${REPO_ROOT}/csrc"
        bash build.sh --pkg --vendor_name=dcut --ops="${OPS}" --soc="${SOC}"
    )
    cleanup
    OVERLAYS=()
fi

if [[ "${BUILD_TORCH}" == "1" ]]; then
    BUILD_DIR="${KERNEL_DIR}/build/torch_extension"
    cmake -S "${KERNEL_DIR}/torch_extension" -B "${BUILD_DIR}" \
        -DREPO_ROOT="${REPO_ROOT}" \
        -DASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
    cmake --build "${BUILD_DIR}" --parallel "${JOBS}"
    echo "Torch registration library: ${BUILD_DIR}/dcut_torch_ops.so"
fi
