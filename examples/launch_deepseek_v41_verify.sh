#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: launch_deepseek_v41_verify.sh [--dry-run] MODEL_PATH [VLLM_ARGS...]

Run inside a prepared remote NPU environment with both paired repositories
installed (including Ascend custom operators). Source the site's CANN/runtime
environment before invoking this script. MODEL_PATH is a path on that host.

Required environment:
  ASCEND_RT_VISIBLE_DEVICES  Devices assigned to this run; no allocation is done.

Optional environment:
  TP=4 DP=4                 Tensor/data parallelism (default: 16-device A3 setup).
  HOST=127.0.0.1 PORT=8900   Server bind address and port.
  SERVED_MODEL_NAME=dsv41    API model name.
  MAX_MODEL_LEN=32768        Maximum context length.
  MAX_NUM_SEQS=8             Maximum concurrent sequences.
  MAX_BATCHED_TOKENS=4096    Prefill batch token budget.
  GPU_MEMORY_UTILIZATION=0.85
  BLOCK_SIZE=128            Main manager/kernel page; SWA uses 32 tokens.
  GRAPH_MODE=eager          eager or FULL_DECODE_ONLY.
  QUANTIZATION=ascend        Explicit backend; none lets checkpoint config select.
  ENGRAM_CONFIG=none         Text attention smoke test; pass JSON to enable Engram.
  ADDITIONAL_CONFIG         Default: fused MC2, FlashComm1, shared-expert DP.
  GLOO_SOCKET_IFNAME=lo      Single-host Gloo interface.
  VLLM_USE_V2_MODEL_RUNNER=1  MRV2 is required by current upstream.
  VLLM_BIN=vllm              CLI from the prepared Python environment.

Extra arguments are passed verbatim to vllm serve. Default execution is eager,
with expert parallelism and prefix caching, without speculative decoding.
The V4.1 backend selects block-outermost storage in the candidate revision.
EOF
}

dry_run=0
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
    shift
fi
if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi
model_path=$1
shift
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the devices assigned to this run}"

ascend_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
pair_root=$(dirname -- "$ascend_root")
vllm_root="$pair_root/vllm"
if [[ ! -d "$vllm_root/vllm" ]]; then
    echo "Missing paired vLLM source: $vllm_root" >&2
    exit 2
fi

export PYTHONPATH="$ascend_root:$vllm_root${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
engram_config=${ENGRAM_CONFIG:-none}
additional_config=${ADDITIONAL_CONFIG:-'{"enable_fused_mc2":1,"enable_flashcomm1":true,"enable_shared_expert_dp":true}'}

command=("${VLLM_BIN:-vllm}" serve "$model_path"
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8900}"
    --served-model-name "${SERVED_MODEL_NAME:-dsv41}"
    --tensor-parallel-size "${TP:-4}" --data-parallel-size "${DP:-4}"
    --enable-expert-parallel
    --max-model-len "${MAX_MODEL_LEN:-32768}"
    --max-num-seqs "${MAX_NUM_SEQS:-8}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --block-size "${BLOCK_SIZE:-128}"
    --tokenizer-mode deepseek_v41
    --reasoning-parser deepseek_v41
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --safetensors-load-strategy lazy
    --additional-config "$additional_config"
    --enable-prefix-caching)
case ${GRAPH_MODE:-eager} in
    eager) command+=(--enforce-eager) ;;
    FULL_DECODE_ONLY) command+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
    *) echo "Unsupported GRAPH_MODE: $GRAPH_MODE" >&2; exit 2 ;;
esac
if [[ $engram_config == none ]]; then
    # vLLM 0.32 otherwise enables Engram automatically from checkpoint config.
    command+=(--hf-overrides '{"engram_layer_ids":[]}')
else
    command+=(--engram-config "$engram_config")
fi
if [[ ${QUANTIZATION:-ascend} != none ]]; then
    command+=(--quantization "${QUANTIZATION:-ascend}")
fi
command+=("$@")

printf 'Ascend source: %s\n' "$ascend_root"
if [[ -e "$ascend_root/.git" ]]; then
    git -C "$ascend_root" rev-parse HEAD
else
    printf 'Git revision unavailable (source archive)\n'
fi
printf 'vLLM source: %s\n' "$vllm_root"
if [[ -e "$vllm_root/.git" ]]; then
    git -C "$vllm_root" rev-parse HEAD
else
    printf 'Git revision unavailable (source archive)\n'
fi
printf 'Devices: %s; model runner V2: %s\n' "$ASCEND_RT_VISIBLE_DEVICES" "$VLLM_USE_V2_MODEL_RUNNER"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if (( dry_run )); then
    exit 0
fi
exec "${command[@]}"
