#!/usr/bin/env bash
set -euo pipefail
role=${1:?Usage: run_node.sh prefill|decode DEVICE MODEL_PATH [PYTHON] [mooncake|memcache]}
device=${2:?Missing device}
model=${3:?Missing model path}
python_bin=${4:-python3}
backend=${5:-mooncake}
case "$role" in
  prefill) kv_role=kv_producer; port=18080 ;;
  decode) kv_role=kv_consumer; port=18081 ;;
  *) echo "Unknown role: $role" >&2; exit 2 ;;
esac
export ASCEND_RT_VISIBLE_DEVICES="$device"
export PYTHONHASHSEED=0
export VLLM_USE_V2_MODEL_RUNNER=0
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
config_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$backend" in
  mooncake) export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-$config_dir/mooncake.json}" ;;
  memcache)
    export MMC_LOCAL_CONFIG_PATH="${MMC_LOCAL_CONFIG_PATH:-$config_dir/mmc-local.conf}"
    port=$((port + 100))
    ;;
  *) echo "Unknown backend: $backend" >&2; exit 2 ;;
esac
kv_config="{\"kv_connector\":\"AscendStoreConnector\",\"kv_role\":\"$kv_role\",\"kv_connector_extra_config\":{\"backend\":\"$backend\",\"use_layerwise\":true,\"pool_pd\":true,\"consumer_is_to_load\":true}}"
exec "$python_bin" -m vllm.entrypoints.openai.api_server \
  --model "$model" --served-model-name qwen3-30b \
  --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 4 \
  --max-num-batched-tokens 128 --gpu-memory-utilization 0.85 \
  --block-size 128 --enforce-eager --no-enable-prefix-caching \
  --kv-transfer-config "$kv_config"
