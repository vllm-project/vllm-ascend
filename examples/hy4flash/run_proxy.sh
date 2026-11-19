#!/usr/bin/env bash
set -euo pipefail

VLLM_ASCEND_PATH="<VLLM_ASCEND_SOURCE_DIRECTORY>"

exec python "$VLLM_ASCEND_PATH/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
  --host 127.0.0.1 --port 8000 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 --decoder-ports 8200
