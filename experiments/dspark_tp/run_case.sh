#!/usr/bin/env bash
set -eo pipefail

source /home/vllm/c00944022/cann/910/ascend-toolkit/set_env.sh
source /home/vllm/c00944022/cann/910/nnal/atb/set_env.sh
set -u

exec python3 "$(dirname "$0")/run_case.py" "$@"
