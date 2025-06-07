# bash fonts colors
cyan='\e[96m'
yellow='\e[33m'
red='\e[31m'
none='\e[0m'

_cyan() { echo -e "${cyan}$*${none}"; }
_yellow() { echo -e "${yellow}$*${none}"; }
_red() { echo -e "${red}$*${none}"; }

_info() { _cyan "Info: $*"; }
_warn() { _yellow "Warn: $*"; }
_err() { _red "Error: $*" && exit 1; }

CURL_TIMEOUT=1
CURL_COOLDOWN=5
CURL_MAX_TRIES=120
VENV_PATH=/tmp/vllm_venv

function wait_url_ready() {
  local serve_name="$1"
  local url="$2"
  i=0
  while true; do
    _info "===> Waiting for ${serve_name} to be ready...${i}s"
    i=$((i + CURL_COOLDOWN))
    set +e
    curl --silent --max-time "$CURL_TIMEOUT" "${url}" >/dev/null
    result=$?
    set -e
    if [ "$result" -eq 0 ]; then
      break
    fi
    if [ "$i" -gt "$CURL_MAX_TRIES" ]; then
      _info "===> \$CURL_MAX_TRIES exceeded waiting for ${serve_name} to be ready"
      return 1
    fi
    sleep "$CURL_COOLDOWN"
  done
  _info "===> ${serve_name} is ready."
}

function wait_for_exit() {
  local VLLM_PID="$1"
  while kill -0 "$VLLM_PID"; do
    _info "===> Wait for ${VLLM_PID} to exit."
    sleep 1
  done
  _info "===> Process ${VLLM_PID} has exited."
}

function check_npus() {
  # shellcheck disable=SC2155
  read npu_count npu_type <<< $(python3 -c "import torch.npu; print(torch.npu.device_count(), torch.npu.get_device_name(0))")

  if [[ -z "$npu_count" || "$npu_count" -eq 0 ]]; then
    _warn "Need at least 1 NPU to run"
    exit 1
  else
    _info "Found NPU conut: $npu_count"
  fi
  _info "NPU type is: $npu_type"
}

function clean_venv() {
  if [[ -n "$VENV_PATH" && -d "$VENV_PATH" ]]; then
    _info "Cleaning up default virtual env path: ${VENV_PATH}"
    deactivate || true
    rm -rf "$VENV_PATH"
  fi
}

function create_vllm_venv() {
  # make a clean env path
  clean_venv
  _info "Creating vllm virtual environment at ${VENV_PATH}"
  python3 -m venv ${VENV_PATH}
  source ${VENV_PATH}/bin/activate
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

