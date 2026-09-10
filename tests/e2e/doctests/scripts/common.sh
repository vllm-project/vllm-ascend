#!/bin/bash
# Shared runtime utilities for the fixed doctest workers.

# Print an informational runtime message to stdout.
log_info() { printf '\033[96mInfo: %s\033[0m\n' "$*"; }

# Print an error to stderr and terminate the worker.
die() {
  printf '\033[31mError: %s\033[0m\n' "$*" >&2
  exit 1
}

URL_REQUEST_TIMEOUT_SECONDS=1
URL_POLL_INTERVAL_SECONDS=5
URL_READY_TIMEOUT_SECONDS=300
PROCESS_EXIT_TIMEOUT_SECONDS=60
PROCESS_POLL_INTERVAL_SECONDS=1

# Poll an HTTP endpoint until curl reports success or the readiness deadline expires.
function wait_for_url_ready() {
  local service_name="$1"
  local url="$2"
  local deadline=$((SECONDS + URL_READY_TIMEOUT_SECONDS))
  local remaining_seconds request_timeout_seconds sleep_seconds
  while (( SECONDS < deadline )); do
    remaining_seconds=$((deadline - SECONDS))
    if (( remaining_seconds <= 0 )); then
      break
    fi
    request_timeout_seconds=${URL_REQUEST_TIMEOUT_SECONDS}
    if (( request_timeout_seconds > remaining_seconds )); then
      request_timeout_seconds=${remaining_seconds}
    fi
    log_info "Waiting for ${service_name} to be ready..."
    if curl --fail --silent --max-time "${request_timeout_seconds}" "${url}" >/dev/null; then
      log_info "${service_name} is ready."
      return 0
    fi
    remaining_seconds=$((deadline - SECONDS))
    if (( remaining_seconds <= 0 )); then
      break
    fi
    sleep_seconds=${URL_POLL_INTERVAL_SECONDS}
    if (( sleep_seconds > remaining_seconds )); then
      sleep_seconds=${remaining_seconds}
    fi
    sleep "${sleep_seconds}"
  done
  printf 'Timed out after %ss waiting for %s to be ready.\n' "${URL_READY_TIMEOUT_SECONDS}" "${service_name}" >&2
  return 1
}

# Wait for a process to disappear without sending signals; fail on timeout.
function wait_for_process_exit() {
  local pid="$1"
  local deadline=$((SECONDS + PROCESS_EXIT_TIMEOUT_SECONDS))
  local remaining_seconds sleep_seconds
  while kill -0 "${pid}" 2>/dev/null; do
    remaining_seconds=$((deadline - SECONDS))
    if (( remaining_seconds <= 0 )); then
      printf 'Timed out after %ss waiting for process %s to exit.\n' "${PROCESS_EXIT_TIMEOUT_SECONDS}" "${pid}" >&2
      return 1
    fi
    log_info "Waiting for process ${pid} to exit."
    sleep_seconds=${PROCESS_POLL_INTERVAL_SECONDS}
    if (( sleep_seconds > remaining_seconds )); then
      sleep_seconds=${remaining_seconds}
    fi
    sleep "${sleep_seconds}"
  done
  # Reap a child if applicable. Shutdown status and non-child PIDs are not errors here.
  wait "${pid}" 2>/dev/null || true
  log_info "Process ${pid} has exited."
}
