#!/usr/bin/env bash
# cleanup_managed_server.sh - stop only the server process owned by this tuning run.

set -euo pipefail

PID_FILE=""
PORT=""
GRACE_SECONDS=15

usage() {
    cat <<'EOF'
Usage: cleanup_managed_server.sh --pid-file <file> [--port <port>] [--grace-seconds <seconds>]

The script only terminates the managed PID recorded in --pid-file and its
process group. If the port is still occupied by a different process, it exits
with an error instead of killing unrelated workloads.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid-file) PID_FILE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --grace-seconds) GRACE_SECONDS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${PID_FILE}" ]]; then
    echo "ERROR: --pid-file is required." >&2
    usage >&2
    exit 1
fi

if [[ ! -f "${PID_FILE}" ]]; then
    echo "INFO: pid file not found, nothing to clean: ${PID_FILE}"
    exit 0
fi

MANAGED_PID="$(tr -d '[:space:]' < "${PID_FILE}")"
if [[ -z "${MANAGED_PID}" ]]; then
    echo "INFO: pid file is empty, nothing to clean."
    rm -f "${PID_FILE}"
    exit 0
fi

kill_managed_process() {
    local pid="$1"
    if ! kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi

    local pgid
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"

    if [[ -n "${pgid}" ]]; then
        kill -TERM "-${pgid}" 2>/dev/null || true
    fi
    kill -TERM "${pid}" 2>/dev/null || true

    sleep "${GRACE_SECONDS}"

    if kill -0 "${pid}" 2>/dev/null; then
        if [[ -n "${pgid}" ]]; then
            kill -KILL "-${pgid}" 2>/dev/null || true
        fi
        kill -KILL "${pid}" 2>/dev/null || true
        sleep 2
    fi
}

kill_managed_process "${MANAGED_PID}"
rm -f "${PID_FILE}"

if [[ -n "${PORT}" ]]; then
    PORT_PIDS="$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)"
    if [[ -n "${PORT_PIDS}" ]]; then
        while IFS= read -r pid; do
            [[ -n "${pid}" ]] || continue
            if [[ "${pid}" != "${MANAGED_PID}" ]]; then
                CMDLINE="$(ps -o command= -p "${pid}" 2>/dev/null | tr -d '\n')"
                echo "ERROR: port ${PORT} is still occupied by non-managed PID ${pid}: ${CMDLINE}" >&2
                exit 1
            fi
        done <<< "${PORT_PIDS}"
    fi
fi

echo "Managed server cleanup completed."
