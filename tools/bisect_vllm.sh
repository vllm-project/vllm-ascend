#!/usr/bin/env bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# bisect_vllm.sh - Automatically bisect vllm commits to find the first
# commit that breaks a given vllm-ascend test case.
#
# Usage:
#   bisect_vllm.sh \
#     --good <commit>          # Known good vllm commit (default: auto from origin/main)
#     --bad  <commit>          # Known bad vllm commit  (default: auto from current branch)
#     --test-cmd "<cmd>"       # The pytest command that is failing
#     [--vllm-repo <path>]     # vllm repo path (default: auto-detect via pip show, fallback ./vllm-empty)
#     [--ascend-repo <path>]   # vllm-ascend repo path (default: .)
#     [--fetch-depth 500]      # git fetch depth (default: 500, increase if commits not reachable)
#     [--step-timeout 600]     # Per-step timeout in seconds (default: 600)
#     [--total-timeout 7200]   # Total timeout in seconds (default: 7200)
#     [--env "K1=V1 K2=V2"]   # Extra environment variables for the test command
#
# Examples:
#   # With environment variables in --test-cmd (inline style):
#   ./tools/bisect_vllm.sh \
#     --test-cmd "VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/multicard/4-cards/long_sequence/test_accuracy.py"
#
#   # Or use --env to set environment variables separately:
#   ./tools/bisect_vllm.sh \
#     --env "VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_MODELSCOPE=true" \
#     --test-cmd "pytest -sv tests/e2e/multicard/4-cards/long_sequence/test_accuracy.py"
#
#   # Specify commits explicitly:
#   ./tools/bisect_vllm.sh \
#     --good abc1234 --bad def5678 \
#     --test-cmd "pytest -sv tests/e2e/singlecard/test_models.py"
#
#   # Use branch/tag names (resolved via git rev-parse):
#   ./tools/bisect_vllm.sh --good v0.14.1 --bad main \
#     --test-cmd "pytest -sv tests/ut/worker/"
#
#   # Local run (skip git fetch, use local repo as-is):
#   ./tools/bisect_vllm.sh --no-fetch \
#     --good abc1234 --bad def5678 \
#     --vllm-repo /path/to/vllm \
#     --test-cmd "pytest -sv tests/ut/ops/test_activation.py"

set -euo pipefail

# ============================================================================
# Constants
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_SCRIPT="${SCRIPT_DIR}/bisect_helper.py"
BISECT_LOG_FILE="/tmp/bisect_log.json"
BISECT_VERDICT_FILE="/tmp/bisect_verdict.txt"

# Detect if running in CI or locally
IS_GITHUB_ACTIONS="${GITHUB_ACTIONS:-false}"

# Colors for terminal output (disable if not a terminal or in CI log)
if [[ -t 1 ]] || [[ "${IS_GITHUB_ACTIONS}" == "true" ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    NC=''
fi

# ============================================================================
# Default values
# ============================================================================
GOOD_COMMIT=""
BAD_COMMIT=""
TEST_CMD=""
VLLM_REPO=""
ASCEND_REPO=""
FETCH_DEPTH=500
STEP_TIMEOUT=600
TOTAL_TIMEOUT=7200
EXTRA_ENV=""
NO_FETCH=false

# ============================================================================
# Logging helpers
# ============================================================================
log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BOLD}${BLUE}==>${NC}${BOLD} $*${NC}"; }

# ============================================================================
# Usage
# ============================================================================
usage() {
    cat <<'EOF'
Usage: bisect_vllm.sh [OPTIONS]

Required:
  --test-cmd <cmd>          The failing pytest command.
                            Environment variables can be included inline:
                            "VLLM_WORKER_MULTIPROC_METHOD=spawn pytest -sv ..."

Optional:
  --good <commit>           Known good vllm commit (default: auto from origin/main)
  --bad  <commit>           Known bad vllm commit  (default: auto from current branch)
  --vllm-repo <path>        Path to vllm repo (default: auto-detect via pip show, fallback ./vllm-empty)
  --ascend-repo <path>      Path to vllm-ascend repo (default: auto-detect via pip show, fallback .)
  --env "K=V K2=V2"         Extra environment variables for the test command
  --fetch-depth <N>         Git fetch depth (default: 500, increase if commits not reachable)
  --no-fetch                Skip git fetch (use local repo history as-is, useful for local runs)
  --step-timeout <seconds>  Per-step timeout (default: 600)
  --total-timeout <seconds> Total timeout (default: 7200)
  -h, --help                Show this help message

Commit can be specified as: hash, branch name, tag (e.g. v0.15.0), or HEAD~N.
EOF
    exit 0
}

# ============================================================================
# Parse arguments
# ============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --good)          GOOD_COMMIT="$2"; shift 2 ;;
            --bad)           BAD_COMMIT="$2";  shift 2 ;;
            --test-cmd)      TEST_CMD="$2";    shift 2 ;;
            --vllm-repo)     VLLM_REPO="$2";   shift 2 ;;
            --ascend-repo)   ASCEND_REPO="$2";  shift 2 ;;
            --env)           EXTRA_ENV="$2";    shift 2 ;;
            --fetch-depth)   FETCH_DEPTH="$2";  shift 2 ;;
            --no-fetch)      NO_FETCH=true;     shift ;;
            --step-timeout)  STEP_TIMEOUT="$2"; shift 2 ;;
            --total-timeout) TOTAL_TIMEOUT="$2"; shift 2 ;;
            -h|--help)       usage ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done

    if [[ -z "${TEST_CMD}" ]]; then
        log_error "--test-cmd is required"
        usage
    fi
}

# ============================================================================
# Auto-detect paths
# ============================================================================
detect_vllm_repo() {
    if [[ -n "${VLLM_REPO}" ]]; then
        log_info "Using specified vllm repo: ${VLLM_REPO}"
        return
    fi

    # Try pip show first (helper prefers Editable project location)
    local pip_location
    pip_location=$(python3 "${HELPER_SCRIPT}" vllm-location 2>/dev/null || true)
    if [[ -n "${pip_location}" && -d "${pip_location}" ]]; then
        VLLM_REPO="${pip_location}"
        log_info "Auto-detected vllm repo via pip show: ${VLLM_REPO}"
        return
    fi

    # Fallback to ./vllm-empty (CI convention)
    if [[ -d "./vllm-empty" ]]; then
        VLLM_REPO="./vllm-empty"
        log_info "Using default vllm repo path: ${VLLM_REPO}"
    else
        log_error "Cannot detect vllm repo path. Use --vllm-repo to specify."
        exit 1
    fi
}

detect_ascend_repo() {
    if [[ -n "${ASCEND_REPO}" ]]; then
        log_info "Using specified vllm-ascend repo: ${ASCEND_REPO}"
        return
    fi

    # Try pip show (prefer Editable project location, fallback to Location)
    local pip_location
    pip_location=$(pip show vllm-ascend 2>/dev/null \
        | grep -E "^(Editable project location|Location):" \
        | head -1 | sed 's/^[^:]*: *//' || true)
    if [[ -n "${pip_location}" && -d "${pip_location}" ]]; then
        ASCEND_REPO="${pip_location}"
        log_info "Auto-detected vllm-ascend repo via pip show: ${ASCEND_REPO}"
        return
    fi

    # Default to current directory
    ASCEND_REPO="."
    log_info "Using default vllm-ascend repo path: ${ASCEND_REPO}"
}

# ============================================================================
# Auto-detect commits
# ============================================================================
detect_commits() {
    local yaml_path="${ASCEND_REPO}/.github/workflows/pr_test_light.yaml"

    if [[ -z "${GOOD_COMMIT}" ]]; then
        log_info "Auto-detecting good commit from origin/main..."
        GOOD_COMMIT=$(python3 "${HELPER_SCRIPT}" get-commit \
            --yaml-path "${yaml_path}" --ref origin/main 2>/dev/null || true)
        if [[ -z "${GOOD_COMMIT}" ]]; then
            log_error "Cannot auto-detect good commit."
            log_warn "This can happen when running locally without origin/main."
            log_warn "Use --good <commit> to specify the known good vllm commit."
            exit 1
        fi
        log_ok "Good commit (from origin/main): ${GOOD_COMMIT}"
    fi

    if [[ -z "${BAD_COMMIT}" ]]; then
        log_info "Auto-detecting bad commit from current branch..."
        BAD_COMMIT=$(python3 "${HELPER_SCRIPT}" get-commit \
            --yaml-path "${yaml_path}" 2>/dev/null || true)
        if [[ -z "${BAD_COMMIT}" ]]; then
            log_error "Cannot auto-detect bad commit."
            log_warn "Use --bad <commit> to specify the known bad vllm commit."
            exit 1
        fi
        log_ok "Bad commit (from current branch): ${BAD_COMMIT}"
    fi
}

# ============================================================================
# Resolve commit references (branch/tag/HEAD~N) to full hash.
# Validates that the resolved object is actually a commit.
# ============================================================================
resolve_commit() {
    local ref="$1"
    local label="$2"
    local resolved

    # First try rev-parse directly
    resolved=$(git -C "${VLLM_REPO}" rev-parse --verify "${ref}^{commit}" 2>/dev/null || true)
    if [[ -z "${resolved}" ]]; then
        # Try with origin/ prefix for remote branches
        resolved=$(git -C "${VLLM_REPO}" rev-parse --verify "origin/${ref}^{commit}" 2>/dev/null || true)
    fi

    if [[ -z "${resolved}" ]]; then
        log_error "Cannot resolve ${label} commit '${ref}'."
        log_error "The commit is not reachable in the current repo history."
        log_warn "Possible causes:"
        log_warn "  - The commit hash is incorrect"
        log_warn "  - The commit is outside --fetch-depth ${FETCH_DEPTH} (try --fetch-depth 0 for full history)"
        log_warn "  - Running locally without remote; use --no-fetch and ensure history is available"
        exit 1
    fi

    # Double-check it's a commit object
    local obj_type
    obj_type=$(git -C "${VLLM_REPO}" cat-file -t "${resolved}" 2>/dev/null || true)
    if [[ "${obj_type}" != "commit" ]]; then
        log_error "'${ref}' resolves to a ${obj_type:-unknown} object, not a commit."
        exit 1
    fi

    echo "${resolved}"
}

# ============================================================================
# Prepare vllm repo for bisect
# ============================================================================
prepare_vllm_repo() {
    log_step "Preparing vllm repo for bisect"

    # Fetch enough history for bisect (skip if --no-fetch)
    if [[ "${NO_FETCH}" == "true" ]]; then
        log_info "Skipping git fetch (--no-fetch specified)"
    else
        log_info "Fetching vllm history (depth=${FETCH_DEPTH})..."
        if [[ "${FETCH_DEPTH}" == "0" ]]; then
            git -C "${VLLM_REPO}" fetch origin --unshallow 2>/dev/null \
                || git -C "${VLLM_REPO}" fetch origin 2>/dev/null \
                || log_warn "git fetch failed; continuing with existing history"
        else
            git -C "${VLLM_REPO}" fetch origin --depth="${FETCH_DEPTH}" 2>/dev/null \
                || git -C "${VLLM_REPO}" fetch origin 2>/dev/null \
                || log_warn "git fetch failed; continuing with existing history"
        fi
    fi

    # Resolve commits to full hashes
    GOOD_COMMIT=$(resolve_commit "${GOOD_COMMIT}" "good")
    BAD_COMMIT=$(resolve_commit "${BAD_COMMIT}" "bad")

    log_ok "Good commit resolved: ${GOOD_COMMIT}"
    log_ok "Bad commit resolved:  ${BAD_COMMIT}"

    # Verify good != bad
    if [[ "${GOOD_COMMIT}" == "${BAD_COMMIT}" ]]; then
        log_error "Good and bad commits are the same (${GOOD_COMMIT:0:12}). Nothing to bisect."
        exit 1
    fi

    # Verify commits are in ancestor relationship
    local commit_count
    commit_count=$(git -C "${VLLM_REPO}" rev-list --count "${GOOD_COMMIT}..${BAD_COMMIT}" 2>/dev/null || echo "0")
    if [[ "${commit_count}" == "0" ]]; then
        log_error "No commits found between good (${GOOD_COMMIT:0:12}) and bad (${BAD_COMMIT:0:12})."
        log_error "Either the range is empty, or good is not an ancestor of bad."
        log_warn "Check that --good is older than --bad, and both are reachable with --fetch-depth ${FETCH_DEPTH}."
        exit 1
    fi
    log_info "Commits in range: ${commit_count}"
}

# ============================================================================
# Check if setup files changed between two commits
# ============================================================================
setup_files_changed() {
    local commit="$1"
    local prev_commit="$2"

    local changed
    changed=$(git -C "${VLLM_REPO}" diff --name-only "${prev_commit}" "${commit}" -- \
        setup.py pyproject.toml setup.cfg requirements*.txt 2>/dev/null || true)

    [[ -n "${changed}" ]]
}

# ============================================================================
# Run the test command.
# Pytest output goes directly to the terminal (stdout/stderr) so that
# ``pytest -sv`` logs are visible in real time.
# The verdict ("good", "bad", or "skip") is written to BISECT_VERDICT_FILE
# so the caller can read it without swallowing pytest output via $().
# ============================================================================
run_test() {
    local exit_code=0
    local saved_dir
    saved_dir="$(pwd)"

    cd "${ASCEND_REPO}"

    # Build the command with optional extra env vars
    local full_cmd="${TEST_CMD}"
    if [[ -n "${EXTRA_ENV}" ]]; then
        full_cmd="${EXTRA_ENV} ${full_cmd}"
    fi

    log_info "Running: ${full_cmd}"

    # Run test with timeout; output goes directly to terminal
    timeout "${STEP_TIMEOUT}" bash -c "${full_cmd}" || exit_code=$?

    cd "${saved_dir}"

    # Write verdict to file (NOT stdout) so pytest output is not swallowed
    if [[ ${exit_code} -eq 0 ]]; then
        echo "good" > "${BISECT_VERDICT_FILE}"
    elif [[ ${exit_code} -eq 124 ]]; then
        # timeout exit code
        log_warn "Test timed out after ${STEP_TIMEOUT}s"
        echo "skip" > "${BISECT_VERDICT_FILE}"
    else
        echo "bad" > "${BISECT_VERDICT_FILE}"
    fi
    return 0
}

# ============================================================================
# Main bisect loop
# ============================================================================
run_bisect() {
    log_step "Starting bisect: good=${GOOD_COMMIT:0:12} bad=${BAD_COMMIT:0:12}"

    # Initialize bisect
    if ! git -C "${VLLM_REPO}" bisect start "${BAD_COMMIT}" "${GOOD_COMMIT}" --no-checkout 2>/dev/null; then
        log_error "Failed to initialize git bisect. Check that good/bad commits are valid."
        exit 1
    fi

    # Track state for reporting
    local step=0
    local skipped_commits=""
    local prev_commit="${GOOD_COMMIT}"
    local start_time
    start_time=$(date +%s)

    # Initialize log file
    echo "[]" > "${BISECT_LOG_FILE}"

    while true; do
        step=$((step + 1))

        # Check total timeout
        local elapsed=$(( $(date +%s) - start_time ))
        if [[ ${elapsed} -ge ${TOTAL_TIMEOUT} ]]; then
            log_error "Total timeout (${TOTAL_TIMEOUT}s) reached after ${step} steps"
            git -C "${VLLM_REPO}" bisect reset 2>/dev/null || true
            exit 1
        fi

        # Get current bisect commit
        local current_commit
        current_commit=$(git -C "${VLLM_REPO}" rev-parse BISECT_HEAD 2>/dev/null || true)

        if [[ -z "${current_commit}" ]]; then
            log_info "Bisect completed (no more BISECT_HEAD)"
            break
        fi

        local short_commit="${current_commit:0:12}"
        local commit_msg
        commit_msg=$(git -C "${VLLM_REPO}" log -1 --format="%s" "${current_commit}" 2>/dev/null || echo "(unable to read commit message)")

        echo ""
        log_step "Step ${step}: testing commit ${short_commit} - ${commit_msg}"
        echo ""

        # Checkout the commit
        if ! git -C "${VLLM_REPO}" checkout "${current_commit}" --quiet 2>/dev/null; then
            log_warn "Failed to checkout ${short_commit}, skipping"
            git -C "${VLLM_REPO}" bisect skip 2>/dev/null || true
            _append_log "${step}" "${short_commit}" "skip"
            skipped_commits="${skipped_commits:+${skipped_commits},}${short_commit}"
            prev_commit="${current_commit}"
            continue
        fi

        # Check if setup files changed → reinstall if needed
        if setup_files_changed "${current_commit}" "${prev_commit}"; then
            log_warn "Setup files changed, reinstalling vllm..."
            local saved_dir
            saved_dir="$(pwd)"
            cd "${VLLM_REPO}"
            VLLM_TARGET_DEVICE=empty pip install -e . --quiet 2>/dev/null || {
                log_warn "pip install failed for ${short_commit}, skipping"
                cd "${saved_dir}"
                git -C "${VLLM_REPO}" bisect skip 2>/dev/null || true

                _append_log "${step}" "${short_commit}" "skip"
                skipped_commits="${skipped_commits:+${skipped_commits},}${short_commit}"
                prev_commit="${current_commit}"
                continue
            }
            cd "${saved_dir}"
        fi

        # Run the test — output goes directly to terminal
        run_test

        # Read verdict from file
        local result
        result=$(cat "${BISECT_VERDICT_FILE}" 2>/dev/null || echo "bad")

        echo ""
        log_info "Step ${step}: commit ${short_commit} → ${result}"

        # Feed result back to git bisect and capture exit code
        local bisect_output
        local bisect_rc=0
        case "${result}" in
            good)
                bisect_output=$(git -C "${VLLM_REPO}" bisect good 2>&1) || bisect_rc=$?
                ;;
            bad)
                bisect_output=$(git -C "${VLLM_REPO}" bisect bad 2>&1) || bisect_rc=$?
                ;;
            skip)
                bisect_output=$(git -C "${VLLM_REPO}" bisect skip 2>&1) || bisect_rc=$?
                skipped_commits="${skipped_commits:+${skipped_commits},}${short_commit}"
                ;;
            *)
                log_error "Unexpected test result: '${result}' for commit ${short_commit}"
                git -C "${VLLM_REPO}" bisect reset --quiet 2>/dev/null || true
                return 1
                ;;
        esac

        # Log entry
        _append_log "${step}" "${short_commit}" "${result}"

        # Check if bisect is done
        if echo "${bisect_output}" | grep -q "is the first bad commit"; then
            echo ""
            log_ok "===== Bisect completed! ====="
            echo "${bisect_output}"

            # Extract the first bad commit
            local first_bad
            first_bad=$(echo "${bisect_output}" | head -1 | awk '{print $1}')

            _generate_report "${first_bad}" "${step}" "${skipped_commits}"

            # Reset bisect state
            git -C "${VLLM_REPO}" bisect reset --quiet 2>/dev/null || true
            return 0
        fi

        # Check for bisect errors via exit code (avoids false positives from commit messages)
        if [[ ${bisect_rc} -ne 0 ]]; then
            log_error "git bisect ${result} failed with exit code ${bisect_rc}:"
            echo "${bisect_output}"
            git -C "${VLLM_REPO}" bisect reset --quiet 2>/dev/null || true
            return 1
        fi

        prev_commit="${current_commit}"
    done

    # If we reach here without finding the bad commit
    log_error "Bisect did not converge. Check the commit range and test command."
    git -C "${VLLM_REPO}" bisect reset --quiet 2>/dev/null || true
    return 1
}

# ============================================================================
# Helper: append entry to bisect log JSON
# ============================================================================
_append_log() {
    local step="$1"
    local commit="$2"
    local result="$3"

    python3 -c "
import json
with open('${BISECT_LOG_FILE}', 'r') as f:
    log = json.load(f)
log.append({'step': ${step}, 'commit': '${commit}', 'result': '${result}'})
with open('${BISECT_LOG_FILE}', 'w') as f:
    json.dump(log, f)
"
}

# ============================================================================
# Helper: generate final report
# ============================================================================
_generate_report() {
    local first_bad="$1"
    local total_steps="$2"
    local skipped="$3"
    local first_bad_info_file="/tmp/bisect_first_bad_info.txt"

    git -C "${VLLM_REPO}" log -1 --format="commit %H%nAuthor: %an <%ae>%nDate:   %ad%n%n    %s%n%n    %b" \
        "${first_bad}" > "${first_bad_info_file}" 2>/dev/null || echo "N/A" > "${first_bad_info_file}"
    local total_commits
    total_commits=$(git -C "${VLLM_REPO}" rev-list --count "${GOOD_COMMIT}..${BAD_COMMIT}" 2>/dev/null || echo "0")

    # Include extra env in reported test command for clarity
    local reported_cmd="${TEST_CMD}"
    if [[ -n "${EXTRA_ENV}" ]]; then
        reported_cmd="${EXTRA_ENV} ${reported_cmd}"
    fi

    python3 "${HELPER_SCRIPT}" report \
        --good-commit "${GOOD_COMMIT}" \
        --bad-commit "${BAD_COMMIT}" \
        --first-bad "${first_bad}" \
        --first-bad-info-file "${first_bad_info_file}" \
        --test-cmd "${reported_cmd}" \
        --total-steps "${total_steps}" \
        --total-commits "${total_commits}" \
        ${skipped:+--skipped "${skipped}"} \
        --log-file "${BISECT_LOG_FILE}"
}

# ============================================================================
# Main
# ============================================================================
main() {
    parse_args "$@"

    log_step "vllm bisect automation"
    log_info "Test command: ${TEST_CMD}"
    if [[ -n "${EXTRA_ENV}" ]]; then
        log_info "Extra env:    ${EXTRA_ENV}"
    fi

    # Detect paths
    detect_vllm_repo
    detect_ascend_repo

    # Detect commits
    detect_commits

    # Prepare vllm repo
    prepare_vllm_repo

    # Run bisect
    run_bisect
    local exit_code=$?

    if [[ ${exit_code} -eq 0 ]]; then
        log_ok "Bisect completed successfully"
    else
        log_error "Bisect failed"
    fi

    return ${exit_code}
}

main "$@"
