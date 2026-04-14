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
# Automatically bisect vllm commits to find the first commit that breaks a
# vllm-ascend test case. Supports both local and CI execution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_SCRIPT="${SCRIPT_DIR}/bisect_helper.py"
BISECT_LOG_FILE="/tmp/bisect_log.json"
BISECT_VERDICT_FILE="/tmp/bisect_verdict.txt"
IS_GITHUB_ACTIONS="${GITHUB_ACTIONS:-false}"

if [[ -t 1 ]] || [[ "${IS_GITHUB_ACTIONS}" == "true" ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; NC=''
fi

GOOD_COMMIT=""
BAD_COMMIT=""
TEST_CMD=""
VLLM_REPO=""
ASCEND_REPO=""
FETCH_DEPTH=""
STEP_TIMEOUT=1800
TOTAL_TIMEOUT=20400
EXTRA_ENV=""
TEST_CMDS_FILE=""
SKIPPED_COMMITS=""
FORCE_REINSTALL=false
SUMMARY_OUTPUT="/tmp/bisect_summary.md"

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "${BOLD}${BLUE}==>${NC}${BOLD} $*${NC}"; }
die()       { log_error "$*"; exit 1; }
reset_bisect() { git -C "${VLLM_REPO}" bisect reset --quiet 2>/dev/null || true; }
canonicalize_dir() { (cd "$1" && pwd -P) 2>/dev/null; }

usage() {
    cat <<'EOF'
Usage: bisect_vllm.sh [OPTIONS]

Required (one of):
  --test-cmd <cmd>          The failing pytest command (single mode)
  --test-cmds-file <path>   File with semicolon-separated test commands (batch mode)

Optional:
  --good <commit>           Known good vllm commit (default: auto from origin/main)
  --bad  <commit>           Known bad vllm commit  (default: auto from current branch)
  --vllm-repo <path>        Path to vllm repo (auto-detect via pip show, fallback ./vllm-empty)
  --ascend-repo <path>      Path to vllm-ascend repo (auto-detect via pip show, fallback .)
  --env "K=V K2=V2"         Extra environment variables for the test command
  --fetch-depth <N>         Fetch remote vllm history before bisecting; use 0 for full history
  --step-timeout <seconds>  Per-step timeout (default: 1800)
  --total-timeout <seconds> Total timeout (default: 20400)
  --summary-output <path>   Markdown summary output path (default: /tmp/bisect_summary.md)
  -h, --help                Show this help message

Examples:
  # 1. Local usage: reproduce a known regression range with explicit good/bad commits
  ./tools/bisect_vllm.sh \
    --good 35141a7eeda941a60ad5a4956670c60fd5a77029 \
    --bad 6e1100889e6a675d17ad82815acf8f02f1cc419e \
    --test-cmd "pytest -sv tests/ut/test_example.py::test_case; pytest -sv tests/ut/test_example_2.py"

  # 2. Run against explicit local checkouts without fetching remote history
  ./tools/bisect_vllm.sh \
    --good abc1234 \
    --bad def5678 \
    --vllm-repo /path/to/vllm \
    --ascend-repo /path/to/vllm-ascend \
    --test-cmd "pytest -sv tests/ut/test_example.py"

  # 3. Batch bisect multiple failing commands from a file
  ./tools/bisect_vllm.sh --good abc1234 --bad def5678 --test-cmds-file /tmp/bisect_cmds.txt

Commit can be specified as: hash, branch name, tag (e.g. v0.15.0), or HEAD~N.
EOF
    exit 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --good) GOOD_COMMIT="$2"; shift 2 ;;
            --bad) BAD_COMMIT="$2"; shift 2 ;;
            --test-cmd) TEST_CMD="$2"; shift 2 ;;
            --test-cmds-file) TEST_CMDS_FILE="$2"; shift 2 ;;
            --vllm-repo) VLLM_REPO="$2"; shift 2 ;;
            --ascend-repo) ASCEND_REPO="$2"; shift 2 ;;
            --env) EXTRA_ENV="$2"; shift 2 ;;
            --fetch-depth) FETCH_DEPTH="$2"; shift 2 ;;
            --step-timeout) STEP_TIMEOUT="$2"; shift 2 ;;
            --total-timeout) TOTAL_TIMEOUT="$2"; shift 2 ;;
            --summary-output) SUMMARY_OUTPUT="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    if [[ -z "${TEST_CMD}" && -z "${TEST_CMDS_FILE}" ]]; then
        die "--test-cmd or --test-cmds-file is required"
    fi
    if [[ -n "${TEST_CMDS_FILE}" && ! -f "${TEST_CMDS_FILE}" ]]; then
        die "Test commands file not found: ${TEST_CMDS_FILE}"
    fi
    if [[ -n "${FETCH_DEPTH}" && ! "${FETCH_DEPTH}" =~ ^[0-9]+$ ]]; then
        die "--fetch-depth must be a non-negative integer"
    fi
}

# Repo auto-detection keeps local usage simple while still matching the CI layout.
detect_vllm_repo() {
    local resolved
    if [[ -n "${VLLM_REPO}" ]]; then
        resolved=$(canonicalize_dir "${VLLM_REPO}") || die "vllm repo path does not exist: ${VLLM_REPO}"
        VLLM_REPO="${resolved}"
        log_info "Using specified vllm repo: ${VLLM_REPO}"
        return
    fi
    local pip_location
    pip_location=$(python3 "${HELPER_SCRIPT}" vllm-location 2>/dev/null || true)
    if [[ -n "${pip_location}" && -d "${pip_location}" ]]; then
        VLLM_REPO=$(canonicalize_dir "${pip_location}") || die "Auto-detected vllm repo path does not exist: ${pip_location}"
        log_info "Auto-detected vllm repo via pip show: ${VLLM_REPO}"
        return
    fi
    [[ -d "./vllm-empty" ]] || die "Cannot detect vllm repo path. Use --vllm-repo to specify."
    VLLM_REPO=$(canonicalize_dir "./vllm-empty") || die "Default vllm repo path does not exist: ./vllm-empty"
    log_info "Using default vllm repo path: ${VLLM_REPO}"
}

detect_ascend_repo() {
    local resolved
    if [[ -n "${ASCEND_REPO}" ]]; then
        resolved=$(canonicalize_dir "${ASCEND_REPO}") || die "vllm-ascend repo path does not exist: ${ASCEND_REPO}"
        ASCEND_REPO="${resolved}"
        log_info "Using specified vllm-ascend repo: ${ASCEND_REPO}"
        return
    fi
    local pip_location
    pip_location=$(pip show vllm-ascend 2>/dev/null \
        | awk -F': ' '
            /^Editable project location:/ { editable=$2 }
            /^Location:/ && !location { location=$2 }
            END { print editable ? editable : location }
        ' || true)
    if [[ -n "${pip_location}" && -d "${pip_location}" ]]; then
        ASCEND_REPO=$(canonicalize_dir "${pip_location}") || die "Auto-detected vllm-ascend repo path does not exist: ${pip_location}"
        log_info "Auto-detected vllm-ascend repo via pip show: ${ASCEND_REPO}"
        return
    fi
    ASCEND_REPO=$(canonicalize_dir ".") || die "Cannot resolve current directory for vllm-ascend repo"
    log_info "Using default vllm-ascend repo path: ${ASCEND_REPO}"
}

detect_commit_from_yaml() {
    local target_var="$1" label="$2" yaml_path="$3" ref="${4:-}"
    local args=(--yaml-path "${yaml_path}")
    [[ -n "${ref}" ]] && args+=(--ref "${ref}")
    local commit
    commit=$(python3 "${HELPER_SCRIPT}" get-commit "${args[@]}" 2>/dev/null || true)
    if [[ -z "${commit}" ]]; then
        if [[ "${label}" == "good" ]]; then
            log_error "Cannot auto-detect good commit."
            log_warn "This can happen when running locally without origin/main."
            log_warn "Use --good <commit> to specify the known good vllm commit."
        else
            log_error "Cannot auto-detect bad commit."
            log_warn "Use --bad <commit> to specify the known bad vllm commit."
        fi
        exit 1
    fi
    printf -v "${target_var}" '%s' "${commit}"
    log_ok "${label^} commit${ref:+ (${ref})}: ${commit}"
}

detect_commits() {
    local yaml_path="${ASCEND_REPO}/.github/workflows/pr_test_light.yaml"
    if [[ -z "${GOOD_COMMIT}" ]]; then
        log_info "Auto-detecting good commit from origin/main..."
        detect_commit_from_yaml GOOD_COMMIT "good" "${yaml_path}" "origin/main"
    fi
    if [[ -z "${BAD_COMMIT}" ]]; then
        log_info "Auto-detecting bad commit from current branch..."
        detect_commit_from_yaml BAD_COMMIT "bad" "${yaml_path}"
    fi
}

# Resolve branch/tag/HEAD~N inputs to concrete commits and validate reachability.
resolve_commit() {
    local ref="$1" label="$2" resolved obj_type
    resolved=$(git -C "${VLLM_REPO}" rev-parse --verify "${ref}^{commit}" 2>/dev/null || true)
    [[ -z "${resolved}" ]] && resolved=$(git -C "${VLLM_REPO}" rev-parse --verify "origin/${ref}^{commit}" 2>/dev/null || true)
    if [[ -z "${resolved}" ]]; then
        log_error "Cannot resolve ${label} commit '${ref}'."
        log_warn "Possible causes:"
        log_warn "  - The commit hash is incorrect"
        log_warn "  - The commit is not present in local history"
        log_warn "  - Try --fetch-depth 500, or --fetch-depth 0 for full history"
        return 1
    fi
    obj_type=$(git -C "${VLLM_REPO}" cat-file -t "${resolved}" 2>/dev/null || true)
    [[ "${obj_type}" == "commit" ]] || die "'${ref}' resolves to a ${obj_type:-unknown} object, not a commit."
    echo "${resolved}"
}

fetch_vllm_history() {
    if [[ -z "${FETCH_DEPTH}" ]]; then
        log_info "Skipping git fetch (local history only)"
        return
    fi
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
}

ensure_clean_vllm_repo() {
    local git_dir
    git_dir=$(git -C "${VLLM_REPO}" rev-parse --git-dir 2>/dev/null) \
        || die "vllm repo is not a valid git repository: ${VLLM_REPO}"
    if [[ -d "${git_dir}/rebase-merge" || -d "${git_dir}/rebase-apply" ]]; then
        die "vllm repo has an in-progress rebase. Abort or finish it before bisecting."
    fi
    if [[ -n "$(git -C "${VLLM_REPO}" diff --name-only --diff-filter=U 2>/dev/null || true)" ]]; then
        die "vllm repo has uncommitted or unmerged changes. Clean the worktree before bisecting."
    fi
    if [[ -n "$(git -C "${VLLM_REPO}" status --porcelain 2>/dev/null || true)" ]]; then
        die "vllm repo has uncommitted or unmerged changes. Clean the worktree before bisecting."
    fi
}

prepare_vllm_repo() {
    log_step "Preparing vllm repo for bisect"
    ensure_clean_vllm_repo
    fetch_vllm_history
    if ! GOOD_COMMIT=$(resolve_commit "${GOOD_COMMIT}" "good"); then
        exit 1
    fi
    if ! BAD_COMMIT=$(resolve_commit "${BAD_COMMIT}" "bad"); then
        exit 1
    fi
    log_ok "Good commit resolved: ${GOOD_COMMIT}"
    log_ok "Bad commit resolved:  ${BAD_COMMIT}"
    [[ "${GOOD_COMMIT}" != "${BAD_COMMIT}" ]] || die "Good and bad commits are the same (${GOOD_COMMIT:0:12}). Nothing to bisect."

    local commit_count
    commit_count=$(git -C "${VLLM_REPO}" rev-list --count "${GOOD_COMMIT}..${BAD_COMMIT}" 2>/dev/null || echo "0")
    if [[ "${commit_count}" == "0" ]]; then
        log_error "No commits found between good (${GOOD_COMMIT:0:12}) and bad (${BAD_COMMIT:0:12})."
        log_error "Either the range is empty, or good is not an ancestor of bad."
        log_warn "Check that --good is older than --bad, and fetch more history with --fetch-depth if needed."
        exit 1
    fi
    log_info "Commits in range: ${commit_count}"
}

setup_files_changed() {
    local commit="$1" prev_commit="$2" changed
    changed=$(git -C "${VLLM_REPO}" diff --name-only "${prev_commit}" "${commit}" -- \
        setup.py pyproject.toml setup.cfg requirements*.txt 2>/dev/null || true)
    [[ -n "${changed}" ]]
}

# Pytest output stays attached to the terminal. The verdict is written to a file
# so callers can inspect it without swallowing test output.
run_test() {
    local exit_code=0 saved_dir full_cmd
    saved_dir="$(pwd)"
    cd "${ASCEND_REPO}"
    full_cmd="${TEST_CMD}"
    [[ -n "${EXTRA_ENV}" ]] && full_cmd="${EXTRA_ENV} ${full_cmd}"
    log_info "Running: ${full_cmd}"
    timeout "${STEP_TIMEOUT}" bash -c "${full_cmd}" || exit_code=$?
    cd "${saved_dir}"

    case "${exit_code}" in
        0) echo "good" > "${BISECT_VERDICT_FILE}" ;;
        124)
            log_warn "Test timed out after ${STEP_TIMEOUT}s"
            echo "skip" > "${BISECT_VERDICT_FILE}"
            ;;
        *) echo "bad" > "${BISECT_VERDICT_FILE}" ;;
    esac
}

get_vllm_install_cmd() {
    python3 "${HELPER_SCRIPT}" vllm-install --test-cmd "${TEST_CMD}" 2>/dev/null || true
}

get_ascend_install_cmd() {
    python3 "${HELPER_SCRIPT}" ascend-install --test-cmd "${TEST_CMD}" 2>/dev/null || true
}

fix_triton_install() {
    python3 -m pip uninstall -y triton triton-ascend >/dev/null 2>&1 || true
    python3 -m pip install -q triton-ascend==3.2.0
}

print_install_failure_log() {
    local label="$1" log_file="$2"
    log_warn "${label} log tail:"
    if [[ -f "${log_file}" ]]; then
        tail -n 100 "${log_file}" >&2 || true
    else
        log_warn "No reinstall log captured at ${log_file}"
    fi
}

_append_log() {
    local step="$1" commit="$2" result="$3"
    python3 -c "import json; p='${BISECT_LOG_FILE}'; log=json.load(open(p)); log.append({'step': ${step}, 'commit': '${commit}', 'result': '${result}'}); json.dump(log, open(p, 'w'))"
}

skip_current_commit() {
    local step="$1" short_commit="$2"
    git -C "${VLLM_REPO}" bisect skip 2>/dev/null || true
    _append_log "${step}" "${short_commit}" "skip"
    SKIPPED_COMMITS="${SKIPPED_COMMITS:+${SKIPPED_COMMITS},}${short_commit}"
}

_generate_report() {
    local first_bad="$1" total_steps="$2" skipped="$3"
    local first_bad_info_file="/tmp/bisect_first_bad_info.txt"
    git -C "${VLLM_REPO}" log -1 --format="commit %H%nAuthor: %an <%ae>%nDate:   %ad%n%n    %s%n%n    %b" \
        "${first_bad}" > "${first_bad_info_file}" 2>/dev/null || echo "N/A" > "${first_bad_info_file}"

    local total_commits reported_cmd="${TEST_CMD}"
    total_commits=$(git -C "${VLLM_REPO}" rev-list --count "${GOOD_COMMIT}..${BAD_COMMIT}" 2>/dev/null || echo "0")
    [[ -n "${EXTRA_ENV}" ]] && reported_cmd="${EXTRA_ENV} ${reported_cmd}"

    python3 "${HELPER_SCRIPT}" report \
        --good-commit "${GOOD_COMMIT}" \
        --bad-commit "${BAD_COMMIT}" \
        --first-bad "${first_bad}" \
        --first-bad-info-file "${first_bad_info_file}" \
        --test-cmd "${reported_cmd}" \
        --total-steps "${total_steps}" \
        --total-commits "${total_commits}" \
        ${skipped:+--skipped "${skipped}"} \
        --log-file "${BISECT_LOG_FILE}" \
        --summary-output "${SUMMARY_OUTPUT}"
}

# Main bisect loop. Keep the per-step logs explicit so CI output stays readable.
run_bisect() {
    log_step "Starting bisect: good=${GOOD_COMMIT:0:12} bad=${BAD_COMMIT:0:12}"
    git -C "${VLLM_REPO}" bisect start "${BAD_COMMIT}" "${GOOD_COMMIT}" --no-checkout 2>/dev/null \
        || die "Failed to initialize git bisect. Check that good/bad commits are valid."

    local step=0 prev_commit="${GOOD_COMMIT}" start_time current_commit short_commit commit_msg
    local result bisect_output bisect_rc first_bad
    start_time=$(date +%s)
    SKIPPED_COMMITS=""
    echo "[]" > "${BISECT_LOG_FILE}"

    while true; do
        step=$((step + 1))
        if [[ $(( $(date +%s) - start_time )) -ge ${TOTAL_TIMEOUT} ]]; then
            log_error "Total timeout (${TOTAL_TIMEOUT}s) reached after ${step} steps"
            reset_bisect
            exit 1
        fi

        current_commit=$(git -C "${VLLM_REPO}" rev-parse BISECT_HEAD 2>/dev/null || true)
        [[ -z "${current_commit}" ]] && { log_info "Bisect completed (no more BISECT_HEAD)"; break; }

        short_commit="${current_commit:0:12}"
        commit_msg=$(git -C "${VLLM_REPO}" log -1 --format="%s" "${current_commit}" 2>/dev/null || echo "(unable to read commit message)")
        echo ""
        log_step "Step ${step}: testing commit ${short_commit} - ${commit_msg}"
        echo ""

        if ! git -C "${VLLM_REPO}" checkout "${current_commit}" --quiet 2>/dev/null; then
            log_warn "Failed to checkout ${short_commit}, skipping"
            skip_current_commit "${step}" "${short_commit}"
            prev_commit="${current_commit}"
            continue
        fi

        if [[ "${FORCE_REINSTALL}" == "true" ]] || setup_files_changed "${current_commit}" "${prev_commit}"; then
            local saved_dir vllm_install_cmd ascend_install_cmd
            local vllm_reinstall_log="/tmp/bisect_vllm_reinstall_${short_commit}.log"
            local ascend_reinstall_log="/tmp/bisect_ascend_reinstall_${short_commit}.log"
            log_warn "Setup files changed, reinstalling vllm..."
            vllm_install_cmd="$(get_vllm_install_cmd)"
            ascend_install_cmd="$(get_ascend_install_cmd)"
            FORCE_REINSTALL=false
            if [[ -z "${vllm_install_cmd}" || -z "${ascend_install_cmd}" ]]; then
                log_warn "Failed to determine reinstall commands for ${short_commit}, skipping"
                FORCE_REINSTALL=true
                skip_current_commit "${step}" "${short_commit}"
                prev_commit="${current_commit}"
                continue
            fi
            saved_dir="$(pwd)"
            python3 -m pip uninstall -y vllm vllm_ascend>/dev/null 2>&1 || true
            cd "${VLLM_REPO}"
            bash -lc "${vllm_install_cmd}" >"${vllm_reinstall_log}" 2>&1 || {
                log_warn "vllm reinstall failed for ${short_commit}, skipping"
                print_install_failure_log "vllm reinstall" "${vllm_reinstall_log}"
                FORCE_REINSTALL=true
                cd "${saved_dir}"
                skip_current_commit "${step}" "${short_commit}"
                prev_commit="${current_commit}"
                continue
            }
            cd "${ASCEND_REPO}"
            bash -lc "${ascend_install_cmd}" >"${ascend_reinstall_log}" 2>&1 || {
                log_warn "vllm-ascend reinstall failed for ${short_commit}, skipping"
                print_install_failure_log "vllm-ascend reinstall" "${ascend_reinstall_log}"
                FORCE_REINSTALL=true
                cd "${saved_dir}"
                skip_current_commit "${step}" "${short_commit}"
                prev_commit="${current_commit}"
                continue
            }
            fix_triton_install >"${ascend_reinstall_log}.triton" 2>&1 || {
                log_warn "triton fixup failed for ${short_commit}, skipping"
                print_install_failure_log "triton fixup" "${ascend_reinstall_log}.triton"
                FORCE_REINSTALL=true
                cd "${saved_dir}"
                skip_current_commit "${step}" "${short_commit}"
                prev_commit="${current_commit}"
                continue
            }
            cd "${saved_dir}"
        fi

        run_test
        result=$(cat "${BISECT_VERDICT_FILE}" 2>/dev/null || echo "bad")
        echo ""
        log_info "Step ${step}: commit ${short_commit} → ${result}"

        bisect_rc=0
        case "${result}" in
            good) bisect_output=$(git -C "${VLLM_REPO}" bisect good 2>&1) || bisect_rc=$? ;;
            bad)  bisect_output=$(git -C "${VLLM_REPO}" bisect bad 2>&1) || bisect_rc=$? ;;
            skip)
                bisect_output=$(git -C "${VLLM_REPO}" bisect skip 2>&1) || bisect_rc=$?
                SKIPPED_COMMITS="${SKIPPED_COMMITS:+${SKIPPED_COMMITS},}${short_commit}"
                ;;
            *)
                log_error "Unexpected test result: '${result}' for commit ${short_commit}"
                reset_bisect
                return 1
                ;;
        esac

        _append_log "${step}" "${short_commit}" "${result}"

        if echo "${bisect_output}" | grep -q "is the first bad commit"; then
            echo ""
            log_ok "===== Bisect completed! ====="
            echo "${bisect_output}"
            first_bad=$(echo "${bisect_output}" | head -1 | awk '{print $1}')
            _generate_report "${first_bad}" "${step}" "${SKIPPED_COMMITS}"
            reset_bisect
            return 0
        fi

        if [[ ${bisect_rc} -ne 0 ]]; then
            log_error "git bisect ${result} failed with exit code ${bisect_rc}:"
            echo "${bisect_output}"
            reset_bisect
            return 1
        fi

        prev_commit="${current_commit}"
    done

    log_error "Bisect did not converge. Check the commit range and test command."
    reset_bisect
    return 1
}

run_single_bisect() {
    log_step "vllm bisect automation"
    log_info "Test command: ${TEST_CMD}"
    [[ -n "${EXTRA_ENV}" ]] && log_info "Extra env:    ${EXTRA_ENV}"
    detect_vllm_repo
    detect_ascend_repo
    detect_commits
    prepare_vllm_repo
    run_bisect
}

# Batch mode reuses the same single-command flow so local and CI behavior stay aligned.
run_batch_bisect() {
    local cmds_content passed=0 failed=0 idx=0 cmd
    cmds_content=$(cat "${TEST_CMDS_FILE}")
    IFS=';' read -ra CMD_ARRAY <<< "${cmds_content}"
    local total=${#CMD_ARRAY[@]}

    log_step "Batch bisect: ${total} test command(s)"
    for cmd in "${CMD_ARRAY[@]}"; do
        cmd=$(echo "${cmd}" | xargs)
        [[ -z "${cmd}" ]] && continue
        idx=$((idx + 1))
        echo ""
        log_step "========== [${idx}/${total}] =========="
        log_info "Test command: ${cmd}"
        TEST_CMD="${cmd}"
        if run_single_bisect; then
            log_ok "[${idx}/${total}] Bisect completed successfully"
            passed=$((passed + 1))
        else
            log_error "[${idx}/${total}] Bisect failed"
            failed=$((failed + 1))
        fi
    done

    echo ""
    log_step "Batch bisect summary: ${passed} passed, ${failed} failed out of ${total}"
    [[ ${failed} -eq 0 ]]
}

main() {
    parse_args "$@"
    if [[ -n "${TEST_CMDS_FILE}" ]]; then
        run_batch_bisect
    else
        run_single_bisect
    fi
}

main "$@"
