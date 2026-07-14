#!/bin/bash
set -x
set -euo pipefail

MIN_FREE_GB=3

get_free_gb() {
    local free_kb
    free_kb=$(df -P / | awk 'NR>1 {print $4}')
    echo $(( free_kb / 1024 / 1024 ))
}

FREE_GB=$(get_free_gb)
if [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
    rm -rf /tmp/shiqiang/* || true
    rm -rf /tmp/* /var/tmp/* || true
    ccache -C || true
    find /opt/agent_* -path "*/workspace/j_*" -mtime +1 -delete || true
    if command -v docker &>/dev/null; then
        docker system prune -af || true
    fi

    FREE_GB_AFTER=$(get_free_gb)
    if [ "${FREE_GB_AFTER}" -lt "${MIN_FREE_GB}" ]; then
        exit 0
    fi
fi

# Global config
API_PREFIX="https://174e1b821a8446f38998a67186ba766e.apic.cn-southwest-2.huaweicloudapis.com/aurogon_service"
MR_THIRD_ID="9270"
#MR_THIRD_ID="${PR_NUMBER:-}"
NETWORK_ZONE=github
PROJECT_PATH=vllm-project/vllm-ascend
BIND_ID_API2=11
PAGE_CURR=1
PAGE_SIZE=100

if [ -z "${MR_THIRD_ID}" ]; then
    echo "ERROR: PR_NUMBER is empty. It must be provided via the pull_request event or workflow_dispatch input."
    exit 1
fi
echo "Using current PR number as MR_THIRD_ID: ${MR_THIRD_ID}"

X_APIG_APPCODE="${X_APIG_APPCODE:-}"
APP_KEY="${APP_KEY:-}"
APP_SECRET="${APP_SECRET:-}"

miss=()
[[ -z "${X_APIG_APPCODE}" ]] && miss+=("X_APIG_APPCODE")
[[ -z "${APP_KEY}" ]] && miss+=("APP_KEY")
[[ -z "${APP_SECRET}" ]] && miss+=("APP_SECRET")
if [[ ${#miss[@]} -gt 0 ]]; then
    echo "FATAL ERROR: Missing auth env vars: ${miss[*]}"
    exit 98
fi

# Api1 save mr with retry
MAX_RETRY=3
retry_cnt=0
SAVE_MR_RET=""
curl_ret=0

while [ ${retry_cnt} -lt ${MAX_RETRY} ]; do
    JSON_BODY=$(printf '{"mrThirdId":%d,"networkZone":"%s","projectPath":"%s"}' "${MR_THIRD_ID}" "${NETWORK_ZONE}" "${PROJECT_PATH}")
    SAVE_MR_RET=$(curl -s --max-time 60 -X POST \
    -H "Accept:*/*" \
    -H "Content-Type:application/json" \
    -H "X-Apig-AppCode: ${X_APIG_APPCODE}" \
    -H "AppKey: ${APP_KEY}" \
    -H "AppSecret: ${APP_SECRET}" \
    -d "${JSON_BODY}" \
    "${API_PREFIX}/third-platform/save-mr") || curl_ret=$?

    if [[ ${curl_ret} -ne 0 || -z "${SAVE_MR_RET}" ]];then
        retry_cnt=$((retry_cnt+1))
        echo "WARN: save mr curl failed, retry ${retry_cnt}/${MAX_RETRY}"
        sleep 2
        continue
    fi

    BUS_SUCCESS=$(echo "${SAVE_MR_RET}" | grep -o '"success":[a-z]*' | cut -d: -f2)
    BUS_CODE=$(echo "${SAVE_MR_RET}" | grep -o '"code":[0-9]*' | cut -d: -f2)
    if [[ "${BUS_CODE}" == "500" ]];then
        retry_cnt=$((retry_cnt+1))
        echo "WARN: server busy code 500, retry ${retry_cnt}/${MAX_RETRY}"
        sleep 2
        continue
    fi
    break
done

if [[ ${retry_cnt} -ge ${MAX_RETRY} ]];then
    echo "ERROR: save mr api max retry reach, resp:${SAVE_MR_RET}"
    exit 10
fi

# Check business success
if [[ "${BUS_SUCCESS}" != "true" ]];then
    echo "ERROR: save mr api business fail, code:${BUS_CODE}, resp:${SAVE_MR_RET}"
    exit 11
fi

# Extract requirement id
REQ_ID=$(echo "${SAVE_MR_RET}" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)
if [[ -z "${REQ_ID}" || "${REQ_ID}" == "null" ]];then
    echo "ERROR: no valid data.id from api"
    exit 12
fi

# Api2 case recommend
echo -e '\n===== Call case recommend api ====='
curl_ret2=0
CASE_JSON=$(printf '{"current":%d,"size":%d,"bindId":%d,"bindType":"version","requirementList":"%s","requirementType":"MR","all":true}' \
"${PAGE_CURR}" "${PAGE_SIZE}" "${BIND_ID_API2}" "${REQ_ID}")

CASE_RET=$(curl -s --max-time 60 -X POST \
-H "Accept:*/*" \
-H "Content-Type:application/json" \
-H "X-Apig-AppCode: ${X_APIG_APPCODE}" \
-H "AppKey: ${APP_KEY}" \
-H "AppSecret: ${APP_SECRET}" \
-d "${CASE_JSON}" \
"${API_PREFIX}/case_recommend/commend_by_mr") || curl_ret2=$?

echo "result:${CASE_RET}"
if [[ ${curl_ret2} -ne 0 ]];then
    echo "ERROR: case recommend api curl failed, code:${curl_ret2}"
    exit 13
fi

BUS_SUCCESS2=$(jq -r '.success' <<<"${CASE_RET}")
BUS_CODE2=$(jq -r '.code' <<<"${CASE_RET}")
if [[ "${BUS_SUCCESS2}" != "true" ]];then
    echo "ERROR: case recommend api business fail, code:${BUS_CODE2}, resp:${CASE_RET}"
    exit 15
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTEST_LIST_FILE="${SCRIPT_DIR}/recommended_pytest_paths.txt"

TMP_JSON=$(mktemp)
echo "${CASE_RET}" > "${TMP_JSON}"

python3 <<'PY' "${TMP_JSON}" > "${PYTEST_LIST_FILE}"
import json
import sys

def name_to_pytest_target(name: str) -> str:
    name = name.strip()
    if not name:
        return ""

    if "--" in name:
        file_part, test_func = name.split("--", 1)
        file_path = file_part.replace("__", "/")
        if not file_path.endswith(".py"):
            file_path += ".py"
        return f"{file_path}::{test_func}"

    file_path = name.replace("__", "/")
    if not file_path.endswith(".py"):
        file_path += ".py"
    return file_path

json_path = sys.argv[1]
with open(json_path, "r", encoding="utf-8") as f:
    raw = f.read()

if not raw.strip():
    print("ERROR: temp json file empty", file=sys.stderr)
    sys.exit(1)

resp = json.loads(raw)
items = resp.get("data") or []

seen = set()
targets = []
for item in items:
    if not isinstance(item, dict):
        continue
    target = name_to_pytest_target(item.get("name", ""))
    if target and target not in seen:
        seen.add(target)
        targets.append(target)

if not targets:
    print("INFO: case recommend returned no targets, writing empty list", file=sys.stderr)
else:
    print("\n".join(targets))
PY

rm -f "${TMP_JSON}"

if [ $? -ne 0 ]; then
    echo "ERROR: failed to parse case recommend response"
    exit 14
fi

echo "===== Recommended pytest paths ====="
cat "${PYTEST_LIST_FILE}"
echo "Total: $(wc -l < "${PYTEST_LIST_FILE}")"
