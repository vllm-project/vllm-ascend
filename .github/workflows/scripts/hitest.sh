#!/bin/bash
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

MR_THIRD_ID=11839
NETWORK_ZONE=github
PROJECT_PATH=vllm-project/vllm-ascend
BIND_ID_API2=11
PAGE_CURR=1
PAGE_SIZE=100

# Global config
# API_PREFIX="https://174e1b821a8446f38998a67186ba766e.apic.cn-southwest-2.huaweicloudapis.com/aurogon_service"
# MR_THIRD_ID=826
# NETWORK_ZONE=gitcode
# PROJECT_PATH=Ascend/MindIE-LLM
# BIND_ID_API2=16
# PAGE_CURR=1
# PAGE_SIZE=100

# APIG auth 硬编码，无需外部传入环境变量
X_APIG_APPCODE="2a934292a6ab4dc08b99d6304794a25443f724c21ab64082a7c168450e4bb883"
APP_KEY="88df727ac9bb4d058c0e81bee9852c24"
APP_SECRET="b049f8c7cc73485ca9912d1acd8b6e79"

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
    # 500服务器繁忙则重试
    if [[ "${BUS_CODE}" == "500" ]];then
        retry_cnt=$((retry_cnt+1))
        echo "WARN: server busy code 500, retry ${retry_cnt}/${MAX_RETRY}"
        sleep 2
        continue
    fi
    break
done

# 重试耗尽仍失败
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
# 使用printf生成合法标准JSON
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

if [[ ${curl_ret2} -ne 0 ]];then
    echo "ERROR: case recommend api curl failed, code:${curl_ret2}"
    exit 13
fi

echo "result:${CASE_RET}"

# 可选：增加业务code判断，和第一个接口逻辑对齐
BUS_SUCCESS2=$(echo "${CASE_RET}" | grep -o '"success":[a-z]*' | cut -d: -f2)
BUS_CODE2=$(echo "${CASE_RET}" | grep -o '"code":[0-9]*' | cut -d: -f2)
if [[ "${BUS_SUCCESS2}" != "true" ]];then
    echo "WARN: case recommend api business fail, code:${BUS_CODE2}, skip ut"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTEST_LIST_FILE="${SCRIPT_DIR}/recommended_pytest_paths.txt"

# Api2 成功后：提取 name 并直接转换
CASE_RET="${CASE_RET}" python3 <<'PY' > "${PYTEST_LIST_FILE}"
import json
import os
import sys

def name_to_pytest_target(name: str) -> str:
    name = name.strip()
    if not name:
        return ""

    # 情况2: file--testcase → file.py::testcase
    if "--" in name:
        file_part, test_func = name.split("--", 1)
        file_path = file_part.replace("__", "/")
        if not file_path.endswith(".py"):
            file_path += ".py"
        return f"{file_path}::{test_func}"

    # 情况1: 纯文件 → file.py
    file_path = name.replace("__", "/")
    if not file_path.endswith(".py"):
        file_path += ".py"
    return file_path

case_ret = os.environ.get("CASE_RET", "")
if not case_ret:
    print("ERROR: CASE_RET is empty", file=sys.stderr)
    sys.exit(1)

resp = json.loads(case_ret)
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
    print("ERROR: no pytest targets found", file=sys.stderr)
    sys.exit(1)

print("\n".join(targets))
PY

if [ $? -ne 0 ]; then
    echo "ERROR: failed to parse case recommend response"
    exit 14
fi

echo "===== Recommended pytest paths ====="
cat "${PYTEST_LIST_FILE}"
echo "Total: $(wc -l < "${PYTEST_LIST_FILE}")"
