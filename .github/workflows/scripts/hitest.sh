#!/bin/bash
set -euo pipefail

MIN_FREE_GB=3

# 从环境变量读取，兜底空值防止unbound variable
API_PREFIX="${AUROGON_API_PREFIX:-}"
X_APIG_APPCODE="${X_APIG_APPCODE:-}"
APP_KEY="${APP_KEY:-}"
APP_SECRET="${APP_SECRET:-}"

# 前置鉴权校验
if [[ -z "${API_PREFIX}" || -z "${X_APIG_APPCODE}" || -z "${APP_KEY}" || -z "${APP_SECRET}" ]]; then
    echo "ERROR: Missing required aurogon apig environment variables!"
    exit 98
fi

# 后续原有逻辑不变

# ===================== 工具函数 =====================
# 获取根目录空闲GB，增加空值容错
get_free_gb() {
    local free_kb
    free_kb=$(df -P / 2>/dev/null | awk 'NR>1 {print $4}')
    if [[ -z "${free_kb}" || ! "${free_kb}" =~ ^[0-9]+$ ]]; then
        echo 0
        return
    fi
    echo $(( free_kb / 1024 / 1024 ))
}

# JSON安全打印日志
print_json() {
    local json="$1"
    echo "=== API RESPONSE ==="
    echo "${json}" | jq . 2>/dev/null || echo "${json}"
    echo "===================="
}

# ===================== 前置校验 =====================
# 1. 校验jq工具存在
if ! command -v jq &>/dev/null; then
    echo "ERROR: jq not found, please install jq first!"
    exit 99
fi

# 2. 校验鉴权环境变量不为空
if [[ -z "${X_APIG_APPCODE}" || -z "${APP_KEY}" || -z "${APP_SECRET}" ]]; then
    echo "ERROR: Missing APIG auth env variables!"
    exit 98
fi

# ===================== 磁盘清理 =====================
FREE_GB=$(get_free_gb)
echo "Current disk free: ${FREE_GB}GB, threshold: ${MIN_FREE_GB}GB"
if [[ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]]; then
    echo "Disk space insufficient, start cleanup..."
    rm -rf /tmp/shiqiang/* || true
    rm -rf /tmp/* /var/tmp/* || true
    ccache -C || true
    find /opt/agent_* -path "*/workspace/j_*" -mtime +1 -delete || true
    if command -v docker &>/dev/null; then
        docker system prune -af || true
    fi

    FREE_GB_AFTER=$(get_free_gb)
    echo "After cleanup disk free: ${FREE_GB_AFTER}GB"
    if [[ "${FREE_GB_AFTER}" -lt "${MIN_FREE_GB}" ]]; then
        echo "ERROR: Disk still insufficient after cleanup, abort pipeline"
        exit 97
    fi
fi

# ===================== 接口1：save-mr 上报MR =====================
retry_cnt=0
SAVE_MR_RET=""
while [[ ${retry_cnt} -lt ${MAX_RETRY} ]]; do
    # 标准JSON生成
    JSON_BODY=$(printf '{"mrThirdId":%d,"networkZone":"%s","projectPath":"%s"}' \
        "${MR_THIRD_ID}" "${NETWORK_ZONE}" "${PROJECT_PATH}")
    echo "Call save-mr api, body: ${JSON_BODY}"

    # 执行curl，捕获完整输出，屏蔽set -e临时失败
    set +e
    SAVE_MR_RET=$(curl -s --max-time 60 -X POST \
        -H "Accept: */*" \
        -H "Content-Type: application/json" \
        -H "X-Apig-AppCode:${X_APIG_APPCODE}" \
        -H "AppKey:${APP_KEY}" \
        -H "AppSecret:${APP_SECRET}" \
        -d "${JSON_BODY}" \
        "${API_PREFIX}/third-platform/save-mr")
    curl_ret=$?
    set -e

    # curl网络异常重试
    if [[ ${curl_ret} -ne 0 || -z "${SAVE_MR_RET}" ]]; then
        retry_cnt=$((retry_cnt + 1))
        echo "WARN: save-mr curl failed(ret:${curl_ret}), retry ${retry_cnt}/${MAX_RETRY}"
        sleep 2
        continue
    fi

    # jq解析JSON，稳定提取字段
    BUS_SUCCESS=$(echo "${SAVE_MR_RET}" | jq -r '.success')
    BUS_CODE=$(echo "${SAVE_MR_RET}" | jq -r '.code')
    # 500服务繁忙重试
    if [[ "${BUS_CODE}" == "500" ]]; then
        retry_cnt=$((retry_cnt + 1))
        echo "WARN: server busy(code 500), retry ${retry_cnt}/${MAX_RETRY}"
        sleep 2
        continue
    fi
    break
done

# 重试耗尽失败
if [[ ${retry_cnt} -ge ${MAX_RETRY} ]]; then
    echo "ERROR: save-mr api max retry exhausted"
    print_json "${SAVE_MR_RET}"
    exit 10
fi

# 业务返回失败
if [[ "${BUS_SUCCESS}" != "true" ]]; then
    echo "ERROR: save-mr business fail, code:${BUS_CODE}"
    print_json "${SAVE_MR_RET}"
    exit 11
fi

# 提取需求ID
REQ_ID=$(echo "${SAVE_MR_RET}" | jq -r '.id')
if [[ -z "${REQ_ID}" || "${REQ_ID}" == "null" ]]; then
    echo "ERROR: Cannot extract valid id from save-mr response"
    print_json "${SAVE_MR_RET}"
    exit 12
fi
echo "Get requirement id: ${REQ_ID}"

# ===================== 接口2：推荐测试用例（修复拼写+数组格式） =====================
echo -e '\n===== Call case recommend api ====='
# requirementList 传JSON数组字符串，修复接口路径拼写错误
CASE_JSON=$(printf '{"current":%d,"size":%d,"bindId":%d,"bindType":"version","requirementList":"[\"%s\"]","requirementType":"MR","all":true}' \
    "${PAGE_CURR}" "${PAGE_SIZE}" "${BIND_ID_API2}" "${REQ_ID}")
echo "Request body: ${CASE_JSON}"

set +e
CASE_RET=$(curl -s --max-time 60 -X POST \
    -H "Accept: */*" \
    -H "Content-Type: application/json" \
    -H "X-Apig-AppCode:${X_APIG_APPCODE}" \
    -H "AppKey:${APP_KEY}" \
    -H "AppSecret:${APP_SECRET}" \
    -d "${CASE_JSON}" \
    "${API_PREFIX}/case_recommend/recommend_by_mr") # 修复commend拼写错误
curl_ret2=$?
set -e

print_json "${CASE_RET}"
if [[ ${curl_ret2} -ne 0 ]]; then
    echo "ERROR: case recommend curl failed, ret code:${curl_ret2}"
    exit 13
fi

BUS_SUCCESS2=$(echo "${CASE_RET}" | jq -r '.success')
BUS_CODE2=$(echo "${CASE_RET}" | jq -r '.code')
if [[ "${BUS_SUCCESS2}" != "true" ]]; then
    echo "WARN: case recommend business fail(code:${BUS_CODE2}), skip UT execution"
    exit 0
fi

echo "All api invoke success, pipeline continue"
exit 0
