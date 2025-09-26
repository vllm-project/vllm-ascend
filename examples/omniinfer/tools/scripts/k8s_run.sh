# Script Name: k8s_run.sh
# User Manual:
# - run time
#     run in docker after docker env is ready
# - Constraints
# 	1) Script Path in Container: The script k8s_run.sh must be located in the third-level directory of the code repository, specifically under omniinfer/tools/scripts/:
# 	    --omni_infer
# 		  |
# 		  --tools
# 		      |
# 		      --scripts
# 			      |
# 				  --k8s_run.sh
# 	2) Required Environment Variables (must be visible inside the container):
# 		# ENV_IP_LIST="[10.10.10.1,10.10.10.2,10.10.10.3,10.10.10.4,10.10.10.5,10.10.10.6,10.10.10.7,10.10.10.8]" # 
# 		# ENV_LOCAL_IP="10.10.10.7" # 
# 		# ENV_SOCKET_NAME="enp23s0f3"
# 		# ENV_NFS_PATH="/dev/nfs"
# 		# ENV_MODEL_PATH="/data/models/deepseek-r1"
# 		# ENV_LOG_PATH="/data/omni/log"
# - Script Permissions
#     Run the script with: bash 1920_run.sh >./run.log 2>&1 &;
	
# - Log
#     Script Logs: If executed as above, monitor logs via:tail -f run.log
# 	Omni Service Logs: tail -f $ENV_LOG_PATH/server_0.log, To check logs for different ranks (e.g., server_0 to server_15)
	
# - Curl Command
# 	IP: The first IP in ENV_IP_LIST.
# 	Port: 7556.


#!/bin/bash
set -euo pipefail

# 0. global var
ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
MAX_MODEL_LEN_P=30000
MAX_MODEL_LEN_D=2048
NODE_PORT_BASE=8556
API_PORT_BASE=9556
PROXY_PORT=7556
GPU_UTIL_P=0.92
GPU_UTIL_D=0.92
ADDITIONAL_CONFIG_P='{"enable_graph_mode":false}'
ADDITIONAL_CONFIG_D='{"enable_graph_mode": true, "use_cached_npu_graph": false}'
EXTRA_ARGS_P="--max-num-batched-tokens 30000 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 16"
EXTRA_ARGS_D="--enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 32"
SCRIPT_PATH=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)
CODE_PATH=$(cd "$(dirname "$(dirname "$SCRIPT_PATH")")" && pwd)

# 1. tmp var
IP_LIST=""
LOCAL_IP=""
NODE_INDEX=-1
NODE_NAME="default_name"
NODE_ROLE="default"
PREFILL_SERVER_LIST=()
DECODE_SERVER_LIST=()
RANKTABLE_SAVE_PATH=""
MODEL_PATH=""
LOG_PATH=""
SOCKET_NAME=""
TIME_OUT="300s"
APP_TIME_OUT="1200s"


run_server() {
    if [ -z "${DECODE_SERVER_LIST}" ]; then
        echo "error: no decode node" >&2
        exit 3
    fi
	
	echo "kill vllm on this machine"
	(pkill -9 -f "vllm" &)
    sleep 2

    echo "D list:"
    echo "${DECODE_SERVER_LIST[@]}"
    STR_DECODE_LIST=$(printf "%s," "${DECODE_SERVER_LIST[@]}" | sed 's/,$//')

    P_COUNT=${#PREFILL_SERVER_LIST[@]}
    KV_PARALLEL_SIZE=$((P_COUNT + 1))
    echo "KV parallel size: $KV_PARALLEL_SIZE"
    KV_RANK=$NODE_INDEX
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    case "${NODE_ROLE}" in
        prefill) 
            echo "--in prefill, wait 30s--"
            sleep 30
		    NODE_PORT=${NODE_PORT_BASE}
			API_PORT=$((API_PORT_BASE+NODE_INDEX))

            echo "node: prefill_$KV_RANK"
            echo "kv_rank: $KV_RANK"

            MODEL_EXTRA_CFG_PATH="${CODE_PATH}/tests/test_config/test_config_prefill.json"
            prefill_server_list=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | awk '$1=$1' | tr -d ',')
            LOCAL_RANKTABLE_FILE=($(ls ${RANKTABLE_SAVE_PATH}/prefill/local_ranktable_*${LOCAL_IP}_$prefill_server_list.json | tr '\n' ' '))
			
			echo "LOCAL_RANKTABLE_FILE: ${LOCAL_RANKTABLE_FILE[@]}"
			echo "P_COUNT: ${P_COUNT}"
			echo "SOCKET_NAME: ${SOCKET_NAME}"
			echo "LOCAL_IP: ${LOCAL_IP}"
			echo "MAX_MODEL_LEN_P: ${MAX_MODEL_LEN_P}"
			echo "NODE_PORT: ${NODE_PORT}"
			echo "API_PORT: ${API_PORT}"
			echo "KV_RANK: ${KV_RANK}"
			echo "KV_PARALLEL_SIZE: ${KV_PARALLEL_SIZE}"
			echo "MODEL_EXTRA_CFG_PATH: ${MODEL_EXTRA_CFG_PATH}"
			echo "GPU_UTIL_P: ${GPU_UTIL_P}"
            echo "STR_DECODE_LIST: ${STR_DECODE_LIST}"
			

            cd ${CODE_PATH}/tools/scripts/
            bash ${CODE_PATH}/tools/scripts/pd_run.sh \
                --global-rank-table-path "${RANKTABLE_SAVE_PATH}/global/global_ranktable_merge.json" \
                --rank-table-path "${LOCAL_RANKTABLE_FILE}" \
                --local-decode-server-ip-list "$STR_DECODE_LIST" \
                --global-decode-server-ip-list "$STR_DECODE_LIST" \
                --prefill-pod-num "${P_COUNT}" \
                --gloo-socket-ifname "${SOCKET_NAME}" \
                --tp-socket-ifname "${SOCKET_NAME}" \
                --model-path "${MODEL_PATH}" \
                --master-ip "${LOCAL_IP}" \
                --role "prefill" \
                --kv-role "kv_producer" \
                --max-model-len "${MAX_MODEL_LEN_P}" \
                --master-port "${NODE_PORT}" \
                --base-api-port "${API_PORT}" \
                --tp 16 \
                --ascend-rt-visible-devices "${ASCEND_RT_VISIBLE_DEVICES}" \
                --kv-rank "${KV_RANK}" \
                --kv-engine-id "${KV_RANK}" \
                --kv-parallel-size "${KV_PARALLEL_SIZE}" \
                --model-extra-cfg-path "${MODEL_EXTRA_CFG_PATH}" \
                --gpu-util "${GPU_UTIL_P}" \
                --additional-config "${ADDITIONAL_CONFIG_P}" \
                --vllm-enable-mc2 1 \
                --extra-args "${EXTRA_ARGS_P}" \
                --hccl-buffsize 200 \
                --hccl-op-expansion-mode "AIV" \
                --log-dir "${LOG_PATH}" &
            echo "--out prefill--"
            ;;
        decode) 
		    echo "--in decode--"
			NODE_PORT=${NODE_PORT_BASE}
			API_PORT=$((API_PORT_BASE+100+NODE_INDEX))
            D_COUNT=${#DECODE_SERVER_LIST[@]}
            if [ $D_COUNT -gt 1 ]; then
                LOCAL_RANKTABLE_FILE=($(ls ${RANKTABLE_SAVE_PATH}/global/local_*merge.json | tr '\n' ' '))
            else
                LOCAL_RANKTABLE_FILE=($(ls ${RANKTABLE_SAVE_PATH}/decode/local_*.json | tr '\n' ' '))
            fi

            IFS=',' read -ra arr <<< "$ASCEND_RT_VISIBLE_DEVICES"
            count=${#arr[@]}
            dp=$(( ${count} * D_COUNT ))
            echo "dp (16 * D_COUNT): $dp"
            MODEL_EXTRA_CFG_PATH="${CODE_PATH}/tests/test_config/test_config_decode.json"
            D0_IP="${DECODE_SERVER_LIST[0]}"
            SERVER_OFFSET=$(( ${count} * NODE_INDEX ))
            echo "SERVER_OFFSET: $SERVER_OFFSET" 
			echo "LOCAL_RANKTABLE_FILE: ${LOCAL_RANKTABLE_FILE[@]}"
			echo "P_COUNT: ${P_COUNT}"
			echo "SOCKET_NAME: ${SOCKET_NAME}"
			echo "dp: ${dp}"
			echo "SERVER_OFFSET: ${SERVER_OFFSET}"
			echo "D0_IP: ${D0_IP}"
			echo "MAX_MODEL_LEN_D: ${MAX_MODEL_LEN_D}"
			echo "NODE_PORT: ${NODE_PORT}"
			echo "API_PORT: ${API_PORT}"
			echo "KV_RANK: ${KV_RANK}"
			echo "KV_PARALLEL_SIZE: ${KV_PARALLEL_SIZE}"
			echo "MODEL_EXTRA_CFG_PATH: ${MODEL_EXTRA_CFG_PATH}"
			echo "GPU_UTIL_D: ${GPU_UTIL_D}"
            echo "STR_DECODE_LIST: ${STR_DECODE_LIST}"
			
			
            cd ${CODE_PATH}/tools/scripts/
            bash ${CODE_PATH}/tools/scripts/pd_run.sh \
                --global-rank-table-path "${RANKTABLE_SAVE_PATH}/global/global_ranktable_merge.json" \
                --rank-table-path "${LOCAL_RANKTABLE_FILE}" \
                --local-decode-server-ip-list "$STR_DECODE_LIST" \
                --global-decode-server-ip-list "$STR_DECODE_LIST" \
                --prefill-pod-num "${P_COUNT}" \
                --gloo-socket-ifname "${SOCKET_NAME}" \
                --tp-socket-ifname "${SOCKET_NAME}" \
                --num-servers "${count}" \
                --num-dp "${dp}" \
                --server-offset "${SERVER_OFFSET}" \
                --model-path "${MODEL_PATH}" \
                --master-ip "${D0_IP}" \
                --role "decode" \
                --kv-role "kv_consumer" \
                --max-model-len "${MAX_MODEL_LEN_D}" \
                --master-port "${NODE_PORT}" \
                --base-api-port "${API_PORT}" \
                --tp 1 \
                --kv-rank "${P_COUNT}" \
                --kv-engine-id "${P_COUNT}" \
                --kv-parallel-size "${KV_PARALLEL_SIZE}" \
                --model-extra-cfg-path "${MODEL_EXTRA_CFG_PATH}" \
                --gpu-util "${GPU_UTIL_D}" \
                --additional-config "$ADDITIONAL_CONFIG_D" \
                --vllm-enable-mc2 1 \
                --extra-args "${EXTRA_ARGS_D}" \
                --hccl-buffsize 550 \
                --hccl-op-expansion-mode "AIV" \
                --log-dir "${LOG_PATH}" &
            echo "--out decode--"
            ;;
        c) echo "local ip $LOCAL_IP is c node" ;;
    esac
}

run_proxy() {
    p_result=""
    for ((i=0; i<${#PREFILL_SERVER_LIST[@]}; i++)); do
	    ip="${PREFILL_SERVER_LIST[$i]}"
		port=$((API_PORT_BASE+i))
		if [[ -z ${p_result} ]]; then
            p_result="$ip:$port"
        else
            p_result="${p_result},$ip:$port"
        fi
	done
	
	d_result=""
    for ((i=0; i<${#DECODE_SERVER_LIST[@]}; i++)); do
		ip="${DECODE_SERVER_LIST[$i]}"
		port=$((API_PORT_BASE+100+i))
        IFS=',' read -ra arr <<< "$ASCEND_RT_VISIBLE_DEVICES"
        count=${#arr[@]}
        for ((j=0; j<${count}; j++)); do
            if [[ -z ${d_result} ]]; then
                d_result="$ip:$port"
            else
                d_result="${d_result},$ip:$port"
            fi
            port=$((port+1))
        done
	done

    echo "p_result:$p_result"
    echo "d_result:$d_result"
    bash ${CODE_PATH}/tools/scripts/global_proxy.sh \
        --listen-port "$PROXY_PORT" \
        --prefill-servers-list "$p_result" \
        --decode-servers-list "$d_result" &
    
}

check_files_exist() {
    local arr=("$@")
    echo "check arr: ${arr[@]}"
    for file in "${arr[@]}"; do
        echo "start check file:"
        echo $file
        if [[ ! -f "$file" ]]; then
            return 1
        fi
    done
    return 0
}

wait_local_ranktable_ready() {
    eval "$1"
    eval "$2"
    echo "P: ${RESULT_ENTRIES_P[@]}"
    echo "D: ${RESULT_ENTRIES_D[@]}"
    while true; do
        if check_files_exist "${RESULT_ENTRIES_P[@]}"; then
            break
        fi
        echo "Waiting for prefill local_ranktable.json to be created..."
        sleep 1
    done

    while true; do
        if check_files_exist "${RESULT_ENTRIES_D[@]}"; then
            break
        fi
        echo "Waiting for decode local_ranktable.json to be created..."
        sleep 1
    done
}

global_ranktable_generate() {
    D_COUNT=${#DECODE_SERVER_LIST[@]}

    declare -a RESULT_ENTRIES_P=()
	RANKTABLE_SUFFIX=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | awk '$1=$1' | tr -d ',')
    for ip in "${PREFILL_SERVER_LIST[@]}"; do
        entry="${RANKTABLE_SAVE_PATH}/prefill/local_ranktable_${ip}_${RANKTABLE_SUFFIX}.json"
        RESULT_ENTRIES_P+=("$entry")
    done

    declare -a RESULT_ENTRIES_D=()
    for ip in "${DECODE_SERVER_LIST[@]}"; do
        entry="${RANKTABLE_SAVE_PATH}/decode/local_ranktable_${ip}_${RANKTABLE_SUFFIX}.json"
        RESULT_ENTRIES_D+=("$entry")
    done

    PREFILL_RANKTABLE_LIST=$(IFS=,; echo "${RESULT_ENTRIES_P[*]}")
    DECODE_RANKTABLE_LIST=$(IFS=,; echo "${RESULT_ENTRIES_D[*]}")

    echo "PREFILL_RANKTABLE_LIST:"
    echo "$PREFILL_RANKTABLE_LIST"
    echo "DECODE_RANKTABLE_LIST:"
    echo "$DECODE_RANKTABLE_LIST"

    p_l=$(declare -p RESULT_ENTRIES_P)
    d_l=$(declare -p RESULT_ENTRIES_D)

    if timeout ${TIME_OUT} bash -c "$(declare -f wait_local_ranktable_ready check_files_exist); wait_local_ranktable_ready '$p_l' '$d_l'"; then
        echo "global ranktable ready."
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "error: wait_local_ranktable_ready time out"
        fi
        exit 1
    fi
    
    prefill_ranktable_list=${PREFILL_RANKTABLE_LIST}
    prefill_ranktable_list=$(echo "$prefill_ranktable_list" | awk '$1=$1' | tr ',' ' ')
    decode_ranktable_list=${DECODE_RANKTABLE_LIST}
    decode_ranktable_list=$(echo "$decode_ranktable_list" | awk '$1=$1' | tr ',' ' ')

    rm -rf ${RANKTABLE_SAVE_PATH}/global
    mkdir -p ${RANKTABLE_SAVE_PATH}/global
    if [ $D_COUNT -gt 1 ]; then
        python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
            --mode merge-local \
            --local-ranktable-list ${decode_ranktable_list} \
            --save-dir ${RANKTABLE_SAVE_PATH}/global
        
        decode_local_ranktable_merge=$(ls ${RANKTABLE_SAVE_PATH}/global/local*merge.json | tr '\n' ' ')
    else
        decode_local_ranktable_merge="${decode_ranktable_list}"
    fi

    api_server_files=$(ls ${RANKTABLE_SAVE_PATH}/prefill/local_ranktable*host.json | head -1)
    python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
        --mode merge-all \
        --api-server-list ${api_server_files} \
        --prefill-server-list ${prefill_ranktable_list} \
        --decode-server-list ${decode_local_ranktable_merge} \
        --save-dir ${RANKTABLE_SAVE_PATH}/global

    rm -f ${RANKTABLE_SAVE_PATH}/rank_dir_ready
}

local_ranktable_generate() {
    case "$NODE_ROLE" in
        prefill) 
            python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
                --mode gen \
                --prefill-server-list "${ASCEND_RT_VISIBLE_DEVICES}" \
                --api-server \
                --save-dir ${RANKTABLE_SAVE_PATH}/prefill \
                --ip ${LOCAL_IP}
            ;;
        decode) 
            python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
                --mode gen \
                --decode-server-list "${ASCEND_RT_VISIBLE_DEVICES}" \
                --save-dir ${RANKTABLE_SAVE_PATH}/decode \
                --ip ${LOCAL_IP}
            ;;
        c) echo " C node, do nothing." ;;
    esac
}

build_global_proxy() {
	(pkill -9 -f "nginx" &)
    sleep 2
    cd ${CODE_PATH}/omni/accelerators/sched/global_proxy/
    bash build.sh
}

install_vllm_and_omniinfer() {
    cd ${CODE_PATH}/infer_engines
	bash bash_install_code.sh
	pip uninstall vllm -y
	pip uninstall vllm_ascend -y
	pip uninstall omni_infer -y
	cd vllm
	SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . 
	pip install numpy==1.26.4
	cd ../../ 
	pip install -e . > ${LOG_PATH}/pip.log

    pip install pybind11
    pip uninstall omni_placement -y 
    cd ${CODE_PATH}/omni/accelerators/placement && python3 setup.py bdist_wheel >> ${LOG_PATH}/pip.log
    pip install ${CODE_PATH}/omni/accelerators/placement/dist/omni_*.whl >> ${LOG_PATH}/pip.log
}

get_node_info() {
    # 1. IP
	IP_LIST=${ENV_IP_LIST:-"[10.10.10.1,10.10.10.2,10.10.10.3,10.10.10.4,10.10.10.5,10.10.10.6,10.10.10.7,10.10.10.8]"}
	LOCAL_IP=${ENV_LOCAL_IP:-"10.10.10.1"}
	if [ -z "$LOCAL_IP" ]; then
		echo "Error: LOCAL_IP environment variable not set"
		exit 1
	fi
	echo "LOCAL_IP: $LOCAL_IP"
	echo "IP_LIST: $IP_LIST"

    cleaned_ip_list="${IP_LIST//[\[\] ]/}"
    IFS=',' read -ra ip_list <<< "$cleaned_ip_list"
    count=${#ip_list[@]}
    half_point=$(( count / 2 ))
	index=0
	
	echo "ip_list: ${ip_list[@]}"
    echo "half_point: $half_point"

    for ip in "${ip_list[@]}"; do
        if [ $index -lt $half_point ]; then
			PREFILL_SERVER_LIST+=("$ip")
		else
			DECODE_SERVER_LIST+=("$ip")
		fi
		index=$((index+1));
    done
	
	echo "PREFILL_SERVER_LIST: ${PREFILL_SERVER_LIST[@]}"
	echo "DECODE_SERVER_LIST: ${DECODE_SERVER_LIST[@]}"

	if [[ " ${PREFILL_SERVER_LIST[@]} " =~ " $LOCAL_IP " ]]; then
		NODE_INDEX=$(printf '%s\n' "${PREFILL_SERVER_LIST[@]}" | grep -n "^${LOCAL_IP}$" | cut -d: -f1)
		NODE_INDEX=$((NODE_INDEX-1))
		NODE_NAME="prefill_${NODE_INDEX}"
		NODE_ROLE="prefill"
	else
		NODE_INDEX=$(printf '%s\n' "${DECODE_SERVER_LIST[@]}" | grep -n "^${LOCAL_IP}$" | cut -d: -f1)
		NODE_INDEX=$((NODE_INDEX-1))
		NODE_NAME="decode_${NODE_INDEX}"
		NODE_ROLE="decode"
	fi

	echo "PREFILL_SERVER_LIST: ${PREFILL_SERVER_LIST[@]}"
	echo "DECODE_SERVER_LIST: ${DECODE_SERVER_LIST[@]}"
	echo "Current node IP: $LOCAL_IP, Role: $NODE_ROLE, NAME: $NODE_NAME, INDEX:$NODE_INDEX"
}

get_rank_dir() {
    NFS_BASE="${ENV_NFS_PATH:-/dev/nfs/}"
    NFS_PATH="${NFS_BASE%/}"
    crc_result=$(echo -n "$IP_LIST" | cksum | awk '{print $1}')
    MIDDLE_DIR="ranktable_dir"
    RANKTABLE_SAVE_PATH="${NFS_PATH}/${MIDDLE_DIR}/${crc_result}"
}

wait_rank_dir_ready() {
    echo "NODE_NAME: ${NODE_NAME}"
    echo "RANKTABLE_SAVE_PATH: ${RANKTABLE_SAVE_PATH}"
    if [ "$NODE_NAME" = "prefill_0" ]; then
        echo "p0 start create rank dir"
        rm -rf "${RANKTABLE_SAVE_PATH}"
        mkdir -p "${RANKTABLE_SAVE_PATH}"
        touch ${RANKTABLE_SAVE_PATH}/rank_dir_ready
    else
        echo "check rank dir"
        while [ ! -f ${RANKTABLE_SAVE_PATH}/rank_dir_ready ]; do
            echo "Waiting for rank dir ready..."
            sleep 1
        done
    fi
}

wait_global_ranktable_ready() {
    echo "NODE_NAME: $NODE_NAME"
    while [ ! -f "$RANKTABLE_SAVE_PATH/global/global_ranktable_merge.json" ]; do
        echo "Waiting for global_ranktable_merge.json to be created..."
        sleep 1
    done
}

wait_app_ok() {
    log_file=$LOG_PATH/server_0.log
    echo "log_file: $log_file"
    while ! grep -q "Application startup complete" "$log_file"; do
        sleep 5
    done
}

check_env_value() {
    if [ -z "${ENV_MODEL_PATH+x}" ]; then
        echo "error: env mode path is not existed"
        exit 1
    fi

    if [ -z "$ENV_MODEL_PATH" ]; then
        echo "error: mode path is null"
        exit 1
    fi

    if [ ! -e "$ENV_MODEL_PATH" ]; then
        echo "error: mode path is not existed"
        exit 1
    fi

    if [ -z "${ENV_SOCKET_NAME+x}" ]; then
        echo "error: env socket name is not existed"
        exit 1
    fi

    if [ -z "$ENV_SOCKET_NAME" ]; then
        echo "error: socket name is null"
        exit 1
    fi

    if [ -z "${ENV_IP_LIST+x}" ]; then
        echo "error: env ip list is not existed"
        exit 1
    fi

    if [ -z "$ENV_IP_LIST" ]; then
        echo "error: ip list is null"
        exit 1
    fi

    if [ -z "${ENV_LOCAL_IP+x}" ]; then
        echo "error: env local ip is not existed"
        exit 1
    fi

    if [ -z "$ENV_LOCAL_IP" ]; then
        echo "error: local ip is null"
        exit 1
    fi

    if [ -z "${ENV_NFS_PATH+x}" ]; then
        echo "error: env nfs path is not existed"
        exit 1
    fi

    if [ -z "$ENV_NFS_PATH" ]; then
        echo "error: nfs path is null"
        exit 1
    fi
}

get_tmp_path() {
    MODEL_PATH="${ENV_MODEL_PATH:-/data/model/DeepSeek-R1-for-v1}"
    LOG_PATH="${ENV_LOG_PATH:-/data/tmp}"
    SOCKET_NAME="${ENV_SOCKET_NAME:-"error"}"
}

print_all_variables() {
    echo "===== global var ====="
    echo "CODE_PATH: $CODE_PATH"
    echo "RANKTABLE_SAVE_PATH: $RANKTABLE_SAVE_PATH"
    echo "LOG_PATH: $LOG_PATH"
    echo "ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
    echo "MAX_MODEL_LEN_P: $MAX_MODEL_LEN_P"
    echo "MAX_MODEL_LEN_D: $MAX_MODEL_LEN_D"
    echo "NODE_PORT_BASE: $NODE_PORT_BASE"
    echo "API_PORT_BASE: $API_PORT_BASE"
    echo "PROXY_PORT: $PROXY_PORT"
    echo "GPU_UTIL_P: $GPU_UTIL_P"
    echo "GPU_UTIL_D: $GPU_UTIL_D"
    echo "ADDITIONAL_CONFIG_P: $ADDITIONAL_CONFIG_P"
    echo "ADDITIONAL_CONFIG_D: $ADDITIONAL_CONFIG_D"
    echo "EXTRA_ARGS_P: $EXTRA_ARGS_P"
    echo "EXTRA_ARGS_D: $EXTRA_ARGS_D"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "ENV_IP_LIST: $ENV_IP_LIST"
    echo "ENV_LOCAL_IP: $ENV_LOCAL_IP"
    
    echo -e "\n===== local var ====="
    echo "IP_LIST: $IP_LIST"
    echo "LOCAL_IP: $LOCAL_IP"
    echo "NODE_INDEX: $NODE_INDEX"
    echo "NODE_NAME: $NODE_NAME"
    echo "NODE_ROLE: $NODE_ROLE"
    echo "SOCKET_NAME: $SOCKET_NAME"
    
    echo -e "\n===== arr ====="
    echo "PREFILL_SERVER_LIST: ${PREFILL_SERVER_LIST[@]}"
    echo "DECODE_SERVER_LIST: ${DECODE_SERVER_LIST[@]}"
}

check_env_value
echo "===== before get var info ====="
get_node_info
get_rank_dir
get_tmp_path
print_all_variables
mkdir -p ${LOG_PATH}
echo "===== after get var info ====="

echo "===== before install_vllm_and_omniinfer ====="
install_vllm_and_omniinfer
echo "===== after install_vllm_and_omniinfer ====="

echo "===== before build_global_proxy ====="
if [ "$NODE_NAME" = "prefill_0" ]; then
    echo "p0 start build global proxy"
    build_global_proxy
fi
echo "===== after build_global_proxy ====="


export NODE_NAME=${NODE_NAME}
export RANKTABLE_SAVE_PATH=${RANKTABLE_SAVE_PATH}
if timeout ${TIME_OUT} bash -c "$(declare -f wait_rank_dir_ready); wait_rank_dir_ready"; then
    echo "rank dir ready."
else
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "error: wait_rank_dir_ready time out"
    fi
    exit 1
fi
mkdir -p ${RANKTABLE_SAVE_PATH}/prefill ${RANKTABLE_SAVE_PATH}/decode


echo "===== before local_ranktable_generate ====="
local_ranktable_generate
echo "===== after local_ranktable_generate ====="

echo "===== before wait global ranktable ready ====="
if [ "$NODE_NAME" = "prefill_0" ]; then
    echo "p0 start generate global ranktable"
    global_ranktable_generate
fi

if timeout ${TIME_OUT} bash -c "$(declare -f wait_global_ranktable_ready); wait_global_ranktable_ready"; then
    echo "global ranktable ready."
else
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "error: wait_global_ranktable_ready time out"
    fi
    exit 1
fi
echo "===== after wait global ranktable ready ====="

echo "===== before run_server ====="
run_server
echo "===== after run_server ====="

echo "===== before run_proxy ====="
if [ "$NODE_NAME" = "prefill_0" ]; then
    run_proxy
fi
echo "===== after run_proxy ====="


export LOG_PATH=${LOG_PATH}
if timeout ${APP_TIME_OUT} bash -c "$(declare -f wait_app_ok); wait_app_ok"; then
    echo "app is ready."
else
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "error: wait app ready time out"
    fi
    exit 1
fi