#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips)
            shift
            # 收集所有后续参数直到遇到下一个选项或结束
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                IPs+=("$1")
                shift
            done
            ;;
        --network-card-name)
            shift
            NETWORK_CARD_NAME="$1"
            shift
            ;;
        --prefill-device-cnt)
            shift
            PREFILL_DEVICE_CNT="$1"
            shift
            ;;
        --decode-device-cnt)
            shift
            DECODE_DEVICE_CNT="$1"
            shift
            ;;
    esac
done
LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
GPUS_PER_NODE=8
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6657
NNODES=${#IPs[@]}
NODE_RANK="2"
for i in "${!IPs[@]}";
do
    echo "${IPs[$i]}"
    if [ "$LOCAL_HOST" == "${IPs[$i]}" ];
    then
        NODE_RANK=$i
        break
    fi
done
if [[ $NODE_RANK == "" ]];then
    echo "[Error] para \"NODE_RANK\" must be confing"
    exit 1
fi

WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
RANKSTART=`expr $GPUS_PER_NODE \* $NODE_RANK`

echo "========>param:"
echo "WORLD_SIZE: " $WORLD_SIZE
echo "RANKSTART": $RANKSTART
echo "NNODES": $NNODES
echo "NODE_RANK": $NODE_RANK
echo "==============="

if [[ -n "${GEN_RANKTABLE}" || ! -e ${PWD}/ranktable.json ]]; then
    GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME torchrun \
        --nproc_per_node 1 \
        --nnodes ${NNODES} \
        --node_rank ${NODE_RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        gen_ranktable.py --prefill-device-cnt $PREFILL_DEVICE_CNT --decode-device-cnt $DECODE_DEVICE_CNT
fi