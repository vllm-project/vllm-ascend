#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}

NPUS_PER_NODE=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                IPs+=("$1")
                shift
            done
            ;;
        --npus-per-node)
            shift
            NPUS_PER_NODE="$1"
            shift
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
        --local-device-ids)
            shift
            LOCAL_DEVICE_IDS="$1"
            shift
            ;;
    esac
done
LOCAL_HOSTS=($(hostname -I))
LOCAL_HOST="127.0.0.1"
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6657
NNODES=${#IPs[@]}
NODE_RANK="8"
for i in "${!IPs[@]}"; do
    ip="${IPs[$i]}"
    for local_host in "${LOCAL_HOSTS[@]}"; do
        if [[ "$local_host" == "$ip" ]]; then
            LOCAL_HOST=$local_host
            NODE_RANK=$i
            break 2
        fi
    done
done

if [[ $NODE_RANK == "" ]];then
    echo "[Error] para \"NODE_RANK\" must be defined"
    exit 1
fi

WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
RANKSTART=`expr $NPUS_PER_NODE \* $NODE_RANK`

echo "========>param:"
echo "LOCAL_HOST": $LOCAL_HOST
echo "WORLD_SIZE: " $WORLD_SIZE
echo "RANKSTART": $RANKSTART
echo "NNODES": $NNODES
echo "NODE_RANK": $NODE_RANK
echo "==============="

if [ -n "$LOCAL_DEVICE_IDS" ]; then
  OPTIONAL_SECTION=" --local-device-ids $LOCAL_DEVICE_IDS"
fi

if [[ -n "${GEN_RANKTABLE}" || ! -e ${PWD}/ranktable.json ]]; then
    GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME torchrun \
        --nproc_per_node 1 \
        --nnodes ${NNODES} \
        --node_rank ${NODE_RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        gen_ranktable.py --local-host $LOCAL_HOST --prefill-device-cnt $PREFILL_DEVICE_CNT --decode-device-cnt $DECODE_DEVICE_CNT $OPTIONAL_SECTION
fi