#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}


IPs=('your_ip_0' 'your_ip_1') # change this value 
LOCAL_HOST=`hostname -I|awk -F " " '{print$2}'`
GPUS_PER_NODE=16 # change this value
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6657
NNODES=${#IPs[@]}
NODE_RANK="0" # change this value 
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
    GLOO_SOCKET_IFNAME=enp48s3u1u1 torchrun \
        --nproc_per_node 1 \
        --nnodes ${NNODES} \
        --node_rank ${NODE_RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        gen_ranktable.py --prefill-device-cnt $1 --decode-device-cnt $2
fi
