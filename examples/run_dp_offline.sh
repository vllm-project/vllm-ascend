rm -rf ./.torchair_cache/
rm -rf ./dynamo_*
rm -rf /root/ascend/log/debug/plog/*

ifname="ifname"
local_ip="local ip"
master_addr="master ip"
model_path="path to model ckpt"

export HCCL_IF_IP=${local_ip}
export GLOO_SOCKET_IFNAME=${ifname}
export TP_SOCKET_IFNAME=${ifname}
export HCCL_SOCKET_IFNAME=${ifname}

export VLLM_USE_V1=1
export ASCEND_LAUNCH_BLOCKING=0
# export VLLM_VERSION=0.9.0

python data_parallel.py \
    --model=${model_path} \
    --dp-size=4 \
    --tp-size=4 \
    --enforce-eager \
    --trust-remote-code \
    --node-size=1 \
    --node-rank=0 \
    --master-addr=${master_addr} \
    --master-port=13345
