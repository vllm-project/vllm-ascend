export HCCL_IF_IP=141.61.29.146
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export TP_SOCKET_IFNAME=enp48s3u1u1
export HCCL_SOCKET_IFNAME=enp48s3u1u1

export VLLM_USE_V1=1

# dp_size = node_size * dp_per_node
node_size=2
node_rank=0
dp_per_node=1
master_addr=141.61.29.146
master_port=12345

rm -rf ./.torchair_cache/
rm -rf ./dynamo_*
rm -rf /root/ascend/log/debug/plog/*

torchrun --nproc_per_node ${dp_per_node} --nnodes ${node_size} \
    --node_rank ${node_rank} --master_addr ${master_addr} --master_port ${master_port} \
    data_parallel.py
