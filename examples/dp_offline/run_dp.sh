export HCCL_IF_IP=${local_ip}
export GLOO_SOCKET_IFNAME=${ifname}
export TP_SOCKET_IFNAME=${ifname}
export HCCL_SOCKET_IFNAME=${ifname}

# dp_size = node_size * dp_per_node
node_size=1
node_rank=0
dp_per_node=2
master_addr=127.0.0.1
master_port=12345

torchrun --nproc_per_node ${dp_per_node} --nnodes ${node_size} \
    --node_rank ${node_rank} --master_addr ${master_addr} --master_port ${master_port} \
    data_parallel.py
