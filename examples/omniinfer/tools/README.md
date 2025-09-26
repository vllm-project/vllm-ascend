# pd run documents

## pd disaggregation example

- 4 node A3 2p1d
- p0_node0 10.10.10.2
- p1_node1 10.10.10.3
- d0_node2 10.10.10.4
- d0_node3 10.10.10.5

p0_node0

```bash
./pd_run.sh \
    --global-rank-table-path global_path/global_ranktable_merge.json \
    --rank-table-path prefill-ranktable-p0-13/local_ranktable_10.10.10.2_0123456789101112131415.json \
    --local-decode-server-ip-list 10.10.10.4,10.10.10.5 \
    --global-decode-server-ip-list 10.10.10.4,10.10.10.5 \
    --role prefill \
    --prefill-pod-num 2 \
    --decode-pod-num 1 \
    --gloo-socket-ifname enp23s0f3 \
    --tp-socket-ifname enp23s0f3 \
    --num-servers 1 \
    --model-path /data/models/DeepSeek-V3-w8a8-0423 \
    --master-ip 10.10.10.2 \
    --max-model-len 4096 \
    --master-port 8503 \
    --base-api-port 9001 \
    --tp 16 \
    --kv-role kv_producer \
    --kv-rank 0 \
    --kv-engine-id 0 \
    --kv-parallel-size 3 \
    --extra-args "--enable-expert-parallel"
```

p1_node1

```bash
./pd_run.sh \
    --global-rank-table-path global_path/global_ranktable_merge.json \
    --rank-table-path prefill-ranktable-p1-13/local_ranktable_10.10.10.3_0123456789101112131415.json \
    --local-decode-server-ip-list 10.10.10.4,10.10.10.5 \
    --global-decode-server-ip-list 10.10.10.4,10.10.10.5 \
    --role prefill \
    --prefill-pod-num 2 \
    --decode-pod-num 1 \
    --gloo-socket-ifname enp23s0f3 \
    --tp-socket-ifname enp23s0f3 \
    --num-servers 1 \
    --model-path /data/models/DeepSeek-V3-w8a8-0423 \
    --master-ip 10.10.10.3 \
    --max-model-len 4096 \
    --master-port 8503 \
    --base-api-port 9001 \
    --tp 16 \
    --kv-role kv_producer \
    --kv-rank 1 \
    --kv-engine-id 1 \
    --kv-parallel-size 3 \
    --extra-args "--enable-expert-parallel"
```

d0_node2

```bash
./pd_run.sh \
    --global-rank-table-path global_path/global_ranktable_merge.json \
    --rank-table-path merge-decode-ranktable/local_ranktable_merge_10.10.10.4_10.10.10.5.json \
    --local-decode-server-ip-list 10.10.10.4,10.10.10.5 \
    --global-decode-server-ip-list 10.10.10.4,10.10.10.5 \
    --role decode \
    --prefill-pod-num 2 \
    --decode-pod-num 1 \
    --gloo-socket-ifname enp23s0f3 \
    --tp-socket-ifname enp23s0f3 \
    --num-servers 16 \
    --num-dp 32 \
    --model-path /data/models/DeepSeek-V3-w8a8-0423 \
    --master-ip 10.10.10.4 \
    --max-model-len 4096 \
    --master-port 8503 \
    --base-api-port 9001 \
    --tp 1 \
    --kv-role kv_consumer \
    --kv-rank 2 \
    --kv-engine-id 2 \
    --kv-parallel-size 3 \
    --extra-args "--enable-expert-parallel"
```

d0_node3

```bash
./pd_run.sh \
    --global-rank-table-path global_path/global_ranktable_merge.json \
    --rank-table-path merge-decode-ranktable/local_ranktable_merge_10.10.10.4_10.10.10.5.json \
    --local-decode-server-ip-list 10.10.10.4,10.10.10.5 \
    --global-decode-server-ip-list 10.10.10.4,10.10.10.5 \
    --role decode \
    --prefill-pod-num 2 \
    --decode-pod-num 1 \
    --gloo-socket-ifname enp23s0f3 \
    --tp-socket-ifname enp23s0f3 \
    --num-servers 16 \
    --num-dp 32 \
    --server-offset 16 \
    --model-path /data/models/DeepSeek-V3-w8a8-0423 \
    --master-ip 10.10.10.4 \
    --max-model-len 4096 \
    --master-port 8503 \
    --base-api-port 9001 \
    --tp 1 \
    --kv-role kv_consumer \
    --kv-rank 2 \
    --kv-engine-id 2 \
    --kv-parallel-size 3 \
    --extra-args "--enable-expert-parallel"
```

- 1 node A3 2p1d
- p0_node0 10.10.10.2
- p1_node0 10.10.10.2
- d0_node0 10.10.10.2

p0_node0

```bash
./pd_run.sh \
    --global-rank-table-path global_path/global_ranktable_merge.json \
    --rank-table-path prefill-ranktable-p0-13/local_ranktable_10.10.10.2_0123.json \
    --local-decode-server-ip-list 10.10.10.2 \
    --global-decode-server-ip-list 10.10.10.2 \
    --role prefill \
    --prefill-pod-num 2 \
    --decode-pod-num 1 \
    --gloo-socket-ifname enp23s0f3 \
    --tp-socket-ifname enp23s0f3 \
    --num-servers 1 \
    --ascend-rt-visible-devices "0,1,2,3" \
    --model-path /data/models/DeepSeek-V2-Lite \
    --master-ip 10.10.10.2 \
    --max-model-len 4096 \
    --master-port 8503 \
    --base-api-port 9001 \
    --tp 4 \
    --kv-role kv_producer \
    --kv-rank 0 \
    --kv-engine-id 0 \
    --kv-parallel-size 3 \
    --extra-args "--enable-expert-parallel"
```

p1_node0

```bash
./pd_run.sh \
    --global-rank-table-path global_path/global_ranktable_merge.json \
    --rank-table-path prefill-ranktable-p1-13/local_ranktable_10.10.10.2_4567.json \
    --local-decode-server-ip-list 10.10.10.2 \
    --global-decode-server-ip-list 10.10.10.2 \
    --role prefill \
    --prefill-pod-num 2 \
    --decode-pod-num 1 \
    --gloo-socket-ifname enp23s0f3 \
    --tp-socket-ifname enp23s0f3 \
    --num-servers 1 \
    --ascend-rt-visible-devices "4,5,6,7" \
    --model-path /data/models/DeepSeek-V2-Lite \
    --master-ip 10.10.10.2 \
    --max-model-len 4096 \
    --master-port 8503 \
    --base-api-port 9001 \
    --tp 4 \
    --kv-role kv_producer \
    --kv-rank 1 \
    --kv-engine-id 1 \
    --kv-parallel-size 3 \
    --extra-args "--enable-expert-parallel"
```

d0_node0

```bash
./pd_run.sh \
    --global-rank-table-path global_path/global_ranktable_merge.json \
    --rank-table-path decode-ranktable-d0-13/local_ranktable_10.10.10.2_56789101112131415.json \
    --local-decode-server-ip-list 10.10.10.2 \
    --global-decode-server-ip-list 10.10.10.2 \
    --role decode \
    --prefill-pod-num 2 \
    --decode-pod-num 1 \
    --gloo-socket-ifname enp23s0f3 \
    --tp-socket-ifname enp23s0f3 \
    --num-servers 8 \
    --num-dp 8 \
    --ascend-rt-visible-devices "8,9,10,11,12,13,14,15" \
    --model-path /data/models/DeepSeek-V2-Lite \
    --master-ip 10.10.10.2 \
    --max-model-len 4096 \
    --master-port 8503 \
    --base-api-port 9001 \
    --tp 1 \
    --kv-role kv_consumer \
    --kv-rank 2 \
    --kv-engine-id 2 \
    --kv-parallel-size 3 \
    --extra-args "--enable-expert-parallel"
```