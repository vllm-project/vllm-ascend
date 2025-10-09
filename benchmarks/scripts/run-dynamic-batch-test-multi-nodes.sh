#!/bin/bash

# This is an example bench test for dynamic batch on two nodes (each with 8 NPUs) for quick start;
# The parallel includes DP size of 4 (2 on each node), TP size of 4 on each node, and EP.


model_path=model/DeepSeek-V3.1-W8A8
dataset_path=azure_transformed.json


# 1:run the following scripts on the two nodes, respetively
# on Node 0:
nic_name_0=""
local_ip_0=""
slo=18
bash benchmarks/tests/serving-test-dynamic-batch-multi-node-0.sh $model_path $dataset_path  $slo $nic_name_0 $local_ip_0

# on Node 1:
nic_name_1=""
local_ip_1=""
bash benchmarks/tests/serving-test-dynamic-batch-multi-node-1.sh $model_path $dataset_path $slo $nic_name_1 $local_ip_1 $local_ip_0

# 2:When the service is ready, run the following scripts on Node0:
bash benchmarks/tests/serving-test-dynamic-batch-request.sh $model_path $dataset_path $slo $nic_name_0 $local_ip_0


# 3:run metrci calculation to extact quantitative results:
python benchmarks/scripts/metric_cal.py

# 3:exit
VLLM_PID=$(pgrep -f "vllm")
kill -2 "$VLLM_PID"
pkill -f timeout
pkill -f python
