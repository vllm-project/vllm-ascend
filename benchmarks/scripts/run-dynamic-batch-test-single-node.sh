#!/bin/bash

# This is an example test for dynamic batch on single node with 8 NPUs;

model=model/new_model/DeepSeek-R1-w4a8-pruning   # path to model weights
dataset=benchmarks/tests/bench_dynamic_batch.sh  # path to test script
num_prompts=1000
TP=8


# 1: run benchmarks of different slo limits
for slo in 0 35 50 75 100;do
    bash bench_dynamic_batch.sh $model $slo $dataset $num_prompts $TP

# 2: run metrci calculation to extact quantitative results:
python benchmarks/scripts/metric_cal.py

# 3: exit
VLLM_PID=$(pgrep -f "vllm")
kill -2 "$VLLM_PID"
pkill -f timeout
pkill -f python


