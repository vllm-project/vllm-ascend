#! /bin/bash
omni=$1
data=$2

batchsize=512
inputsize=256
outputsize=16

if [ $# -ne 2 ]; then
    echo "missing parameters"
    echo "usage: start_benchmark_tools.sh <omni|base> <dataset(human_eval|random)>"
fi

timestamp=$(date +"%Y%m%d%H%M%S")
output_file="${timestamp}_${omni}_${data}.csv"

echo ${output_file}

if [ "${data}" = "human_eval" ]; then
    python benchmark_parallel.py --backend openai --host 127.0.0.1 --port 8999 --dataset-type human-eval --dataset-path /home/yjf/test-00000-of-00001.json --tokenizer /opt/models/models/dsv3/DeepSeek-V3-w8a8-0208-50/  --epochs 3 --parallel-num ${batchsize} --prompt-tokens ${inputsize} --output-tokens ${outputsize} --benchmark-csv "${output_file}" --served-model-name deepseek
elif [ "${data}" = "random" ]; then
    python benchmark_parallel.py --backend openai --host 127.0.0.1 --port 8999 --tokenizer /opt/models/models/dsv3/DeepSeek-V3-w8a8-0208-50/ --epochs 3 --parallel-num ${batchsize} --prompt-tokens ${inputsize} --output-tokens ${outputsize} --benchmark-csv "${output_file}" --served-model-name deepseek
fi