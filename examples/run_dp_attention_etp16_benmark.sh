#!/bin/bash
# Concurrency array
concurrency_array=(48)
#best rate
rate_array=(0.7)

# Result file
result_file="benchmark_results.txt"
echo "Benchmark Results" > $result_file
echo "===================" >> $result_file

# Loop through all combinations
for concurrency in "${concurrency_array[@]}"; do
    for rate in "${rate_array[@]}"; do
        echo "Testing with concurrency=$concurrency, rate=$rate"
        echo "" >> $result_file
        echo "Concurrency: $concurrency, Request Rate: $rate" >> $result_file
        echo "-------------------" >> $result_file

        # Run benchmark test
        python /mnt/deepseek/vllm/benchmarks/benchmark_serving.py \
            --backend vllm \
            --trust-remote-code \
            --model /mnt/deepseek/DeepSeek-R1-W8A8-VLLM \
            --dataset-name random \
            --random-input-len 4096 \
            --random-output-len 1536 \
            --ignore-eos \
            --num-prompts 400 \
            --max-concurrency $concurrency \
            --request-rate $rate \
            --metric-percentiles 90 \
            --base-url http://localhost:8006 2>&1 | tee -a $result_file

        # Wait for system cool down
        sleep 30
    done
done

# Analyze results
echo "Analysis Results" > analysis_results.txt
echo "=================" >> analysis_results.txt

# Extract and analyze TPOT data
echo "TPOT Analysis:" >> analysis_results.txt
grep "Mean TPOT" $result_file | awk -F':' '{
    printf "Concurrency %s, Rate %s: %s ms\n", $1, $2, $3
}' >> analysis_results.txt

# Extract and analyze throughput data
echo -e "\nThroughput Analysis:" >> analysis_results.txt
grep "Output token throughput" $result_file | awk -F':' '{
    printf "Concurrency %s, Rate %s: %s tokens/s\n", $1, $2, $3
}' >> analysis_results.txt

echo "Testing completed. Results saved in $result_file and analysis in analysis_results.txt"
