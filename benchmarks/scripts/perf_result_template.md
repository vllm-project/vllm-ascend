## Online serving tests

- Input length: randomly sample 200 prompts from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json) and [lmarena-ai/vision-arena-bench-v0.1](https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1/tree/main)(multi-modal) dataset (with fixed random seed).
- Output length: the corresponding output length of these 200 prompts.
- Batch size: dynamically determined by vllm and the arrival pattern of the requests.
- **Average QPS (query per second)**: 1, 4, 16 and inf. QPS = inf means all requests come at once. For other QPS values, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
- Models: Qwen/Qwen3-8B, Qwen/Qwen2.5-VL-7B-Instruct
- Evaluation metrics: throughput, TTFT (median time to the first token ), ITL (median inter-token latency) TPOT(median time per output token).

{serving_tests_markdown_table}

## Offline tests
### Latency tests

- Input length: 32 tokens.
- Output length: 128 tokens.
- Batch size: fixed (8).
- Models: Qwen/Qwen3-8B, Qwen/Qwen2.5-VL-7B-Instruct
- Evaluation metrics: end-to-end latency.

{latency_tests_markdown_table}

### Throughput tests

- Input length: randomly sample 200 prompts from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json) and [lmarena-ai/vision-arena-bench-v0.1](https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1/tree/main)(multi-modal) dataset (with fixed random seed).
- Output length: the corresponding output length of these 200 prompts.
- Batch size: dynamically determined by vllm to achieve maximum throughput.
- Models: Qwen/Qwen3-8B, Qwen/Qwen2.5-VL-7B-Instruct
- Evaluation metrics: throughput.

{throughput_tests_markdown_table}