``` yaml
env_common: &env_common
  OMP_NUM_THREADS: "100"
  OMP_PROC_BIND: "false"
  HCCL_BUFFSIZE: "200"
  VLLM_RPC_TIMEOUT: "3600000"
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: "3600000"
  DISABLE_L2_CACHE: "1"
  DYNAMIC_EPLB: "true"
  SERVER_PORT: "8080"

cmd_common: &cmd_common
  - "--port"
  - "$SERVER_PORT"
  - "--quantization"
  - "ascend"
  - "--seed"
  - "1024"
  - "--no-enable-prefix-caching"
  - "--data-parallel-size"
  - "4"
  - "--tensor-parallel-size"
  - "4"
  - "--enable-expert-parallel"
  - "--max-model-len"
  - "40000"
  - "--max-num-batched-tokens"
  - "4096"
  - "--max-num-seqs"
  - "12"
  - "--trust-remote-code"
  - "--gpu-memory-utilization"
  - "0.92"
  - "--speculative-config"
  - '{"num_speculative_tokens": 1, "method": "mtp"}'
  - "--compilation-config"
  - '{"cudagraph_capture_sizes": [24], "cudagraph_mode": "FULL_DECODE_ONLY"}'
  - "--additional-config"
  - '{"enable_shared_expert_dp": false, "multistream_overlap_shared_expert": false, "eplb_config":{"dynamic_eplb": true, "expert_heat_collection_interval": 512, "algorithm_execution_interval": 100, "num_redundant_experts": 0}}'

benchmarks_common: &benchmarks_common
  acc:
    case_type: accuracy
    dataset_path: vllm-ascend/gsm8k-lite
    request_conf: vllm_api_general_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_chat_prompt
    max_out_len: 32768
    batch_size: 32
    baseline: 95
    threshold: 5

test_cases:
  - name: "DeepSeek-R1-0528-W8A8-EPLB-acc"
    model: "vllm-ascend/DeepSeek-R1-0528-W8A8"
    prompts:
      - "San Francisco is a"
    api_keyword_args:
      max_tokens: 16
    envs:
      <<: *env_common
    cmd_base: *cmd_common
    cmd_extra:
      - "--eplb-config"
      - '{"dynamic_eplb": true, "expert_heat_collection_interval": 512, "algorithm_execution_interval": 100, "num_redundant_experts": 0}'
    benchmarks:
      <<: *benchmarks_common

  - name: "DeepSeek-R1-0528-W8A8-EPLB-latency"
    model: "vllm-ascend/DeepSeek-R1-0528-W8A8"
    envs:
      <<: *env_common
    cmd_base: *cmd_common
    cmd_extra:
      - "--eplb-config"
      - '{"dynamic_eplb": true, "expert_heat_collection_interval": 512, "algorithm_execution_interval": 100, "num_redundant_experts": 0}'
    benchmarks:
      <<: *benchmarks_common
```
