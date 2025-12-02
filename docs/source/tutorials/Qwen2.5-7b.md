# Qwen2.5-7B-Instruct Deployment and Verification Guide

## Introduction

Qwen2.5-7B-Instruct is a 7-billion-parameter large language model pre-trained on 18 trillion tokens. It supports a maximum context window of 128K, enables generation of up to 8K tokens, and delivers enhanced capabilities in multilingual processing, instruction following, programming, mathematical computation, and structured data handling.

This document details the complete deployment and verification workflow for the model, including supported features, environment preparation, single-node deployment, functional verification, accuracy and performance evaluation, and troubleshooting of common issues. It is designed to help users quickly complete model deployment and validation.

## Supported Features

Qwen2.5-7B-Instruct offers the following core capabilities:
- **Multilingual Support**: Compatible with over 29 languages (Chinese, English, French, Spanish, Russian, Japanese, etc.).
- **Instruction Following**: Optimized through instruction tuning to accurately understand and execute user commands.
- **Programming & Mathematical Proficiency**: Delivers excellent performance on benchmarks such as HumanEval (programming) and MATH (mathematics).
- **Structured Data Handling**: Enhanced ability to process and generate structured data (e.g., tables, JSON formats).
- **Long Context Processing**: Supports a maximum context length of 128K for efficient handling of ultra-long text sequences.

## Environment Preparation

### Model Weight

Qwen2.5-7B-Instruct model weights can be downloaded from the official ModelScope repository (Note: Corrected from VL version to the correct language model link):
- [Qwen2.5-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)

It is recommended to download the model weights to a local directory (e.g., `./Qwen2.5-7B-Instruct/`) for quick access during deployment.

### Hardware and System Requirements

| Component | Specification |
|-----------|---------------|
| Hardware Platform | 910B4 (8 cards × 32GB) |
| Operating System | Ubuntu 22.04 (Corrected from non-official 22.03 version) |
| Driver Version | 25.0.rc1.1 |
| Python Version | 3.12 |

### Software Dependencies

| Component | Version Requirement | Notes |
|-----------|---------------------|-------|
| CANN      | 8.2.RC1             | Ascend Computing Architecture Dependency |
| PyTorch   | 2.5.1.post0         | Base Deep Learning Framework |
| torch-npu | 2.7.1rc1            | Ascend-adapted version |
| vLLM      | 0.9.1               | Must match vLLM-Ascend version |
| vLLM-Ascend | 0.9.1-dev        | Ascend-optimized version |

### Environment Check and Verification

Verify hardware status and network connectivity before installation:
```bash
# Check NPU device status
npu-smi info

# Verify network interface and connectivity
for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done
for i in {0..15}; do hccn_tool -i $i -link -g; done
for i in {0..15}; do hccn_tool -i $i -net_health -g; done

# Check IP configuration
for i in {0..15}; do hccn_tool -i $i -ip -g; done
```

### Container Environment Setup

Create a privileged container to isolate the deployment environment:
```bash
docker run -it --privileged --name=test_vllm_Qwen_2.5_7B --net=host --shm-size=500g \
--device=/dev/davinci{0..15} \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /home/:/home \
-w /home/<name> \
mindie:dev-2.2.RC1.B070-800I-A2-py312-ubuntu22.03-x86_64 \
/bin/bash
```
Replace `<name>` with your actual username.

### Installation

Install the required software dependencies in the container following these steps:

#### Step 1: Install CANN Toolkit
```bash
# Execute the CANN installation package (adjust path to match local file)
./Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run --install --install-path=/home/<name>/cmc/cann_8.2.rc1

# Configure CANN environment variables (New: Ensure dependencies take effect)
echo "source /home/<name>/cmc/cann_8.2.rc1/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

#### Step 2: Configure PyTorch Environment
```bash
# Set up pip mirror sources for faster installation
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"

# Install PyTorch and torch-npu (Fixed version compatibility)
pip install torch==2.5.1.post0 torchvision==0.18.0 torch-npu==2.7.1rc1
```

#### Step 3: Install vLLM and vLLM-Ascend
```bash
# Install dependency packages (New: Avoid compilation failures)
pip install cmake ninja sentencepiece transformers

# Install vLLM (v0.9.1)
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout releases/v0.9.1
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vLLM-Ascend (v0.9.1-dev, Ascend-optimized version)
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && git checkout v0.9.1-dev
pip install -v -e .
cd ..
```

#### Step 4: Install Accuracy Evaluation Tool (AISBench)
The AISBench tool is used for model accuracy and performance evaluation. Follow these installation steps:

:::{note}
The server may be in a restricted network zone (Yellow Zone) and require a Green Zone proxy tool for internet access. Download the proxy tool from the internal repository, run `PortMapping.exe` to obtain the proxy IP, and update `ip_addr` in `portproxy_remote.sh` before executing the script.
:::

```bash
# Clone the AISBench repository
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/

# Install core dependencies
pip3 install -e ./ --use-pep517

# Install dependencies for service-oriented model evaluation (vLLM/Triton)
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt

# Install BFCL evaluation dependencies
pip3 install -r requirements/bfcl_dependencies.txt --no-deps

# Disable proxy after installation
unset https_proxy
unset http_proxy
```

For detailed installation instructions, refer to the [AISBench Official Documentation](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/install.html).

## Deployment

### Single-node Deployment

Qwen2.5-7B-Instruct supports single-node single-card deployment on the 910B4 platform. Follow these steps to start the inference service:

1. Prepare model weights: Ensure the downloaded model weights are stored in the `./Qwen2.5-7B-Instruct/` directory.
2. Download the gsm8k dataset (for evaluation): [gsm8k.zip](https://vision-file-storage/api/file/download/attachment-v2/WIKI202511118986704/32978033/20251111T144846Z_9658c67a0fb349f9be081ab9ab9fd2bc.zip?attachment_id=32978033)
3. Create and execute the deployment script (save as `deploy.sh`):

```shell
#!/bin/sh
# Set environment variables for Ascend optimization
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export PAGED_ATTENTION_MASK_LEN=max_seq_len
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1

# Start vLLM inference service
vllm serve ./Qwen2.5-7B-Instruct/ \
          --host <IP> \          # Replace with server IP (e.g., 0.0.0.0 for all interfaces)
          --port <Port> \        # Replace with available port (e.g., 8080)
          --served-model-name qwen-2.5-7b-instruct \  # Standardized model name for consistency
          --trust-remote-code \
          --dtype bfloat16 \
          --max-model-len 32768 \ # Maximum context length (adjust based on requirements)
          --tensor-parallel-size 1 \ # Single-card deployment
          --disable-log-requests \
          --enforce-eager

# Execution command: chmod +x deploy.sh && ./deploy.sh
```

### Multi-node Deployment

This document currently focuses on single-node deployment. For multi-node deployment, refer to the [vLLM-Ascend Multi-node Guide](https://github.com/vllm-project/vllm-ascend) and ensure consistent environment configuration across all nodes.

### Prefill-Decode Disaggregation

This feature is not supported at this time.

## Functional Verification

After starting the service, verify functionality using a `curl` request:

```bash
curl http://<IP>:<Port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen-2.5-7b-instruct",  # Must match --served-model-name from deployment
        "prompt": "Beijing is a",
        "max_tokens": 5,
        "temperature": 0
    }'
```

A valid response (e.g., `"Beijing is a vibrant and historic capital city"`) indicates successful deployment.

### Supplementary Verification Method (New)
If `curl` verification fails, use this Python script:
```python
import requests

url = "http://<IP>:<Port>/v1/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "qwen-2.5-7b-instruct",
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 100,
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

## Accuracy Evaluation

Two accuracy evaluation methods are provided: AISBench (recommended) and manual testing with standard datasets.

### Using AISBench

#### Prerequisites
1. Extract the gsm8k dataset to `benchmark/datasets/gsm8k/` (download from the link above).
2. Configure model evaluation parameters.

#### Configuration Steps
1. Locate the AISBench configuration file:
```bash
cd benchmark/
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```
2. Modify the configuration file (e.g., `vllm_api_general_chat.py`) to match the deployed service:
```Python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",
        model="qwen-2.5-7b-instruct",  # Must match --served-model-name from deployment
        request_rate=0,
        retry=2,
        host_ip="<IP>",  # Deployment server IP
        host_port=<Port>,  # Deployment server port
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            seed=None,
            repetition_penalty=1.03,
        )
    )
]
```

#### Execution Command
```bash
# Specify visible NPU cards (adjust based on available hardware)
export ASCEND_RT_VISIBLE_DEVICES=0

# Run evaluation (debug logs recommended for first execution)
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug

# Generate summary report
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```

#### Evaluation Results
Results and logs are saved to `benchmark/outputs/default/`. A sample accuracy report is shown below:
![Accuracy Evaluation Result](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202511118986704/32976454/30bf146f86ab472697430f8efae66c1a.png)

### Pure Model Accuracy Evaluation
For local model evaluation (without service deployment), modify `attr="local"` in the AISBench configuration file:
```Python
dict(
    attr="local",  # Change from "service" to "local"
    type=VLLMCustomAPIChat,
    abbr='vllm-api-general-chat',
    path="./Qwen2.5-7B-Instruct/",  # Path to local model weights
    model="qwen-2.5-7b-instruct",
    # ... (other parameters remain unchanged)
)
```

## Performance Evaluation

### Using AISBench
Add `--mode perf` to the accuracy evaluation command to run performance testing:
```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example --mode perf
```

#### Performance Metrics
Key metrics include throughput (tokens/sec), latency (ms), and NPU utilization. A sample result is shown below:
![Performance Evaluation Result](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202511118986704/32976455/2b68624624a2436db5959e51aebaa106.png)

For detailed metric explanations, refer to the [AISBench Performance Documentation](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/results_intro/performance_metric.html#).

### Using vLLM Benchmark
vLLM includes a built-in benchmark tool for evaluating throughput and latency. Example command for online serving performance testing:
```bash
export VLLM_USE_MODELSCOPE=true
vllm bench serve \
  --model ./Qwen2.5-7B-Instruct/ \
  --dataset-name random \
  --random-input 200 \
  --num-prompt 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./perf_results/
```

For more details, refer to the [vLLM Benchmark Documentation](https://docs.vllm.ai/en/latest/contributing/benchmarks.html).

## Common Issues and Solutions

### How to Check Service Status and Metrics?
- **Enable Monitoring**: Add the `--metrics` parameter to the deployment command. Access metrics via `http://<IP>:<Port>/metrics` (Prometheus-compatible) to view NPU utilization, queue length, and inference latency.
- **Debug Logs**: Add the `--log-level debug` parameter to the deployment command to output detailed logs for troubleshooting.

### Deployment Verification Failure
- **Issue**: `curl` request returns an error or no response.
- **Solutions**:
  1. Verify the server IP and port are correct (use `netstat -tuln | grep <Port>` to check port occupancy).
  2. Ensure the model weight path is correct and the model is fully loaded (look for "Model loaded successfully" in logs).
  3. Confirm firewall rules allow traffic on the deployment port (use `ufw status` to check firewall status).
  4. Verify dependency version compatibility (especially vLLM and vLLM-Ascend must match).

### Multi-Card Load Imbalance
- **Symptom**: Uneven memory usage across NPU cards.
- **Solutions**:
  1. Ensure `--tensor-parallel-size` matches the number of cards (e.g., set to 8 for 8-card deployment).
  2. For large models, adjust the `--gpu-memory-utilization` parameter (e.g., 0.9) to optimize memory allocation.
  3. Enable Ascend-specific optimizations (e.g., `VLLM_ASCEND_ENABLE_FLASHCOMM=1`).

### Network Restriction Issues
- **Issue**: Failed dependency downloads (restricted network environment).
- **Solution**: Configure the proxy using the Green Zone proxy tool as described in the [Installation](#step-4-install-accuracy-evaluation-tool-aisbench) section, then disable the proxy after installation.

### Compilation Failure When Installing vLLM
- **Issue**: Compilation errors occur when executing `pip install -v -e .`.
- **Solutions**:
  1. Ensure dependency packages are installed: `pip install cmake ninja sentencepiece`.
  2. Verify Python version is 3.12 (lower versions are not supported).
  3. Clean cache and reinstall: `rm -rf build/ dist/ *.egg-info && pip install -v -e .`.

