# IQuestLab/IQuest-Coder-V1-40B-Instruct

## Overview

**IQuest-Coder-V1-40B-Instruct** is a state-of-the-art code large language model (LLM) developed by IQuestLab. With 40 billion parameters, this model represents a significant advancement in autonomous software engineering and code intelligence capabilities.

Built on an innovative code-flow multi-stage training paradigm, IQuest-Coder-V1 captures the dynamic evolution of software logic, delivering exceptional performance across multiple dimensions of code understanding, generation, and reasoning tasks.

### Key Features

| Feature | Description |
|---------|-------------|
| **Architecture** | 40B parameter decoder-only transformer |
| **Context Window** | Up to 32,768 tokens |
| **Precision** | BF16 (BFloat16) optimized for Ascend NPU |
| **Training Paradigm** | Code-flow multi-stage training for dynamic software logic capture |
| **Specialization** | Software engineering, code generation, debugging, and analysis |
| **Deployment** | Optimized for vLLM Ascend with tensor parallelism support |

### Model Capabilities

- **Code Generation**: Generate high-quality, contextually appropriate code snippets across multiple programming languages
- **Code Completion**: Intelligent autocomplete for ongoing development workflows
- **Code Explanation**: Provide detailed explanations of complex code structures and algorithms
- **Debugging Assistance**: Identify potential bugs and suggest fixes
- **Refactoring**: Transform and optimize existing code while preserving functionality
- **Documentation Generation**: Automatically generate code documentation and comments
- **Algorithm Design**: Assist in designing efficient algorithms and data structures

## Intended Use Cases

### Primary Applications

1. **Software Development**
   - Accelerating developer productivity through intelligent code suggestions
   - Rapid prototyping and boilerplate code generation
   - Legacy code modernization and migration

2. **Code Review and Quality Assurance**
   - Automated code review assistance
   - Style guide enforcement and consistency checking
   - Security vulnerability detection

3. **Technical Education**
   - Interactive coding tutorials and explanations
   - Algorithm visualization and step-by-step walkthroughs
   - Programming language learning assistance

4. **DevOps and Automation**
   - Infrastructure-as-code generation
   - CI/CD pipeline scripting
   - Configuration file management

### Target Users

- Professional software engineers and architects
- Data scientists and ML engineers
- Technical educators and students
- DevOps and SRE teams
- Open-source contributors

## Performance Benchmarks

IQuest-Coder-V1-40B-Instruct demonstrates competitive performance on industry-standard code evaluation benchmarks:

| Benchmark | Description | Performance |
|-----------|-------------|-------------|
| HumanEval | Functional code generation | Industry-leading |
| MBPP (Mostly Basic Python Problems) | Python programming problems | Excellent |
| CodeContests | Competitive programming | Strong |
| APPS | Advanced programming problems | Competitive |
| DS-1000 | Data science code generation | High-performing |

> **Note**: Exact benchmark scores may vary based on evaluation configuration and hardware setup. For the most current performance metrics, refer to the official [IQuestLab model card](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Instruct).

## Installation and Setup

### Prerequisites

- Ascend NPU hardware (A2 or A3 series recommended)
- Docker environment configured for Ascend
- Sufficient NPU memory (40B model requires multi-card setup for optimal performance)

### Model Weight Download

Download the model weights from Hugging Face:

```bash
# Model: IQuestLab/IQuest-Coder-V1-40B-Instruct (BF16 version)
# URL: https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Instruct
```

> **Important**: The public model files use custom model and tokenizer code, requiring `--trust-remote-code` during deployment.

It is recommended to download the model weights to a shared local directory such as `/root/.cache/huggingface/` before deployment.

### Docker Deployment

Choose the appropriate Docker image based on your Ascend hardware:

#### A3 Series

```bash
export IMAGE=quay.io/ascend/vllm-ascend:<vllm_ascend_version>-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

#### A2 Series

```bash
export IMAGE=quay.io/ascend/vllm-ascend:<vllm_ascend_version>
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

## Deployment Guide

### Single-Node Deployment

`IQuest-Coder-V1-40B-Instruct` is a BF16 40B text generation model with a 32K maximum context length. For initial deployment, use single-node configuration with tensor parallelism.

Create a deployment script (`deploy_IQuest_Coder_V1.sh`):

```shell
#!/bin/sh
export HF_HOME=/data/huggingface_home
export HF_ENDPOINT=https://hf-mirror.com
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MODEL_PATH="IQuestLab/IQuest-Coder-V1-40B-Instruct"

vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name IQuest-Coder-V1-40B-Instruct \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enforce-eager
```

**Configuration Notes**:
- Adjust `ASCEND_RT_VISIBLE_DEVICES` to match your available NPU devices
- Modify `--tensor-parallel-size` based on available NPU memory (recommend 2+ cards for 40B model)
- Reduce `--max-model-len` if encountering memory constraints
- Increase tensor parallel size for larger batch sizes or context lengths

### Multi-Node Deployment

Single-node deployment is recommended for initial verification. Multi-node deployment should only be attempted after confirming basic inference functionality and memory requirements on your target hardware.

### Prefill-Decode Disaggregation

Not currently verified in this release.

## Usage Examples

### Basic API Request

After starting the service, interact with the OpenAI-compatible endpoint:

```shell
curl http://<IP>:<Port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "IQuest-Coder-V1-40B-Instruct",
        "messages": [
            {"role": "system", "content": "You are an expert programming assistant."},
            {"role": "user", "content": "Write a Python function to implement quicksort with type hints and docstrings."}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }'
```

### Python Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="IQuest-Coder-V1-40B-Instruct",
    messages=[
        {"role": "system", "content": "You are an expert software engineer."},
        {"role": "user", "content": "Explain the difference between asyncio and threading in Python."}
    ],
    temperature=0.2,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Sample Use Cases

1. **Code Generation**
   ```
   "Generate a REST API endpoint in FastAPI for user authentication with JWT tokens."
   ```

2. **Code Review**
   ```
   "Review this Python code for potential memory leaks and suggest improvements."
   ```

3. **Debugging**
   ```
   "This function is throwing a KeyError. Help me identify and fix the issue."
   ```

4. **Documentation**
   ```
   "Generate comprehensive documentation for this class including usage examples."
   ```

## Functional Verification

After service startup, verify functionality with domain-specific requests:

### Software Engineering Verification

```shell
curl http://<IP>:<Port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "IQuest-Coder-V1-40B-Instruct",
        "messages": [
            {"role": "system", "content": "You are an expert software engineering assistant."},
            {"role": "user", "content": "Design a thread-safe singleton pattern in Java with lazy initialization."}
        ],
        "temperature": 0,
        "max_tokens": 512
    }'
```

A valid response should contain syntactically correct code, proper thread-safety mechanisms, and clear explanations.

## Accuracy Evaluation

### Using LM Evaluation Harness

An evaluation configuration is provided at `tests/e2e/models/configs/IQuest-Coder-V1-40B-Instruct.yaml`.

Run the accuracy evaluation:

```shell
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
    --config tests/e2e/models/configs/IQuest-Coder-V1-40B-Instruct.yaml
```

The current configuration uses standard code evaluation benchmarks for validation.

## Performance Benchmarking

### Using vLLM Benchmark Tools

After confirming service stability, run performance tests:

```shell
vllm bench serve \
  --model IQuestLab/IQuest-Coder-V1-40B-Instruct \
  --trust-remote-code \
  --dataset-name random \
  --random-input 512 \
  --num-prompts 100 \
  --request-rate 1 \
  --save-result \
  --result-dir ./perf_results/
```

**Benchmarking Recommendations**:
- Start with moderate prompt lengths and concurrency
- Gradually increase `--max-model-len` and request rate
- Monitor NPU memory utilization throughout testing
- Establish baseline metrics before production deployment

## Supported Features

For a complete feature compatibility matrix, refer to [supported features](../../user_guide/support_matrix/supported_models.md).

For feature configuration options, see the [feature guide](../../user_guide/feature_guide/index.md).

## Limitations and Safety Guidelines

### Known Limitations

1. **Hardware Requirements**: The 40B parameter model requires significant NPU memory. Multi-card tensor parallelism is recommended for optimal performance.

2. **Custom Code Dependency**: The public Hugging Face model uses custom model and tokenizer code, requiring `trust_remote_code=True` during deployment.

3. **Context Window**: While the model supports up to 32K tokens, actual usable context depends on available NPU memory and deployment configuration.

4. **Language Support**: Performance may vary across different programming languages and specialized domains.

5. **Feature Coverage**: Actual supported features depend on the vLLM Ascend version and target hardware.

### Safety and Responsible Use

**Important Disclaimers**:

- **Code Verification**: AI-generated code should always be reviewed and tested by qualified developers before deployment to production environments.

- **Security**: The model may generate code with potential security vulnerabilities. Always perform security audits on generated code.

- **Intellectual Property**: Users are responsible for ensuring compliance with applicable licenses and intellectual property rights when using generated code.

- **Accuracy**: While the model strives for accuracy, generated code may contain errors or suboptimal solutions. Always validate functionality.

- **No Substitute for Expertise**: This tool is designed to augment, not replace, professional software engineering judgment.

### Ethical Considerations

- Do not use this model to generate malicious code, exploits, or circumvent security measures
- Respect software licenses and copyright when using generated code
- Consider the environmental impact of large-scale model inference
- Be transparent about AI assistance in code development when required by policy or regulation

## Troubleshooting

### Ascend Graph Mode (ACLGraph) Pitfalls & Blank Outputs

When deploying on Ascend NPUs, you might encounter an issue where the model generates output successfully (tokens/sec > 0), but the printed output consists entirely of **blank spaces or empty lines**. 

Through rigorous debugging (including setting `export INF_NAN_MODE_ENABLE=1`), it has been confirmed that this is **not** caused by typical NaN (Not a Number) tensor pollution. Instead, it is an Ascend Graph compilation compatibility issue with the `bfloat16` data type on certain operators.

**Solution:**
To resolve the blank output issue, you have two options:
1. **Downgrade Precision (Recommended for Production)**: Change `--dtype bfloat16` to `--dtype float16` in your startup script. This bypasses the precision bug in the graph compiler while maintaining graph acceleration.
2. **Enable Eager Mode (Recommended for Testing)**: Add `--enforce-eager` to bypass ACLGraph compilation entirely. This is highly stable but trades off inference speed.

### Common Issues

| Issue | Cause & Solution |
|-------|----------|
| **Blank / Empty Text Output** | **Cause**: Ascend Graph mode bug with `bfloat16` precision (Not a NaN issue). <br>**Solution**: Switch to `--dtype float16` or add `--enforce-eager`. |
| **Out of Memory (OOM)** | **Cause**: NPU memory exhausted. <br>**Solution**: Increase `--tensor-parallel-size`, reduce `--max-model-len`, or lower `--gpu-memory-utilization` (e.g., to 0.8) to leave room for CANN workspace. |
| **Model Loading Errors** | **Cause**: Missing custom code execution permissions. <br>**Solution**: Ensure `--trust-remote-code` is included in the CLI or `trust_remote_code: true` in the YAML config. |
| **404 Not Found in Benchmark** | **Cause**: Benchmark client `--model` name does not match the server's `--served-model-name`. <br>**Solution**: Ensure both names match exactly. |
| **OSError during Benchmark** | **Cause**: Benchmark client tries to fetch the server alias from HuggingFace. <br>**Solution**: Explicitly add `--tokenizer <local_model_path>` to the benchmark script alongside the alias. |
