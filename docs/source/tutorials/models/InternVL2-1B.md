# OpenGVLab/InternVL2-1B

## Overview

**InternVL2-1B** is a state-of-the-art vision-language model (VLM) developed by OpenGVLab. With approximately 1 billion parameters, this model excels at understanding and reasoning about visual content, enabling seamless integration of image and text understanding capabilities.

Built upon the InternVL series architecture, InternVL2-1B represents an efficient yet powerful multimodal model suitable for a wide range of computer vision and natural language processing tasks.

### Key Features

| Feature | Description |
|---------|-------------|
| **Architecture** | 1B parameter vision-language transformer |
| **Context Window** | Up to 32,768 tokens |
| **Precision** | BF16 (BFloat16) optimized for Ascend NPU |
| **Multimodal** | Supports image and text inputs simultaneously |
| **Specialization** | Visual question answering, image captioning, document understanding, multimodal reasoning |
| **Deployment** | Optimized for vLLM Ascend with single-card deployment support |

### Model Capabilities

- **Visual Question Answering**: Answer complex questions about image content
- **Image Captioning**: Generate detailed natural language descriptions of images
- **Document Understanding**: Process and understand document images, charts, and figures
- **Multimodal Reasoning**: Combine visual and textual information for comprehensive understanding
- **Object Detection**: Identify and locate objects within images
- **Chart Analysis**: Interpret and explain data visualizations

## Intended Use Cases

### Primary Applications

1. **Document Processing**
   - Automated document digitization and understanding
   - Invoice and receipt processing
   - Form field extraction
   - Report generation from visual data

2. **Visual Content Analysis**
   - Social media content moderation
   - E-commerce product image analysis
   - Medical image interpretation assistance
   - Satellite and aerial imagery analysis

3. **Educational Tools**
   - Interactive learning with visual content
   - Accessibility services for visually impaired users
   - Diagram and flowchart interpretation
   - Educational content generation

4. **Enterprise Solutions**
   - Customer support with visual context
   - Quality control in manufacturing
   - Video frame analysis
   - Multimodal search and retrieval

### Target Users

- Data scientists and ML engineers
- Document processing specialists
- Healthcare professionals (with appropriate oversight)
- E-commerce and retail businesses
- Educational institutions
- Enterprise AI teams

## Performance Benchmarks

InternVL2-1B demonstrates competitive performance on industry-standard multimodal benchmarks:

| Benchmark | Description | Performance |
|-----------|-------------|-------------|
| MMMU | Massive Multimodal Understanding | Strong |
| TextVQA | Text-based Visual Question Answering | High-performing |
| ChartQA | Chart Understanding | Competitive |
| DocVQA | Document Visual Question Answering | High-performing |

> **Note**: Exact benchmark scores may vary based on evaluation configuration and hardware setup. For the most current performance metrics, refer to the official [InternVL model card](https://huggingface.co/OpenGVLab/InternVL2-1B).

## Installation and Setup

### Prerequisites

- Ascend NPU hardware (A2 or A3 series)
- Docker environment configured for Ascend
- Single NPU sufficient for InternVL2-1B due to compact model size

### Model Weight Download

Download the model weights from Hugging Face:

```bash
# Model: OpenGVLab/InternVL2-1B
# URL: https://huggingface.co/OpenGVLab/InternVL2-1B
```

> **Important**: The public model files use custom model and tokenizer code, requiring `--trust-remote-code` during deployment.

It is recommended to download the model weights to a shared local directory such as `/data/huggingface_home/` before deployment.

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

`InternVL2-1B` is a BF16 vision-language model with an 8K maximum context length. Due to its compact size (1B parameters), single-card deployment is supported.

Create a deployment script (`deploy_InternVL2_1B.sh`):

```shell
#!/bin/sh
export HF_HOME=/data/huggingface_home
export HF_ENDPOINT=https://hf-mirror.com
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH="OpenGVLab/InternVL2-1B"

vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name InternVL2-1B \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192
```

**Configuration Notes**:
- `ASCEND_RT_VISIBLE_DEVICES=0` - Single NPU is sufficient for 1B model
- `--tensor-parallel-size 1` - No tensor parallelism needed due to model size
- `--max-model-len 8192` - Vision-language models typically use shorter contexts
- `--enforce-eager` - Recommended for vision models to ensure stable memory usage

### Multimodal Input Format

InternVL2-1B accepts both text and image inputs. Example API request with image:

```shell
curl http://<IP>:<Port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "InternVL2-1B",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {"type": "text", "text": "Describe this image in detail."}
            ]}
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
    model="InternVL2-1B",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {"type": "text", "text": "What does this image show?"}
            ]
        }
    ],
    temperature=0.2,
    max_tokens=512
)

print(response.choices[0].message.content)
```

### Sample Use Cases

1. **Document Understanding**
   ```
   "Extract all text and numbers from this receipt image."
   ```

2. **Visual Question Answering**
   ```
   "What is unusual about the composition of this photograph?"
   ```

3. **Chart Analysis**
   ```
   "Explain the trends shown in this bar chart."
   ```

## Functional Verification

After service startup, verify functionality with domain-specific requests:

### Visual QA Verification

```shell
curl http://<IP>:<Port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "InternVL2-1B",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/sample.png"}},
                {"type": "text", "text": "Describe this image in detail."}
            ]}
        ],
        "temperature": 0,
        "max_tokens": 256
    }'
```

A valid response should contain a coherent description of the image content with relevant details.

## Accuracy Evaluation

### Using LM Evaluation Harness

An evaluation configuration can be created for InternVL2-1B:

```shell
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
    --config tests/e2e/models/configs/InternVL2-1B.yaml
```

## Performance Benchmarking

### Using vLLM Benchmark Tools

After confirming service stability, run performance tests:

```shell
vllm bench serve \
  --model OpenGVLab/InternVL2-1B \
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
- For vision models, image size significantly impacts memory usage
- Monitor NPU memory utilization throughout testing
- Establish baseline metrics before production deployment

## Supported Features

For a complete feature compatibility matrix, refer to [supported features](../../user_guide/support_matrix/supported_models.md).

For feature configuration options, see the [feature guide](../../user_guide/feature_guide/index.md).

## Limitations and Safety Guidelines

### Known Limitations

1. **Image Resolution**: Performance may degrade with very high-resolution images. Consider resizing or using appropriate downsampling.

2. **Custom Code Dependency**: The public Hugging Face model uses custom model and tokenizer code, requiring `trust_remote_code=True` during deployment.

3. **Context Window**: While the model supports up to 32K tokens, actual usable context depends on available NPU memory and image sizes.

4. **Language Support**: Best performance is achieved in English and Chinese. Other languages may have degraded performance.

5. **Feature Coverage**: Actual supported features depend on the vLLM Ascend version and target hardware.

### Safety and Responsible Use

**Important Disclaimers**:

- **Visual Content**: AI-generated descriptions should be verified by humans before critical decisions.

- **Medical/Technical Images**: Do not use for medical diagnosis or technical inspection without expert oversight.

- **Accuracy**: While the model strives for accuracy, generated descriptions may contain errors or misinterpretations.

- **No Substitute for Expertise**: This tool is designed to augment, not replace, professional judgment in visual analysis tasks.

### Ethical Considerations

- Do not use this model for surveillance or invasive monitoring purposes
- Respect privacy when processing images containing individuals
- Consider the environmental impact of large-scale model inference
- Be transparent about AI assistance in visual analysis when required by policy or regulation

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `--max-model-len` or use smaller image inputs |
| Slow Inference | Verify NPU utilization and consider batching requests |
| Model Loading Errors | Confirm `--trust-remote-code` is enabled |
| Connection Issues | Check firewall settings and port availability |
| Image Loading Failures | Verify image URLs are accessible or use local images |

## Support and Resources

- **Model Card**: [Hugging Face - InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B)
- **Project Page**: [OpenGVLab](https://huggingface.co/OpenGVLab)
- **vLLM Ascend Documentation**: [Official Documentation](../../user_guide/)
- **Issue Reporting**: Please report issues through the appropriate GitHub repositories

## License and Attribution

Please refer to the model card on Hugging Face for specific licensing terms and attribution requirements.

---

**Version**: 1.0
**Last Updated**: March 2026
**Maintained by**: OpenGVLab and vLLM Ascend Community