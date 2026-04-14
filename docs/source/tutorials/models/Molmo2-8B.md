# Molmo2-8B (Molmo2ForConditionalGeneration)

## Introduction

[Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) is an open vision–language model from the Allen Institute for AI (Ai2). It uses a **Qwen3-8B** language backbone and **SigLIP 2** as the vision encoder, and supports **image** and **video** inputs with pointing and dense captioning-style outputs. In upstream vLLM it is registered as `Molmo2ForConditionalGeneration`; see the [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html) list.

This tutorial summarizes how to prepare the environment on Ascend, run **offline inference** and an **OpenAI-compatible server**, and points to the **nightly accuracy** configuration used in this repository (`mmmu_val`).

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) for the feature matrix.

Refer to the [feature guide](../../user_guide/feature_guide/index.md) for configuration details.

## Environment Preparation

### Model weights

- Hugging Face: [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B)
- Optional mirror: set `export VLLM_USE_MODELSCOPE=true` if you pull weights via ModelScope equivalents.

Weights are large; use a shared cache (for example `/root/.cache/huggingface`) on the host and mount it into the container when using Docker.

### Installation

Use the same Ascend driver/CANN stack and vLLM Ascend image as in [Installation](../../installation.md) and [Quickstart](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html).

Example (single NPU, adapt image tag to your release):

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -p 8000:8000 \
  -it $IMAGE bash
```

Recommended environment variables:

```bash
export VLLM_USE_MODELSCOPE=true   # optional, for faster downloads in CN regions
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

## Deployment

### Offline inference (Python API)

Align engine settings with upstream examples: `trust_remote_code=True`, `dtype=bfloat16`, and a generous `max_num_batched_tokens` when long visual sequences are used.

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

model_name = "allenai/Molmo2-8B"
question = "What is in this image?"
image = ImageAsset("cherry_blossom")  # built-in demo asset; replace with your PIL / path pipeline

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image.pil_image},
            {"type": "text", "text": question},
        ],
    }
]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    dtype="bfloat16",
    limit_mm_per_prompt={"image": 1},
    max_num_batched_tokens=36864,
)
out = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": [image.pil_image]},
    },
    sampling_params=SamplingParams(max_tokens=128, temperature=0),
)
print(out[0].outputs[0].text)
```

You can also use the upstream script [vision_language_multi_image.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language_multi_image.py) with `--model-type molmo2`.

### OpenAI-compatible server

```bash
vllm serve allenai/Molmo2-8B \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-num-batched-tokens 36864 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 1
```

Then call `/v1/chat/completions` with an image URL or base64 payload in the message `content` array (same structure as in the offline `apply_chat_template` example).

:::{note}
On a single **64G** NPU, if you hit OOM, lower `max_model_len`, reduce `max_num_batched_tokens`, or serve with `tensor_parallel_size` across more NPUs.
:::

## Accuracy and CI

Nightly accuracy for this model is configured in `tests/e2e/models/configs/Molmo2-8B.yaml` using the `mmmu_val` task. The reference `acc,none` target is aligned with the public **~53% MMMU** figure for Molmo2-8B; if the lm-eval harness on Ascend differs, maintainers may adjust the YAML baseline within the test’s relative tolerance.

## Local Verification Report

### Test Steps

1. Create a quick-check config at `tests/e2e/models/configs/Molmo2-8B-quick.yaml` with `enforce_eager: True`, reduced `limit`, and conservative memory settings for triage.
2. Stage-1 run (online model path) with timeout guard:

```bash
HF_HOME=/data/huggingface_home TRANSFORMERS_CACHE=/data/huggingface_home \
timeout 45m /data/vllm-ascend-env/bin/python -m pytest -sv \
tests/e2e/models/test_lm_eval_correctness.py::test_lm_eval_correctness_param \
--config tests/e2e/models/configs/Molmo2-8B-quick.yaml \
--tp-size 1 \
--report-dir /tmp/molmo2_verify_quick
```

3. Stage-2 run (local model path after full pre-download):

```bash
timeout 60m /data/vllm-ascend-env/bin/python -m pytest -sv \
tests/e2e/models/test_lm_eval_correctness.py::test_lm_eval_correctness_param \
--config tests/e2e/models/configs/Molmo2-8B-quick.yaml \
--tp-size 1 \
--report-dir /tmp/molmo2_verify_quick_local_retry2
```

4. Observe pytest logs and NPU runtime status (`npu-smi info`) during model load and evaluation.

5. Stage-4 run (official config after adaptation):

```bash
timeout 120m /data/vllm-ascend-env/bin/python -m pytest -sv \
tests/e2e/models/test_lm_eval_correctness.py::test_lm_eval_correctness_param \
--config tests/e2e/models/configs/Molmo2-8B.yaml \
--tp-size 1 \
--report-dir /tmp/molmo2_verify_full
```

### Error Logs

- Stage-1 (network):
  - `Error while downloading ... model-00005-of-00008.safetensors ... Read timed out.`
  - `Trying to resume download...`
  - timeout exit: `exit_code: 124`
- Stage-2 (runtime before patch):
  - `TypeError: 'function' object is not subscriptable`
  - crash path in `vllm_ascend/worker/block_table.py` at `_compute_slot_mapping_kernel[(...)](...)`
- Stage-3 (after fallback patch):
  - first rerun failed with prompt length guard:
    - `ValueError: The decoder prompt (length 4416) is longer than the maximum model length of 4096.`
  - after adjusting quick config `max_model_len` to `8192`, run passed and produced metric:
    - `mmmu_val | acc,none: ground_truth=0.53 | measured=0.5267 | success=✅`
- Stage-4 (official config after adaptation):
  - full config (`tests/e2e/models/configs/Molmo2-8B.yaml`) passed:
    - `mmmu_val | acc,none: ground_truth=0.53 | measured=0.5344 | success=✅`

### Root Cause Analysis

Two-stage diagnosis was completed:

1. **Stage-1 (network bottleneck)**: online model download from Hugging Face CAS/Xet timed out repeatedly.
2. **Stage-2 (runtime failure after local weights were fully prepared)**: with local model path `/data/models/allenai/Molmo2-8B`, eval proceeded into `generate_until` and then crashed in EngineCore:
   - `TypeError: 'function' object is not subscriptable`
   - stack path:
     - `vllm_ascend/worker/block_table.py`
     - `_compute_slot_mapping_kernel[(num_reqs + 1,)](...)`
   - this indicates the expected Triton-style kernel launcher object is actually a plain Python function in current runtime/backend combination.
3. **Stage-3 (after patch and config correction)**:
   - added launcher compatibility fallback in `vllm_ascend/worker/block_table.py`;
   - increased quick-check `max_model_len` from `4096` to `8192`;
   - model completed quick lm-eval successfully.
4. **Stage-4 (official config validated)**:
   - set `enforce_eager: True` in `tests/e2e/models/configs/Molmo2-8B.yaml` to avoid torch-dynamo import path on this runtime;
   - full MMMU eval completed successfully with measured accuracy close to baseline.

### GAP Conclusion

- **Environment/data-path GAP**: Confirmed (large model online shard download is unstable; local pre-download is strongly recommended for verification).
- **Runtime compatibility GAP**: Confirmed and mitigated by fallback launcher path in `block_table` slot mapping.
- **Current verification status**: Official config is **runnable** on Ascend and completed MMMU evaluation (`measured=0.5344`).

### Adaptation Development Plan (If Still Not Runnable)

1. Keep local pre-download workflow for Molmo2-8B and validate all shard files before pytest.
2. Keep `block_table` launcher compatibility fallback and add a regression test for this path.
3. Run `Molmo2-8B-quick.yaml` as precheck in CI/dev before full config.
4. Run full `tests/e2e/models/configs/Molmo2-8B.yaml` on stable network/storage; update baseline only if full-run measured value deviates beyond test tolerance.

## References

- [Molmo2 announcement (Ai2)](https://allenai.org/blog/molmo2)
- [vLLM Ascend documentation](https://docs.vllm.ai/projects/ascend/en/latest/)
- [Molmo2-8B model card](https://huggingface.co/allenai/Molmo2-8B)
