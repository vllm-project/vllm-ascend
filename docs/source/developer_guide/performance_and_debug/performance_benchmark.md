# Performance Benchmark

This document details the benchmark methodology for vllm-ascend, aimed at evaluating the performance under a variety of workloads. To maintain alignment with vLLM, we use the [benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks) script provided by the vllm project.

**Benchmark Coverage**: We measure offline E2E latency and throughput, and fixed-QPS online serving benchmarks. For more details, see [vllm-ascend benchmark scripts](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks).

## 1. Run docker container

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device $DEVICE \
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
-e VLLM_USE_MODELSCOPE=True \
-it $IMAGE \
/bin/bash
```

## 2. Install dependencies

```bash
cd /workspace/vllm-ascend
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install -r benchmarks/requirements-bench.txt
```

## 3. Run basic benchmarks

This section introduces how to perform performance testing using the benchmark suite built into VLLM.

### 3.1 Dataset

VLLM supports a variety of [datasets](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/datasets.py).

<style>
th {
  min-width: 0 !important;
}
</style>

| Dataset | Online | Offline | Data Path |
|---------|--------|---------|-----------|
| ShareGPT | ✅ | ✅ | `wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` |
| ShareGPT4V (Image) | ✅ | ✅ | `wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json`<br>Note that the images need to be downloaded separately. For example, to download COCO's 2017 Train images:<br>`wget http://images.cocodataset.org/zips/train2017.zip` |
| ShareGPT4Video (Video) | ✅ | ✅ | `git clone https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video` |
| BurstGPT | ✅ | ✅ | `wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv` |
| Sonnet (deprecated) | ✅ | ✅ | Local file: `benchmarks/sonnet.txt` |
| Random | ✅ | ✅ | `synthetic` |
| RandomMultiModal (Image/Video) | 🟡 | 🚧 | `synthetic` |
| RandomForReranking | ✅ | ✅ | `synthetic` |
| Prefix Repetition | ✅ | ✅ | `synthetic` |
| HuggingFace-VisionArena | ✅ | ✅ | `lmarena-ai/VisionArena-Chat` |
| HuggingFace-MMVU | ✅ | ✅ | `yale-nlp/MMVU` |
| HuggingFace-InstructCoder | ✅ | ✅ | `likaixin/InstructCoder` |
| HuggingFace-AIMO | ✅ | ✅ | `AI-MO/aimo-validation-aime`, `AI-MO/NuminaMath-1.5`, `AI-MO/NuminaMath-CoT` |
| HuggingFace-Other | ✅ | ✅ | `lmms-lab/LLaVA-OneVision-Data`, `Aeala/ShareGPT_Vicuna_unfiltered` |
| HuggingFace-MTBench | ✅ | ✅ | `philschmid/mt-bench` |
| HuggingFace-Blazedit | ✅ | ✅ | `vdaita/edit_5k_char`, `vdaita/edit_10k_char` |
| Spec Bench | ✅ | ✅ | `wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl` |
| Custom | ✅ | ✅ | Local file: `data.jsonl` |

:::{note}
The datasets mentioned above are all links to datasets on huggingface.
The dataset's `dataset-name` should be set to `hf`.
For local `dataset-path`, please set `hf-name` to its Hugging Face ID like

```bash
--dataset-path /datasets/VisionArena-Chat/ --hf-name lmarena-ai/VisionArena-Chat
```

:::

### 3.2 Run basic benchmark

#### 3.2.1 Online serving

First start serving your model:

```bash
VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-8B
```

Then run the benchmarking script:

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
export VLLM_USE_MODELSCOPE=True
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

If successful, you will see the following output:

```shell
============ Serving Benchmark Result ============
Successful requests:                     10        
Failed requests:                         0         
Benchmark duration (s):                  19.92     
Total input tokens:                      1374      
Total generated tokens:                  2663      
Request throughput (req/s):              0.50      
Output token throughput (tok/s):         133.67    
Peak output token throughput (tok/s):    312.00    
Peak concurrent requests:                10.00     
Total Token throughput (tok/s):          202.64    
---------------Time to First Token----------------
Mean TTFT (ms):                          127.10    
Median TTFT (ms):                        136.29    
P99 TTFT (ms):                           137.83    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.85     
Median TPOT (ms):                        25.78     
P99 TPOT (ms):                           26.64     
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.78     
Median ITL (ms):                         25.74     
P99 ITL (ms):                            28.85     
==================================================
```

#### 3.2.2 Offline Throughput Benchmark

```bash
VLLM_USE_MODELSCOPE=True
vllm bench throughput \
  --model Qwen/Qwen3-8B \
  --dataset-name random \
  --input-len 128 \
  --output-len 128
```

If successful, you will see the following output

```shell
Processed prompts: 100%|█| 10/10 [00:03<00:00,  2.74it/s, est. speed input: 351.02 toks/s, output: 351.02 t
Throughput: 2.73 requests/s, 699.93 total tokens/s, 349.97 output tokens/s
Total num prompt tokens:  1280
Total num output tokens:  1280
```

#### 3.2.4 Multi-Modal Benchmark

```shell
export VLLM_USE_MODELSCOPE=True
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --allowed-local-media-path /path/to/sharegpt4v/images
```

```shell
export HF_ENDPOINT="https://hf-mirror.com"
vllm bench serve --model Qwen/Qwen2.5-VL-7B-Instruct \
--backend "openai-chat" \
--dataset-name hf \
--hf-split train \
--endpoint "/v1/chat/completions" \
--dataset-path "lmarena-ai/vision-arena-bench-v0.1" \
--num-prompts 10 \
--no-stream
```

```shell
============ Serving Benchmark Result ============
Successful requests:                     10        
Failed requests:                         0         
Benchmark duration (s):                  4.89      
Total input tokens:                      7191      
Total generated tokens:                  951       
Request throughput (req/s):              2.05      
Output token throughput (tok/s):         194.63    
Peak output token throughput (tok/s):    290.00    
Peak concurrent requests:                10.00     
Total Token throughput (tok/s):          1666.35   
---------------Time to First Token----------------
Mean TTFT (ms):                          722.22    
Median TTFT (ms):                        589.81    
P99 TTFT (ms):                           1377.02   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.13     
Median TPOT (ms):                        34.58     
P99 TPOT (ms):                           124.72    
---------------Inter-token Latency----------------
Mean ITL (ms):                           33.14     
Median ITL (ms):                         28.01     
P99 ITL (ms):                            182.28    
==================================================
```

#### 3.2.5 Embedding Benchmark

```shell
vllm serve Qwen/Qwen3-Embedding-8B --trust-remote-code
```

```shell
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
export VLLM_USE_MODELSCOPE=true
vllm bench serve \
  --model Qwen/Qwen3-Embedding-8B \
  --backend openai-embeddings \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --num-prompts 10 \
  --dataset-path <your dataset path>/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

```shell
============ Serving Benchmark Result ============
Successful requests:                     10        
Failed requests:                         0         
Benchmark duration (s):                  0.18      
Total input tokens:                      1372      
Request throughput (req/s):              56.32     
Total Token throughput (tok/s):          7726.76   
----------------End-to-end Latency----------------
Mean E2EL (ms):                          154.06    
Median E2EL (ms):                        165.57    
P99 E2EL (ms):                           166.66    
==================================================
```
