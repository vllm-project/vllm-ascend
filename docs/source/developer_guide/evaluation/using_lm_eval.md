# Using lm-eval
This document will guide you have a accuracy testing using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).

## Online Server
###  1. start the vLLM server
You can run docker container to start the vLLM server on a single NPU:

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
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
-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-it $IMAGE \
/bin/bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max_model_len 4096
```

### 2. Run ceval accuracy test using lm-eval
Install lm-eval in the container.

```bash
export HF_ENDPOINT="https://hf-mirror.com"
pip install lm-eval[api]
```
Run the following command:

```
# Only test gsm8k dataset in this demo
lm_eval \
  --model local-completions \
  --model_args model=Qwen/Qwen2.5-0.5B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```
After 30 mins, the output is as shown below:

```
The markdown format results is as below:

Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3215|±  |0.0129|
|     |       |strict-match    |     5|exact_match|↑  |0.2077|±  |0.0112|

```
## Offline Server
###  1. Run docker container

You can run docker container on a single NPU:

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
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
-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-it $IMAGE \
/bin/bash
```

### 2. Run ceval accuracy test using lm-eval
Install lm-eval in the container.

```bash
export HF_ENDPOINT="https://hf-mirror.com"
pip install lm-eval
```
Run the following command:

```
# Only test gsm8k dataset in this demo
lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,max_model_len=4096 \
  --tasks gsm8k \
  --batch_size auto
```

After 1-2 mins, the output is as shown below:

```
The markdown format results is as below:

Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3412|±  |0.0131|
|     |       |strict-match    |     5|exact_match|↑  |0.3139|±  |0.0128|

```

## Use offline Datasets

Take gsm8k as an example.

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
cd lm_eval/tasks/gsm8k
```

set gsm8k.yaml as follows:
```
tag:
  - math_word_problems
task: gsm8k

# set dataset_path arrow or json according to the downloaded dataset
dataset_path: arrow

# set dataset_name to null
dataset_name: null
output_type: generate_until

# add dataset_kwargs 
dataset_kwargs:
  data_files:
    # train data download path
    train: /root/.cache/gsm8k-train.arrow
    # test data download path
    test: /root/.cache/gsm8k-test.arrow

training_split: train
fewshot_split: train
test_split: test
doc_to_text: 'Q: {{question}}
  A(Please follow the summarize the result at the end with the format of "The answer is xxx", where xx is the result.):'
doc_to_target: "{{answer}}" #" {{answer.split('### ')[-1].rstrip()}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
      - "\\.$"
generation_kwargs:
  until:
    - "Question:"
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: 5
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "regex"
        group_select: -1
        regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
      - function: "take_first"
metadata:
  version: 3.0
```


You can see more usage on [Lm-eval Docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/README.md).
