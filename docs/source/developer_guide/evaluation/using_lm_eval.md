# Using lm-eval
This document will guide you have a accuracy testing using [lm-eval][1].

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
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max_model_len 4096 &
```

Started the vLLM server successfully,if you see log as below:
```
INFO:     Started server process [9446]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 2. Run gsm8k accuracy test using lm-eval


You can query result with input prompts:
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "'"<|im_start|>system\n你是一名专业会计师，请根据会计知识回答问题，只输出选项字母（A/B/C/D）。<|im_end|>\n"\
"<|im_start|>user\n题目：某公司2023年12月31日资产负债表显示：\n"\
"  流动资产：现金及等价物 500万元，应收账款 800万元，存货 600万元\n"\
"  非流动资产：固定资产净值 1200万元\n"\
"  流动负债：短期借款 400万元，应付账款 300万元\n"\
"  非流动负债：长期借款 900万元\n"\
"  所有者权益：实收资本 1000万元，未分配利润 ？\n"\
"要求：计算该公司的资产负债率和流动比率（结果保留两位小数）。\n"\
"选项：\n"\
"A. 资产负债率=58.33%，流动比率=1.90\n"\
"B. 资产负债率=62.50%，流动比率=2.17\n"\
"C. 资产负债率=65.22%，流动比率=1.75\n"\
"D. 资产负债率=68.00%，流动比率=2.50<|im_end|>\n"\
"<|im_start|>assistant\n"'",
        "max_tokens": 1,
        "temperature": 0,
        "stop": ["<|im_end|>"]
    }' | python3 -m json.tool
```

The output format matches the following:
```
{
    "id": "cmpl-4a22963b6cca4ba4be3badf9e8a18545",
    "object": "text_completion",
    "created": 1751945850,
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "choices": [
        {
            "index": 0,
            "text": "C",
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null,
            "prompt_logprobs": null
        }
    ],
    "usage": {
        "prompt_tokens": 271,
        "total_tokens": 272,
        "completion_tokens": 1,
        "prompt_tokens_details": null
    },
    "kv_transfer_params": null
}
```

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

### 2. Run gsm8k accuracy test using lm-eval
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

Take gsm8k(single dataset) and mmlu(multi-subject dataset) as examples, and you can see more from [here][2].

```bash
# set HF_DATASETS_OFFLINE when using offline datasets
export HF_DATASETS_OFFLINE=1
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
# gsm8k yaml path
cd lm_eval/tasks/gsm8k
# mmlu yaml path
cd lm_eval/tasks/mmlu/default
```

set [gsm8k.yaml][3] as follows:
```yaml
tag:
  - math_word_problems
task: gsm8k

# set dataset_path arrow or json or parquet according to the downloaded dataset
dataset_path: arrow

# set dataset_name to null
dataset_name: null
output_type: generate_until

# add dataset_kwargs 
dataset_kwargs:
  data_files:
    # train and test data download path
    train: /root/.cache/gsm8k/gsm8k-train.arrow
    test: /root/.cache/gsm8k/gsm8k-test.arrow

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

set [_default_template_yaml][4] as follows:
```yaml
# set dataset_path according to the downloaded dataset
dataset_path: /root/.cache/mmlu
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
dataset_kwargs:
  trust_remote_code: true
```


You can see more usage on [Lm-eval Docs][5].

[1]: https://github.com/EleutherAI/lm-evaluation-harness
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#using-local-datasets
[3]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
[4]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml
[5]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/README.md