# Using OpenCompass 
This document will guide you have a accuracy testing using [OpenCompass](https://github.com/open-compass/opencompass).

## 1. Online Serving

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
vllm serve Qwen/Qwen2.5-7B-Instruct --max_model_len 26240
```
If your service start successfully, you can see the info shown below:
```
INFO:     Started server process [6873]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts in new terminal:
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "The future of AI is",
        "max_tokens": 7,
        "temperature": 0
    }'
```

## 2. Run ceval accuracy test using OpenCompass
Install OpenCompass and configure the environment variables in the container.

```bash
# Pin Python 3.10 due to:
# https://github.com/open-compass/opencompass/issues/1976
conda create -n opencompass python=3.10
conda activate opencompass
pip install opencompass modelscope[framework]
export DATASET_SOURCE=ModelScope
git clone https://github.com/open-compass/opencompass.git
```

Add `opencompass/configs/eval_vllm_ascend_demo.py` with the following content:

```python
from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets

# Only test ceval-computer_network dataset in this demo
datasets = ceval_datasets[:1]

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='Qwen2.5-7B-Instruct-vLLM-API',
        type=OpenAISDK,
        key='EMPTY', # API key
        openai_api_base='http://127.0.0.1:8000/v1', 
        path='Qwen/Qwen2.5-7B-Instruct', 
        tokenizer_path='Qwen/Qwen2.5-7B-Instruct', 
        rpm_verbose=True, 
        meta_template=api_meta_template,
        query_per_second=1, 
        max_out_len=1024, 
        max_seq_len=4096, 
        temperature=0.01, 
        batch_size=8,
        retry=3,
    )
]
```

Run the following command:

```
python3 run.py opencompass/configs/eval_vllm_ascend_demo.py --debug
```

After 1-2 mins, the output is as shown below:

```
The markdown format results is as below:

| dataset | version | metric | mode | Qwen2.5-7B-Instruct-vLLM-API |
|----- | ----- | ----- | ----- | -----|
| ceval-computer_network | db9ce2 | accuracy | gen | 68.42 |
```

You can see more usage on [OpenCompass Docs](https://opencompass.readthedocs.io/en/latest/index.html).
