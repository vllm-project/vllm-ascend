# Using lm-eval
This document will guide you have a accuracy testing using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).

##  1. Run docker container

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

## 2. Run ceval accuracy test using lm-eval
Install lm-eval in the container.

```bash
pip install lm-eval
```
Run the following command:

```
# Only test ceval-valid-computer_network dataset in this demo
lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,max_model_len=4096,block_size=4,tensor_parallel_size=1 \
  --tasks ceval-valid_computer_network \
  --batch_size 8
```

After 1-2 mins, the output is as shown below:

```
The markdown format results is as below:

|           Tasks            |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|----------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|ceval-valid_computer_network|      2|none  |     0|acc     |↑  |0.6842|±  |0.1096|
|                            |       |none  |     0|acc_norm|↑  |0.6842|±  |0.1096|

```

You can see more usage on [Lm-eval Docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/README.md).
