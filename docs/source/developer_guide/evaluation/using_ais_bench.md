# Using AISBench
This document guides you to conduct accuracy testing using [AISBench](https://gitee.com/aisbench/benchmark/tree/master). AISBench provides accuracy and performance evaluation for many datasets.

## Online Server
### 1. Start the vLLM server
You can run docker container to start the vLLM server on a single NPU:

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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
-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-it $IMAGE \
/bin/bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max_model_len 4096 &
```

The vLLM server is started successfully, if you see logs as below:

```
INFO:     Started server process [9446]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 2. Run C-Eval dataset using AISBench

#### Install AISBench

Refer to [AISBench](https://gitee.com/aisbench/benchmark/tree/master) for details.
Install AISBench from source.

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```

Install extra AISBench dependencies.
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

Run `ais_bench -h` to check the installation.

#### Download Dataset

Take `C-Eval` dataset as an example. And you can refer to [Datasets](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/datasets) for more datasets. Every datasets have a `README.md` for detailed download and installation process.

Download dataset and install it to specific path.
```shell
cd ais_bench/datasets
mkdir ceval/
mkdir ceval/formal_ceval
cd ceval/formal_ceval
wget https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip
unzip ceval-exam.zip
rm ceval-exam.zip
```

#### Update Model Config Python File

Update the file `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`.
There are several arguments that you should update according to your environment.

- `path`: Update to your model weight path.
- `model`: Update to your model name in vLLM.
- `host_ip` and `host_port`: Update to your vLLM server ip and port.
- `max_out_len`: Note `max_out_len` + LLM input length should be less than `max-model-len`(config in your vllm server).
- `batch_size`: Update according to your dataset.
- `temperature`: Update inference argument.

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="xxxx",
        model="xxxx",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8000,
        max_out_len = xxx,
        batch_size = xxx,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.6,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]

```

#### Execute Accuracy Evaluation

Run the following code to execute the accuracy evaluation.

```shell
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt.py --mode all --dump-eval-details --merge-ds
```

After execution, you can get the result from saved files such as `outputs/default/20250628_151326`, there is an example as follows:

```
20250628_151326/
├── configs # Combined configuration file for model tasks, dataset tasks, and result presentation tasks
│   └── 20250628_151326_29317.py
├── logs # Execution logs; if --debug is added to the command, no intermediate logs are saved to disk (all are printed directly to the screen)
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # Logs of the accuracy evaluation process based on inference results in the predictions/ folder
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # Logs of the inference process
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Inference results (all outputs returned by the inference service)
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Raw scores calculated from the accuracy evaluation
└── summary
    ├── summary_20250628_151326.csv # Final accuracy scores (in table format)
    ├── summary_20250628_151326.md # Final accuracy scores (in Markdown format)
    └── summary_20250628_151326.txt # Final accuracy scores (in text format)
```


#### Execute Performance Evaluation

```shell
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt.py --summarizer default_perf --mode perf
```

After execution, you can get the result from saved files, there is an example as follows:

```
20251031_070226/
|-- configs # Combined configuration file for model tasks, dataset tasks, and result presentation tasks
|   `-- 20251031_070226_122485.py
|-- logs
|   `-- performances
|       `-- vllm-api-general-chat
|           `-- cevaldataset.out # Logs of the performance evaluation process
`-- performances
    `-- vllm-api-general-chat
        |-- cevaldataset.csv # Final performance results (in table format)
        |-- cevaldataset.json # Final performance results (in json format)
        |-- cevaldataset_details.h5 # Final performance results in details
        |-- cevaldataset_details.json # Final performance results in details
        |-- cevaldataset_plot.html # Final performance results (in html format)
        `-- cevaldataset_rps_distribution_plot_with_actual_rps.html # Final performance results (in html format)
```