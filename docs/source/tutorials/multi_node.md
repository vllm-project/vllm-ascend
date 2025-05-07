# Multi-Node (DeepSeek)

## Online Serving on Multi node

Run docker container on each machine:

```{code-block} bash
   :substitutions:

docker run --rm \
--name vllm-ascend \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2\
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
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
-it quay.io/ascend/vllm-ascend:|vllm_ascend_version| bash
```

(Optional) Install MindIE Turbo for performance acceleration:

```{code-block} bash
   :substitutions:

pip install mindie_turbo==|pip_mindie_turbo_version|
```

Choose one machine as head node, the other are worker nodes, then start ray on each machine:

:::{note}
Check out your `nic_name` by command `ip addr`.
:::

```shell
# Head node
export HCCL_IF_IP={local_ip}
export GLOO_SOCKET_IFNAME={nic_name}
export TP_SOCKET_IFNAME={nic_name}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
ray start --head --num-gpus=8

# Worker node
export HCCL_IF_IP={local_ip}
export ASCEND_PROCESS_LOG_PATH={plog_save_path}
export GLOO_SOCKET_IFNAME={nic_name}
export TP_SOCKET_IFNAME={nic_name}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --address='{head_node_ip}:{port_num}' --num-gpus=8 --node-ip-address={local_ip}
```

:::{note}
If you're running DeepSeek V3/R1, please remove `quantization_config` section in `config.json` file since it's not supported by vllm-ascend currentlly.
:::

Start the vLLM server on head node:

```shell
export VLLM_HOST_IP={head_node_ip}
export HCCL_CONNECT_TIMEOUT=120
export ASCEND_PROCESS_LOG_PATH={plog_save_path}
export HCCL_IF_IP={head_node_ip}

if [ -d "{plog_save_path}" ]; then
    rm -rf {plog_save_path}
    echo ">>> remove {plog_save_path}"
fi

LOG_FILE="multinode_$(date +%Y%m%d_%H%M).log"
VLLM_TORCH_PROFILER_DIR=./vllm_profile
python -m vllm.entrypoints.openai.api_server  \
       --model="Deepseek/DeepSeek-V2-Lite-Chat" \
       --trust-remote-code \
       --enforce-eager \
       --max-model-len {max_model_len} \
       --distributed_executor_backend "ray" \
       --tensor-parallel-size 16 \
       --disable-log-requests \
       --disable-log-stats \
       --disable-frontend-multiprocessing \
       --port {port_num} \
```

Once your server is started, you can query the model with input prompts:

```shell
curl -X POST http://127.0.0.1:{prot_num}/v1/completions  \
     -H "Content-Type: application/json" \
     -d '{
         "model": "Deepseek/DeepSeek-V2-Lite-Chat",
         "prompt": "The future of AI is",
         "max_tokens": 24
     }'
```

If you query the server successfully, you can see the info shown below (client):

```
{"id":"cmpl-6dfb5a8d8be54d748f0783285dd52303","object":"text_completion","created":1739957835,"model":"/home/data/DeepSeek-V2-Lite-Chat/","choices":[{"index":0,"text":" heavily influenced by neuroscience and cognitiveGuionistes. The goalochondria is to combine the efforts of researchers, technologists,","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":6,"total_tokens":30,"completion_tokens":24,"prompt_tokens_details":null}}
```

Logs of the vllm server:

```
INFO:     127.0.0.1:59384 - "POST /v1/completions HTTP/1.1" 200 OK
INFO 02-19 17:37:35 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1.9 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
```
