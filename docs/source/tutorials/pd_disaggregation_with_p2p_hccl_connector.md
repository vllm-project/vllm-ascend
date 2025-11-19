# Prefill-Decode Disaggregation With P2pHcclConnector

## Quick Start for 1p1d

Setup environment variables:

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1 # modify according to your environment
modelName=xx # use your own model
servedModelName=xxx # use your own model name
```

Start vllm instance:

```shell
# prefill instance
vllm serve ${modelName} --host 0.0.0.0 --port 20005 --seed 42 --served-model-name ${servedModelName} --max-model-len 10000 --max-num-seqs 256 --gpu-memory-utilization 0.7 --disable-log-request --kv-transfer-config \{\"kv_rank\":0,\"kv_connector\":\"P2pHcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"21001\",\"kv_connector_extra_config\":\{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"http_port\":\"20005\",\"send_type\":\"PUT_ASYNC\"}\} > prefill.log &

# decode instance
vllm serve ${modelName} --host 0.0.0.0 --port 20009 --seed 1024 --served-model-name  ${servedModelName} --max-model-len 10000 --max-num-seqs 256 --gpu-memory-utilization 0.7 --disable-log-request --kv-transfer-config \{\"kv_rank\":1,\"kv_connector\":\"P2pHcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"22001\",\"kv_connector_extra_config\":\{\"mem_pool_size_gb\":256,\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"http_port\":\"20009\",\"send_type\":\"PUT_ASYNC\"\}\} > decode.log &
```

Start the disaggregate prefill proxy in vllm project:

```shell
python {vllm project directory}/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py > proxy.log &
```

Send a test request:

```shell
curl  http://localhost:10001/v1/completions     -H "Content-Type: application/json"     -d '{
    "model": "'$servedModelName'",
    "prompt": "Why is the sky blue?",
    "max_tokens": 100
}'
```

Since the P2pHcclConnector is based on P2pNcclConnector, you can find more details on [document for p2p_nccl_connector](https://github.com/vllm-project/vllm/blob/main/docs/design/p2p_nccl_connector.md).
