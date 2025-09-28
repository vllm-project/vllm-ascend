# Mooncake connector deployment Guide

## Environmental Dependencies

 *  Software:
     *  Python >= 3.9, < 3.12
     *  CANN >= 8.2.rc1
     *  PyTorch >= 2.7.1, torch-npu >= 2.7.1.dev20250724
     *  vLLM (same version as vllm-ascend)
     *  mooncake-transfer-engine reference documentation: https://github.com/kvcache-ai/Mooncake/blob/main/doc/zh/ascend_transport.md

The vllm version must be the same as the main branch of vllm-ascend, for example, 2025/07/30. The version is

 *  vllm: v0.10.1
 *  vllm-ascend: v0.10.1rc1

## run

### 1.Run `prefill` Node

```
bash run_prefill.sh
```

Content of the run_prefill.sh script

```
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=localhost
export GLOO_SOCKET_IFNAME="xxxxxx"
export TP_SOCKET_IFNAME="xxxxxx"
export HCCL_SOCKET_IFNAME="xxxxxx"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PHYSICAL_DEVICES=$(ls /dev/davinci* 2>/dev/null | grep -o '[0-9]\+' | sort -n | paste -sd',' -)

vllm serve "/xxxxx/DeepSeek-V2-Lite-Chat" \
  --host localhost \
  --port 8100 \
  --tensor-parallel-size 2\
  --seed 1024 \
  --max-model-len 2000  \
  --max-num-batched-tokens 2000  \
  --trust-remote-code \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-address localhost \
  --data-parallel-rpc-port 9100 \
  --gpu-memory-utilization 0.8  \
  --kv-transfer-config  \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_rank": 0,
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 2
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 2
             }
      }
  }'
```

`HCCL_EXEC_TIMEOUT`, `HCCL_CONNECT_TIMEOUT`, and `HCCL_IF_IP` are hccl-related configurations.<br>
Set `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `HCCL_SOCKET_IFNAME` to the corresponding NIC.<br>
`ASCEND_RT_VISIBLE_DEVICES` specifies the cards on which the node run resides. The total number of cards equals `dp_size*tp_size`.<br>
`/xxxxx/DeepSeek-V2-Lite-Chat` is configured as a model that requires run.<br>
`--host`: indicates the IP address of the node to be started.<br>
`--port`: indicates the port to be started, which corresponds to the port in step 4.<br>
`--seed`, --max-model-len, and --max-num-batched-tokens model basic configuration. Set this parameter based on the site requirements.<br>
`--tensor-parallel-size`: specifies the TP size.<br>
`--data-parallel-size`: indicates the DP size.<br>
`--data-parallel-address`: indicates the IP address of the DP. Set this parameter to the IP address of the node.--data-parallel-rpc-port: indicates the RPC port for communication in the DP group.<br>
`--trust-remote-code` can load the local model.<br>
`--enforce-eager` Turn off the map mode<br>
`--gpu-memory-utilization`: Percentage of video memory occupied by the card<br>
`--kv-transfer-config`: follow kv_connector, kv_connector_module_path: mooncakeconnect, kv_buffer_device, and run on the NPU card. For kv_role, set kv_producer to the p node, kv_consumer to the d node, kv_parallel_size to 1, and kv_port to the port used by the node. For the p node, set engine_id and kv_rank to 0 and for the d node to 1. Configure the distributed parallel policy for the p and d nodes in the kv_connector_extra_config file based on --tensor-parallel-size and --data-parallel-size.<br>


### 2. Run `decode` Node

```
bash run_decode.sh
```

Content of the run_decode.sh script

```
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=localhost
export GLOO_SOCKET_IFNAME="xxxxxx"
export TP_SOCKET_IFNAME="xxxxxx"
export HCCL_SOCKET_IFNAME="xxxxxx"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export PHYSICAL_DEVICES=$(ls /dev/davinci* 2>/dev/null | grep -o '[0-9]\+' | sort -n | paste -sd',' -)

vllm serve "/xxxxx/DeepSeek-V2-Lite-Chat" \
  --host localhost \
  --port 8200 \
  --tensor-parallel-size 2\
  --seed 1024 \
  --max-model-len 2000  \
  --max-num-batched-tokens 2000  \
  --trust-remote-code \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-address localhost \
  --data-parallel-rpc-port 9100 \
  --gpu-memory-utilization 0.8  \
  --kv-transfer-config  \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20002",
  "engine_id": "1",
  "kv_rank": 1,
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 2
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 2
             }
      }
  }'
```

### 3. Start proxy_server. ###

```
cd /vllm-ascend/examples/disaggregate_prefill_v1/
python load_balance_proxy_server_example.py --host localhost --prefiller-hosts host1 host2 --prefiller-ports 8100 8101 --decoder-hosts host3 host4 --decoder-ports 8200 8201
```

`--host`: indicates the active node. The value of localhost in the curl command delivered in step 5 must be the same as the host. The default port number for starting the service proxy is 8000.<br>
`--prefiller-hosts`: Set this parameter to the IP addresses of all p nodes. In the xpyd scenario, add the IP addresses to the end of this configuration item and leave a blank space between the IP addresses.<br>
`--prefiller-ports`: Set this parameter to the port number of all p nodes, which is the configuration of the port number for the vllm to start the service in step 3. Write the port number after the configuration in sequence and leave a blank space between the port number and the port number. The sequence must be one-to-one mapping to the IP address of --prefiller-hosts.<br>
`--decoder-hosts`: Set this parameter to the IP addresses of all d nodes. In the xpyd scenario, add the IP addresses to the end of this configuration item and leave a blank space between the IP addresses.<br>
`--decoder-ports`: Set this parameter to the port number of all d nodes, which is the configuration of the port number for the vllm to start the service in step 4. Set port to the end of the configuration, and leave a blank space between port and port. The sequence must be one-to-one mapping to the IP address of --decoder-hosts.<br>


### 4. Run Inference

Set the IP address in the inference file to the actual IP address. Set the model variable to the path of the model. Ensure that the path is the same as that in the shell script.

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
"model": "model_path",
"prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?",
"max_tokens": 256
}'
```