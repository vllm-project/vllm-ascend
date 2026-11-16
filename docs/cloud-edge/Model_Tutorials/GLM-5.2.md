# GLM-5.2

## 1 简介

GLM-5.2 采用混合专家（MoE）架构，面向复杂系统工程和长周期智能体任务。

本文档将演示模型的边云协同特性的部署方式。

## 2 前提条件

### 2.1 模型权重

GLM-5.2-w4a8c8（实验性）：需要 1 个 Atlas 800 A3（128GB × 8）节点，[下载模型权重](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w4a8c8)。

### 2.2 节点通信

部署边云特性环境，需要按照[安装指南](../install.md)中的***节点通信验证***章节验证多节点通信。


## 3 环境安装

我们提供了[编译后版本的镜像](https://quay.io/repository/kyoshiki/vllm-ascend-v0.33.0),使用该镜像进行环境准备。

### 3.1 容器创建
参考以下命令创建容器。
```commandline
export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0-a3 #可修改
docker run -itd --privileged \
--name=<container-name> \
--ipc=host --net=host\
--shm-size 500g \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device /dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/sbin/npu‑smi:/usr/local/sbin/npu‑smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /home/weight:/home/weight \
-it $IMAGE bash
```

表 3.1 参数说明

| 参数                                                 | 参数说明                                                                                                                                                                                                                                                                                                                               |
|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --name                                               | 设置容器名称。                                                                                                                                                                                                                                                                                                                         |
| --device                                             | 表示映射的设备，可以挂载一个或者多个设备。需要挂载的设备如下：/dev/davinciX：NPU 设备，X 是 ID 号，如：davinci0。/dev/davinci_manager：davinci 相关的管理设备。/dev/hisi_hdc：hdc 相关管理设备。/dev/devmm_svm：内存管理相关设备。可根据命令`ll /dev/`查询 device 个数及名称方式，根据需要绑定设备，修改上面命令中的 "--device=****"。 |
| -v /usr/local/Ascend/driver:/usr/local/Ascend/driver | 将宿主机目录 “/usr/local/Ascend/driver” 挂载到容器，请根据驱动所在实际路径修改。                                                                                                                                                                                                                                                       |
| -v /path‑to‑weights:/path‑to‑weights                 | 设定权重挂载的路径，需要根据用户的情况修改。请将权重文件和数据集文件同时放置于该路径下。                                                                                                                                                                                                                                               |

### 3.2 执行以下命令进入容器。

```commandline
docker exec -it <container-name> bash
```

### 3.3 边云全局共享的静态角色注册表创建

创建yaml文件(边云服务器一致)
```commandline
deployment:
  mode: 0   #标识使用边云协同
  scene: single_instance    #标识部署单实例
  hccl_world_size: 17        #设置为边云一共使用多少张npu卡
  edges_num: 1              #标识使用一个边侧服务
  clouds_num: 1             #标识使用一个云侧服务
feature_ctrl:
  scramble:                 #标识加扰功能   
    enable: false
    device_id: null
    custom_factor: null
edges:                      #标识边侧配置 
- id: 0                     #标识索引为0的边侧服务
  dp:                       #每个dp下的配置
  - dp_idx: 0               #dp 0 的配置
    addr: <边侧业务面IP>    
    ranks:
    - 0                     #hccl_world_size从0开始计数
  - dp_idx: 1               #dp 1 的配置
    addr: <边侧业务面IP>    
    ranks:
    - 0                     #hccl_world_size从0开始计数，和dp0使用同一张npu卡
clouds:                     #标识云侧配置                 
- id: 0
  dp:
  - dp_idx: 0
    addr: <云侧业务面IP>
    ranks:
    - 1
    - 2
    - 3
    - 4
    - 5 
    - 6 
    - 7
    - 8
    ctrl_port: <port1>       #标识该dp使用的端口号
  - dp_idx: 1
    addr: <云侧业务面IP>
    ranks:
    - 9
    - 10
    - 11
    - 12
    - 13
    - 14 
    - 15
    - 16
    ctrl_port: <port2>       #标识该dp使用的端口号
instance_links:
- edge: 0
  cloud: 0
```

## 4 服务部署

(1) 边侧服务器
```bash
#!/bin/sh
export VLLM_HOST_IP=<边侧业务面IP>
export GLOO_SOCKET_IFNAME=<边侧业务面网卡>
export TP_SOCKET_IFNAME=<边侧业务面网卡>
export HCCL_SOCKET_IFNAME=<边侧业务面网卡>
export MASTER_ADDR=<边侧业务面IP>

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LWD_LOG_LEVEL=INFO
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export ASCEND_RT_VISIBLE_DEVICES=3

vllm serve <model_path> \
    --host 0.0.0.0 \
    --port <port> \
    --master-addr <边侧业务面IP> \
    --master-port <master-port> \
    --safetensors-load-strategy prefetch \
    --api-server-count 1 \
    --enable-expert-parallel \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name glm5 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-num-seqs 40 \
    --max-model-len 133100 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "NONE"}' \
    --additional-config '{"enable_dsa_cp": true,"enable_sparse_li_c8": true,"multistream_overlap_shared_expert":true,"enable_fused_mc2":0, "lwd_config":{"path": "<边云全局共享的静态角色注册表.yaml>", "role": "edge", "instance_id": 0}}' \
    --no-enable-prefix-caching \
    --async-scheduling
```

(2) 云侧服务器
```bash
#!/bin/sh
export VLLM_HOST_IP=<云侧业务面IP>
export GLOO_SOCKET_IFNAME=<云侧业务面网卡名>
export TP_SOCKET_IFNAME=<云侧业务面网卡名>
export HCCL_SOCKET_IFNAME=<云侧业务面网卡名>
export MASTER_ADDR=<边侧业务面IP>

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LWD_LOG_LEVEL=INFO
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve <model_path> \
    --host 0.0.0.0 \
    --port <port> \
    --master-addr <边侧业务面IP> \
    --master-port <master-port> \
    --safetensors-load-strategy prefetch \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --served-model-name glm5 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-num-seqs 40 \
    --max-model-len 133100 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_dsa_cp": true,"enable_sparse_li_c8": true,"multistream_overlap_shared_expert":true,"enable_fused_mc2":0, "lwd_config":{"path": "<边云全局共享的静态角色注册表.yaml>", "role": "cloud", "instance_id": 0}}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}' \
    --no-enable-prefix-caching \
    --async-scheduling
```
根据实际情况修改启动命令中的配置参数，参数说明如下

表4.1 配置参数说明

| 配置项                    | 配置说明                                                                                |
|---------------------------|-----------------------------------------------------------------------------------------|
| ASCEND_RT_VISIBLE_DEVICES | 表示启用哪几张卡。对于每个模型实例分配的 npuIds，使用芯片逻辑 ID 表示。                 |
| --served‑model‑name       | 模型名称。                                                                              |
| --host                    | 模型服务 IP，0.0.0.0 表示不指定 IP 地址，该服务器的 IP 皆可访问。                       |
| --port                    | 模型服务端口号。                                                                        |
| --max‑model‑len           | 模型服务上下文长度。                                                                    |
| --pipeline-parallel-size  | 边云推理特性不使用pp并行策略，默认配置为1                                               |
| --tensor-parallel-size    | tp配置为当前服务在本机所用npu卡数                                                       |
| lwd_config                | 边云特性配置；path:边云全局共享的静态角色注册表文件路径；role:边云角色；instance:实例id |
| max-num-seqs              | 最大并发数                                                                              |
| --speculative-config      | 该配置为官方推荐mtp配置，不建议修改                                                     |
| cudagraph_capture_sizes   | 配置为<num_speculative_tokens+1>的整数倍即可，根据并发情况设置为合适值                  |


## 5 功能验证
服务启动后，可通过发送提示词调用模型。支持两个API接口：completions和chat.completions。使用您配置的`--served-model-name`。

Completions API：
```commandline
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<自定义模型名称>",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

Chat Completions API：
```commandline
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<自定义模型名称>",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_completion_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95
    }'
```
预期结果：服务返回HTTP 200 OK。JSON响应包含choices字段及生成的文本。completions API的示例输出（内容已截断以保持简洁）：
```commandline
{
    "id": "cmpl-xxxxxxxxxxxxx",
    "object": "text_completion",
    "created": 1780971952,
    "model": "<自定义模型名称>",
    "choices": [
        {
            "index": 0,
            "text": "The future of AI is a rapidly evolving landscape with breakthroughs in natural language understanding, multimodal reasoning, and autonomous agents. As models grow more capable and efficient...",
            "logprobs": null,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 54,
        "completion_tokens": 50
    }
}
```