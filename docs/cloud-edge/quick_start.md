# 快速入门

本文档以 Atlas 800I A2 推理服务器和 Qwen3.6‑27B 模型为例，让开发者快速开始使用 VLLM 进行大模型推理流程。更多模型部署请参考[模型支持列表](./model_support_lists.md)最后的文档列

## 1 环境准备


(1) Atlas服务器安装

操作系统安装参考：[Atlas服务器openEuler 22.03 LTS SP4 操作系统安装](https://support.huawei.com/enterprise/zh/doc/EDOC1100469526/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

NPU驱动和固件参考：[Atlas A2 中心推理和训练硬件 26.0.RCx NPU驱动和固件](https://support.huawei.com/enterprise/zh/doc/EDOC1100568431/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

(2) 一体机服务器安装

操作系统安装参考：[Atlas服务器openEuler 24.03 LTS SPX 操作系统安装](https://support.huawei.com/enterprise/zh/doc/EDOC1100492838/426cffd9?idPath=23710424|269761314|269769397|269769917|261457531)

NPU驱动和固件安装参考：[Atlas A2 中心推理和训练硬件 26.0.RCx NPU驱动和固件](https://support.huawei.com/enterprise/zh/doc/EDOC1100568431/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

SP681网卡驱动安装参考：[SP220&SP600 标准网卡 用户指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100573093/426cffd9),
[驱动下载地址](https://support.huawei.com/enterprise/zh/computing-module/in220-pid-253287505/software/268628573?idAbsPath=fixnode01|23710424|269761301|269764423|269766398|253287505)

(3) 获取模型权重

下载权重，这里以 [Qwen3.6‑27B](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B) 为例，将权重文件上传至服务器任意目录（如 /home/weight）

## 2 环境部署
我们推荐使用[预构建镜像](https://quay.io/repository/kyoshiki/vllm-ascend-v0.30.0)进行环境部署，该镜像在[vLLM-ascend镜像](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest)基础上编译边云协同功能后获得。参考以下命令创建容器完成环境部署。

```commandline
export IMAGE=quay.io/repository/kyoshiki/vllm-ascend-v0.30.0 #视情况修改
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

表 2.1 参数说明

| 参数                                                 | 参数说明                                                                                                                                                                                                                                                                                                                               |
|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| IMAGE                                                | 镜像的REPOSITORY:TAG。                                                                                                                                                                                                                                                                                                                 |
| --name                                               | 设置容器名称。                                                                                                                                                                                                                                                                                                                         |
| --device                                             | 表示映射的设备，可以挂载一个或者多个设备。需要挂载的设备如下：/dev/davinciX：NPU 设备，X 是 ID 号，如：davinci0。/dev/davinci_manager：davinci 相关的管理设备。/dev/hisi_hdc：hdc 相关管理设备。/dev/devmm_svm：内存管理相关设备。可根据命令`ll /dev/`查询 device 个数及名称方式，根据需要绑定设备，修改上面命令中的 "--device=****"。 |
| -v /usr/local/Ascend/driver:/usr/local/Ascend/driver | 将宿主机目录 “/usr/local/Ascend/driver” 挂载到容器，请根据驱动所在实际路径修改。                                                                                                                                                                                                                                                       |
| -v /path‑to‑weights:/path‑to‑weights                 | 设定权重挂载的路径，需要根据用户的情况修改。请将权重文件和数据集文件同时放置于该路径下。                                                                                                                                                                                                                                               |


## 3 边云协同分布式推理服务部署

(1) 前提条件
- 服务器安装完成

- 容器部署完成

- 网络部署完成，需要分别配置边云服务器业务面和参数面网卡互通

(2) 边云全局共享的静态角色注册表创建

创建yaml文件(边云服务器一致)
```commandline
deployment:
  mode: 0   #标识使用边云协同
  scene: single_instance    #标识部署单实例
  hccl_world_size: 5        #设置为边云一共使用多少张npu卡
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
    ctrl_port: <port>       #标识该dp使用的端口号
instance_links:
- edge: 0
  cloud: 0
```
(3) 边云协同服务部署

进入容器后，执行以下命令启动服务。

a. 边侧服务器
```bash
#!/bin/sh
# ========== 环境变量 ==========
export VLLM_HOST_IP=<边侧业务面IP>
export GLOO_SOCKET_IFNAME=<边侧业务面网卡>
export TP_SOCKET_IFNAME=<边侧业务面网卡>
export HCCL_SOCKET_IFNAME=<边侧业务面网卡>
export MASTER_ADDR=<边侧业务面IP>

export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_GDN_FAST_PATH=1
export VLLM_ASCEND_GDN_MAX_PADDING_RATIO=2.0
export VLLM_ASCEND_GDN_MAX_H_OVERALLOC_RATIO=2.0
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LWD_LOG_LEVEL=INFO
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export ASCEND_RT_VISIBLE_DEVICES=1

# ========== 启动命令 ==========
vllm serve <model_path> \
    --host 0.0.0.0 \
    --port <port> \
    --master-addr <边侧业务面IP> \
    --master-port <master-port> \
    --max-model-len 262144 \
    --served-model-name <自定义模型名称> \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --additional-config '{"enable_cpu_binding":true, "lwd_config":{"path": "<边云全局共享的静态角色注册表.yaml>", "role": "edge", "instance_id": 0}}' \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --max-num-seqs 150 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --mm_processor_cache_type="shm" \
    --mm-processor-cache-gb 0
```
b. 云侧服务器
```bash
#!/bin/sh
# ========== 环境变量 ==========
export VLLM_HOST_IP=<云侧业务面IP>
export GLOO_SOCKET_IFNAME=<云侧业务面网卡>
export TP_SOCKET_IFNAME=<云侧业务面网卡>
export HCCL_SOCKET_IFNAME=<云侧业务面网卡>
export MASTER_ADDR=<云侧业务面IP>

export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_GDN_FAST_PATH=1
export VLLM_ASCEND_GDN_MAX_PADDING_RATIO=2.0
export VLLM_ASCEND_GDN_MAX_H_OVERALLOC_RATIO=2.0
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LWD_LOG_LEVEL=INFO
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ========== 启动命令 ==========
vllm serve model_path \
    --host 0.0.0.0 \
    --port <port> \
    --master-addr <边侧业务面IP> \
    --master-port <master-port> \
    --max-model-len 262144 \
    --served-model-name <自定义模型名称> \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --additional-config '{"enable_cpu_binding":true, "lwd_config":{"path": "<边云全局共享的静态角色注册表.yaml>", "role": "cloud", "instance_id": 0}}' \
    --max-num-seqs 150 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 520, 528, 536, 544, 552, 560, 568, 576, 584, 592, 600, 608, 616, 624, 632, 640, 648, 656, 664, 672, 680, 688, 696, 704, 712, 720, 728, 736, 744, 752, 760, 768, 776, 784, 792, 800]}' \
    --mm_processor_cache_type="shm" \
    --mm-processor-cache-gb 0
```
根据实际情况修改启动命令中的配置参数，参数说明如下

表3.1.1 配置参数说明

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

## 4 功能验证
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