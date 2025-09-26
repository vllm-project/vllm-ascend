# run_model_qwen.sh脚本介绍

## 功能概述

本脚本提供两种部署模式的一键启动：

- 1. 混部模式(Default)
- 2. PD分离模式(PD_Separate)：自动拉起创建ranktable,单机的1P1D服务与 Global Proxy 服务

并且提供两种执行方式：

- 1. 单算子模式
- 2. 入图模式：启用图优化加速,需设置`--graph-true 'true'`

## 文件中参数配置说明如下:

- 必填
  - `model-path` ：指定模型存放的绝对路径目录
  - `deploy-mode` ：部署模式：default(混部) 或 pd_separate(pd分离)
  - `graph-true` ：图优化模式：true(启动)/false(标准模式)

- 可选
  - `server-list`：  混部模式下，使用NPU的ID列表(逗号分隔)，默认'0,1,2,3,4,5,6,7'
  - `prefill-server-list`：  PD分离模式下，prefill使用NPU的ID列表(逗号分隔)，默认'0,1,2,3,4,5,6,7'
  - `decode-server-list`：  PD分离模式下，decode使用NPU的ID列表(逗号分隔)，默认'8,9,10,11,12,13,14,15'

  - `network-interface`： 指定分布式通信网卡，若不设置，自动检测
  - `host-ip` ： 本机IP地址，若不设置，自动检测
  - `service-port` ：  PD分离模式下，Prefill服务端口(Decoder端口=该值+offset)，默认6660
  - `master-port` ：分布式核心协调端口，默认8888
  - `https-port`  ：HTTPS API服务端口，用户请求应发送至该端口，默认8001

  - `model-name`：  模型标识，默认default_model
  - `max-model-len`：  模型最大上下文长度，模型支持的token数量上限，默认20960
  - `log-path` ：服务日志存储目录，默认./apiserverlog

  - `additional-config` : 用于传递JSON格式的高级运行时配置，注意会抢占原有的默认配置

## 注意事项

1. 脚本不支持多机PD，多机PD参考ansible脚本
2. `server-list`、`max-model-len`依据模型实际修改
3. 入图参数若有修改，直接修改`additional-config`