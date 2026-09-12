# 需求描述

通过接入 flash-attention-npu 中的 flash_attn_with_kvcache/flash_attn_varlen_func 算子（即 fa3 算子），对 vllm-ascend 的 attention_v1.py 进行重构，以支持 GQA 的 tiling 下沉从而提升性能，并且要支持图模式。目标是分析当前 fa3 算子支持的硬件类型和功能特性，识别所有可以被 fa3 算子平替的部分，全量进行重构。

# 代码构建安装方法

## flash-attn-npu 算子源码构建安装方法

1. cd /mnt/share/z00586359/code/fia_opt/flash-attention-npu/
2. source /usr/local/Ascend/cann/set_env.sh
3. python setup.py install

## vllm 源码构建安装方法

1. cd /mnt/share/z00586359/code/fia_opt/vllm/
2. pip install setuptools_rust
3. VLLM_TARGET_DEVICE=empty pip install -v -e . --no-build-isolation --no-index --no-deps

## vllm-ascend 源码构建安装方法

1. cd /mnt/share/z00586359/code/fia_opt/vllm-ascend/
2. pip install -v -e .   --no-build-isolation   -i https://mirrors.aliyun.com/pypi/simple/   --trusted-host mirrors.aliyun.com   --no-index --no-deps

# 代码验证方式

## 注意事项

- 代码验证依赖 npu 服务器，验证的前提条件是可以通过 ssh 远程连接 npu 服务器。连接 npu 服务器，检查 npu 卡的状态等操作可通过本项目的 SKILL.md 来帮助完成

- 服务端容器 zty_fia_opt 中已通过源码安装 vllm/vllm-ascend/flash-attn-npu，源码目录在 /mnt/share/z00586359/code/fia_opt；正常情况下不需要修改 vllm/flash-attn-npu，只需要修改 vllm-ascend 来实现我们的需求，因为我们的修改不涉及算子等 C++ 代码，在仅修改 vllm-ascend/vllm_ascend 目录下的 python 代码的情况下，改完代码后重新启动服务即可验证修改后的效果

- 验证过程中出现的常见异常和建议策略如下：
1. npu 卡被其他容器进程占用：等待1分钟后重新检查
2. 服务启动失败/请求推理报错/测评精度异常：定位修改代码的问题，解决后重新验证

## 验证流程

1. [ssh1] ssh 远程连接 npu 服务器（后续通过 [ssh1] 来标注是这个连接上进行的操作，[ssh1] 主要用来启动服务/停止服务）
2. [ssh1] 检查 npu 卡的状态是否空闲，确保空闲才可以正常启动服务，如果 npu 卡被其他容器进程占用，建议等待1分钟后重新检查
3. [ssh1] 进入服务端容器：docker exec -it zty_fia_opt bash
4. [ssh1] 进入工作目录：cd /mnt/share/z00586359/run/qwen/qwen3_235b
5. [ssh1] 执行服务启动脚本：bash start_vllm_server.sh
6. [ssh1] 等待服务启动成功，因为当前设置的 DP=4，服务启动成功的标志是4个 DP 组的进程都打印出 "Application startup complete." 字样
7. [ssh2] ssh 远程连接 npu 服务器（后续通过 [ssh2] 来标注是这个连接上进行的操作，[ssh2] 主要用来启动测评工具/查看测评结果）
8. [ssh2] 进入测评容器：docker exec -it zty_aisbench bash
9. [ssh2] 进入工作目录：cd /mnt/share/z00586359/run/qwen/qwen3_235b
10. [ssh2] 执行测评启动脚本：bash run_gpqa.sh
11. [ssh2] 等待测评结束，标志是打印出 "[ais_bench] [INFO] write markdown summary to /mnt/share/z00586359/run/qwen/qwen3_235b/outputs/default/xxxxxxxx_xxxxxx/summary/summary_xxxxxxxx_xxxxxx.md"
12. [ssh2] 确认测评结果，对于 GPQA 数据集，精度基线为 72.12 分，最大允许波动 3%，即最低需要达到 69.12 分以上认为是达标，否则就是精度异常
13. [ssh1] 测评结束后停止服务，在服务端容器 zty_fia_opt 内进行操作：pkill -9 -f VLLM
