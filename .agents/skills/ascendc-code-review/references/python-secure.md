# CANN Python 安全编码规范

<适用>
语言: Python
侧别: N/A
领域: false
默认启用: true
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

> **适用场景**：算子仓 Python 代码，分三类：
> - **测试代码**（tests/ 下的 gen_data、executor、断言比对）——造数据、调 API、验证结果
> - **工具/构建脚本**（scripts/、cmake/、tools/）——编译、覆盖率、路径处理
> - **PyTorch 接口注册代码**（torch_extension/ 等）——被用户 import 调用，处理用户传入的 tensor/参数，属产品级 Python
>
> **说明**：本规范只保留算子仓 Python 实际会触达的安全条例。网络安全、密码学、序列化、XML、临时文件等通用 Python 服务端主题在算子仓无触达场景，已移除，避免全量检视产生误报。

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| 1.1 | 文件头注释包含 COPYRIGHT 和 LICENSE 声明 | 文件头 | 中 |
| 2.1 | 除法和模运算除零保护 | 数值安全 | 高 |
| 3.1 | 禁止通过异常泄露敏感数据 | 异常处理 | 高 |
| 5.2 | 外部文件路径必须校验和规范化 | 文件操作 | 高 |
| 7.4 | 禁止使用 subprocess 的 shell=True | 命令执行 | 高 |

## 说明

本规范是 CANN 开源社区的 Python 安全编码规范，仅覆盖算子仓 Python 代码实际触达的数值运算安全、异常处理、文件路径安全、命令执行安全四类。

## 适用范围

CANN 相关开源仓的 Python 代码安全检视。

---

### 1. 文件头注释

##### 规则 1.1 文件头注释应该包含COPYRIGHT和LICENSE声明

**【适用场景】** 所有 Python 文件（测试、工具、torch_extension 全场景）。

**【描述】** 在每个源文件的开头添加COPYRIGHT和LICENSE声明，确保代码的合法性和可追溯性。这有助于保护代码的知识产权，并明确使用条款。版权年份取文件创建当年的年份。

**【检视方法】** 执行 `date +%Y` 获取当前系统年份，用于判断新文件是否遗漏版权头、或年份与当前年份是否明显异常（如未来年份、过早年份）。

举例：

```python
#
# Copyright (c) {当前年份} Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
```

---

### 2. 数值运算安全

##### 规则 2.1 对除法运算和模运算中的除数为0的情况做相应保护

**【适用场景】** 处理用户传入 tensor 或参数并参与除法/取模运算的 Python 代码（典型为 torch_extension 的 op 注册代码中 shape 推导、参数校验）。

**【需要注意的场景】**

- **tensor shape 维度相除**：`q.shape[x] / ori_kv.shape[y]`——除数来自用户传入 tensor 的维度，用户传特定 shape（某维为 0）即除零
- **用户参数取模**：`a % b` 其中 `b` 来自函数参数，调用方可能传 0
- **常量除法无需校验**：`h / 128` 这类除数为字面量且非零的，不触发本规则

**【描述】** 在进行除法运算或取模运算时，应先检查除数是否为0，以防止程序因除零错误而崩溃。可以通过条件语句或异常处理来实现。

**【示例】**

风险点（除数来自用户 tensor 维度，无校验）：

```python
# torch_extension/cann_ops_transformer/ops/sparse_flash_mla.py
# q.shape[2] / ori_kv.shape[2] —— ori_kv.shape[2] 来自用户输入，可能为 0
softmax_lse = torch.empty([q.shape[0], ori_kv.shape[2], q.shape[1], q.shape[2] / ori_kv.shape[2]],
                          dtype=torch.float32, device="meta")
```

已合规（取模前有前置非零校验）：

```python
# torch_extension/cann_ops_transformer/ops/moe_distribute_dispatch.py
# shared_expert_num > 0 的前置校验保证了取模数非零
is_valid_shared = (
    (shared_expert_num > 0)
    and ((shared_expert_rank_num // shared_expert_num) > 0)
    and ((shared_expert_rank_num % shared_expert_num) == 0)
)
```

---

### 3. 异常处理

##### 规则 3.1 禁止通过异常泄露敏感数据

**【适用场景】** raise message 中包含用户输入值或内部结构信息的代码（典型为 torch_extension 的参数校验 raise）。

**【描述】** 在捕获和处理异常时，不应将敏感数据（如密码、密钥等）包含在异常消息中，以防止这些信息被意外泄露。

**【示例】**

```python
# torch_extension 中参数校验的 raise —— 把维度值写进 message
# scales.dim() 非敏感，可接受；但若 raise message 引用用户传入的 token/密钥等则违规
raise RuntimeError(f"Expected scales to be at least 2-d, but got {scales.dim()}-d.")
```

> 算子仓 Python 无密码/密钥等敏感数据，本规则价值较低，主要提醒：raise message 不要引用可能含敏感内容的用户输入。

---

### 5. 文件操作安全

##### 规则 5.2 使用外部数据构造的文件路径前必须进行校验，校验前必须对文件路径进行规范化处理

**【适用场景】** 读取环境变量或外部输入拼路径的代码（典型为 builder、工具脚本读取 `ASCEND_HOME_PATH` 等环境变量）。

**【描述】** 在使用外部数据生成文件路径之前，应对路径进行规范化处理（如去除多余的路径分隔符），并进行校验，以防止路径遍历攻击。

**【示例】**

```python
# torch_extension/cann_ops_transformer/op_builder/builder.py
# ASCEND_HOME_PATH 来自环境变量（外部输入），使用前校验存在性
if ASCEND_HOME_PATH in os.environ and os.path.exists(os.environ[ASCEND_HOME_PATH]):
    return os.environ[ASCEND_HOME_PATH]
return None
```

---

### 7. 命令执行安全

##### 规则 7.4 禁止使用subprocess模块中的shell=True选项

**【适用场景】** 使用 subprocess 调用系统命令的构建/工具脚本（编译、运行测试等）。

**【描述】** `subprocess` 模块的 `shell=True` 选项会通过系统 shell 执行命令，存在注入攻击的风险。应使用 `shell=False` 并传递命令列表。

**【示例】**

```python
# 已合规 —— 列表传参，shell 默认 False
subprocess.run([cmake_path, "--version"])
subprocess.run(compile_cmd, capture_output=True, text=True)  # compile_cmd 为列表
```
