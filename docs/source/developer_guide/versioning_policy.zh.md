# 版本策略

从vLLM的0.7.x版本开始，vLLM Ascend Plugin ([vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)) 整体遵循[PEP 440](https://peps.python.org/pep-0440/)的版本策略，与vLLM ([vllm-project/vllm](https://github.com/vllm-project/vllm)) 配套发布。

## vLLM Ascend Plugin版本

vllm-ascend的版本号为：`v[major].[minor].[micro][rcN][.postN]`（比如`v0.7.1rc1`, `v0.7.1`, `v0.7.1.post1`）

- **Final releases （正式版）**: 通常3个月发布一次正式版，将会综合考虑vLLM上游发布及昇腾产品软件发布策略。
- **Pre releases （尝鲜版）**: 通常为按需发布，以rcN结尾，代表第N个Release Candidate版本，提供在final release之前的尝鲜版（早期试用版）。
- **Post releases （补丁版）**: 通常在final release发布后按需发布，主要是修复最终版本的错误。这个策略与[PEP-440提到的策略](https://peps.python.org/pep-0440/#post-releases)有所不同，它会包含实际的bug修复，考虑到正式版与vLLM的版本（`v[major].[minor].[micro]`）配套发布。因此，Post releases通常是Final release的补丁版本。

例如：
- `v0.7.x`: 是配套 vLLM `v0.7.x` 版本的正式版。
- `v0.7.1rc1`: 是vllm-ascend第一个尝鲜版（早期试用版）。
- `v0.7.1.post1`: 是`v0.7.1`版本的post release。

## 分支管理策略

vllm-ascend有主干和开发两种分支。

- **main**: 主干分支，与vLLM的主干分支对应，并通过昇腾CI持续进行质量看护。
- **vX.Y.Z-dev**: 开发分支，随vLLM部分新版本发布而创建，比如`v0.7.1-dev`是vllm-ascend针对vLLM `v0.7.1`版本的开发分支。


通常，一个commit需要先合入到主干分支，然后再反合到开发分支，从而尽可能地减少版本维护成本。


### 分支维护和EOL
某个分支的状态将会以下三种之一：
| 分支            | 维护时间                 | 备注                                                              |
|-------------------|----------------------------|----------------------------------------------------------------------|
| Maintained （维护中）        | 大概2-3个minor版本 | 合入所有已解决的问题，发布版本，CI保证 |
| Unmaintained （无维护）     | 社区诉求/兴趣驱动  | 合入所有已解决的问题，无版本发布，无CI承诺 |
| End of Life (EOL, 生命周期终止) | 无                        | 分支不再接受任何代码                                   |

### 分支状态

注意：对于`*-dev`分支，vllm-ascend将仅针对 vLLM 某个特定版本创建开发分支，而非全量版本。 因此，您可能看到部分vLLM版本没有对应的开发分支（比如只能看到`0.7.1-dev` / `0.7.3-dev`分支，而没有`0.7.2-dev`分支），这是符合预期的。

通常来说，vLLM每个minor版本（比如0.7）均会对应一个vllm-ascend版本分支，并支持其最新的版本（例如我们计划支持0.7.3版本）。如下所示：

| 分支         | 状态           | 备注                  |
|------------|--------------|---------------------|
| main       | Maintained   | 基于vLLM main分支CI看护   |
| v0.7.3-dev | Maintained   | 基于vLLM v0.7.3版本CI看护 |
| v0.7.1-dev | Unmaintained | 被v0.7.3-dev分支代替     |

## 文档分支管理策略

为了减少维护成本，**所有分支文档内容应当保持一致，版本的差异可以通过[docs/source/conf.py](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/conf.py)中的变量控制**，这不是一件简单的事情，但这是我们应该努力遵循的原则。

| 版本 | 定位 | 代码分支 |
|-----|-----|---------|
| latest | 最新开发分支文档 | `vX.Y.Z-dev` （首个版本发布后为`main`） |
| version | 历史发布版本的文档 | `vX.Y.Z[rcN]`等git tag |
| stable（尚未发布） | 最新正式版分支文档 | 首个版本发布后为`vX.Y.Z-dev` |

如上所示：
- `latest`文档：最新发布版维护分支的文档，匹配当前维护分支`vX.Y.Z-dev`的文档（当首个正式版发布后，会更换为`main`分支），持续更新，保证最新发布版的文档持续可用。
- `version`文档：发布版本的文档，对应版本为vX.Y.Z[rcN]（例如`v0.7.3`, `v0.7.3rc1`），版本发布后不会再更新。
- `stable`文档（尚未发布）：正式版文档，**文档允许在release后实时更新**，通常是`vX.Y.Z-dev`。有稳定文档后，在非稳定版的文档顶部，应该提示: `您正在查看最新的开发者预览版文档。单击此处查看最新稳定版本的文档。`。

## 版本配套

vLLM Ascend Plugin (`vllm-ascend`) 的关键配套关系如下:

| vllm-ascend  | vLLM    | Python | Stable CANN | PyTorch/torch_npu |
|--------------|---------| --- | --- | --- |
| v0.7.1rc1 | v0.7.1 | 3.9 - 3.12 | 8.0.0 |  2.5.1 / 2.5.1.dev20250218 |

## 发布节奏

### 下一个正式版(`v0.7.x`)发布窗口

| 时间       | 事件                            |
|----------|-------------------------------|
| 2025年03月 | RC版本, v0.7.3rc1               |
| 2025年03月 | 正式版, 匹配0.7.3最新的vLLM版本: v0.7.3 |
