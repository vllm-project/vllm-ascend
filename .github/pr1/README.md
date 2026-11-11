# PR1 提交材料总览

**目标仓库**：https://github.com/vllm-project/vllm-ascend
**PR 标题**：`[Bugfix] Skip MambaSpec groups in native offloading to support hybrid attention models`
**关联 Issue**：先提交 Issue（内容见 01_issue_report.md），PR 中引用 `Fixes #<issue_number>`
**基线**：vllm-ascend main（2026-11-05 前后），基于 v0.30.0rc2 验证

## 文件清单

| 文件 | 用途 |
|---|---|
| `01_issue_report.md` | GitHub Issue 正文（英文，先于 PR 提交） |
| `02_pr_description.md` | PR description 正文（英文，PR 提交时粘贴） |
| `03_code_change.patch` | 代码改动（git diff 格式，可直接 `git apply`） |
| `test_skip_mamba_groups.py` | 单元测试（pytest，验证跳过逻辑 + 对齐 + 回归） |
| `05_submission_guide.md` | 逐步操作指南（fork → branch → commit → push → PR） |
| `README.md` | 本文件（材料总览） |

## 提交流程

```
1. Fork vllm-project/vllm-ascend → 你的 GitHub 账号
2. git clone https://github.com/<你的用户名>/vllm-ascend.git
3. cd vllm-ascend && git checkout -b fix/skip-mamba-groups-native-offload
4. 把 03_code_change.patch 的改动应用到
   vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py
5. 把 test_skip_mamba_groups.py 复制到 tests/
6. git add -A && git commit -s -m "[Bugfix] Skip MambaSpec groups in native offloading to support hybrid attention models"
7. git push origin fix/skip-mamba-groups-native-offload
8. 在 GitHub 上从 fork 创建 PR → 粘贴 02_pr_description.md → 关联 Issue
9. 等 CI 通过 + maintainer review
```

## 验证清单（提交前自查）

- [ ] 纯 dense 模型（Qwen3-32B）启动 + 推理正常（回归测试）
- [ ] 混合模型（Qwen3.5-27B）启动不再崩溃（Bug 修复验证）
- [ ] 混合模型 P1 填充后 external hits 正常产生（仅全注意力组）
- [ ] Mamba 组日志出现 "Skipping MambaSpec group"（跳过确认）
- [ ] `pytest tests/ -q` 通过（含新增单测）
- [ ] `git diff --stat` 只改了 offloading_connector.py + 新增 test 文件
- [ ] commit 带 `-s`（DCO sign-off）

## 评审要点（预判 reviewer 可能问的问题）

**Q: 为什么不把 Mamba 状态也 offload？**
A: Mamba 状态是每序列一份的 O(1) 递推状态（非 per-token KV），其语义与 offloading 的"块级部分前缀命中"模型不匹配——状态的复用粒度是"整前缀全有全无"，不支持"命中 31000/32000 token 后分叉"。且量级极小（~2MB/会话 vs 全注意力 ~8GB），不存在 offload 的存储压力。此设计已由 MooncakeConnectorV1（GDN tested）和 UCM/AscendStoreConnector（align 模式）验证可行。

**Q: 跳过 Mamba 组后，回访时线性层的性能怎么办？**
A: Mamba 状态由 vLLM 的 HybridKVCacheCoordinator 状态缓存管理（不走 offloading connector），同一会话的多轮续跑仍可通过状态缓存命中（类似 MooncakeConnector 的 h(N-1) 重建协议）。跨重启场景的 Mamba 状态持久化可由后续 PR 实现（参考 UCM 的 lookup_on_reverse 设计）。

**Q: 为什么不修 `_make_int8_block_view` 让它支持交错布局？**
A: 交错布局（stride > payload）意味着两个组的存储共享同一物理池——正确处理需要 per-group 视图 + 独立的准入/驱逐语义，改动范围跨越 model_runner 的内存分配策略。本次 PR 的目标是先解除启动崩溃（最小安全修复），布局支持可作为后续 PR 独立推进。
