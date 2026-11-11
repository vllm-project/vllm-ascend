# PR1 提交操作指南

## 前置条件

- GitHub 账号已 fork vllm-project/vllm-ascend
- 本机已 clone fork 到本地工作目录
- DCO（Developer Certificate of Origin）已签署（commit 加 `-s` 即可）

## 第一步：提交 Issue（先于 PR）

```bash
# 打开 https://github.com/vllm-project/vllm-ascend/issues/new
# 选择 "Bug report" 模板
# 粘贴 01_issue_report.md 的内容
# 标题：Native OffloadingConnector crashes on hybrid attention models (GDN + full attention) due to non-coalescible interleaved layout
# 提交后记下 Issue 编号（如 #1234）
```

## 第二步：创建分支 + 应用代码改动

```bash
cd <你的 vllm-ascend 本地仓库>
git checkout main && git pull origin main
git checkout -b fix/skip-mamba-groups-native-offload

# 应用代码改动
# 手动编辑 vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py
# 按 03_code_change.patch 中的 diff 修改（两处：import + 两个循环）
# 或者直接 git apply：
git apply /home/z00833621/kv-offload-eval/pr1_hybrid_support/03_code_change.patch

# 添加测试文件
cp /home/z00833621/kv-offload-eval/pr1_hybrid_support/test_skip_mamba_groups.py tests/
```

## 第三步：本地验证

```bash
# 跑全量单测（不含新测试）
python3 -m pytest tests/ -q --ignore=tests/test_skip_mamba_groups.py

# 跑新增单测
python3 -m pytest tests/test_skip_mamba_groups.py -q

# 硬件验证（如果 4-7 卡空闲）
# 1. 重建容器（镜像换成你 fork 的 dev 版本，或直接 docker exec 改文件）
# 2. 启动混合模型 → 确认不崩
# 3. 启动纯 dense 模型 → 确认无回归
```

## 第四步：提交 + 推送

```bash
git add vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py tests/test_skip_mamba_groups.py
git commit -s -m "[Bugfix] Skip MambaSpec groups in native offloading to support hybrid attention models

Hybrid attention models (e.g. Qwen3.5 with GDN linear attention layers)
crash on Ascend NPUs when using the native OffloadingConnector, because
the Ascend worker's register_kv_caches attempts to build int8 block views
for ALL KV cache groups, including MambaSpec groups whose state tensors
have a non-coalescible interleaved layout (stride != payload due to
hybrid pool packing).

This PR skips MambaSpec groups in the Ascend worker's canonicalization,
allowing hybrid models to start and offload their full-attention groups
normally. Mamba states remain in NPU, managed by vLLM's
HybridKVCacheCoordinator. The upstream _bounding_group_ids already
bounds load depth for non-offloaded groups.

Fixes #<issue_number>

Signed-off-by: <你的姓名> <你的邮箱>"
git push origin fix/skip-mamba-groups-native-offload
```

## 第五步：创建 PR

```bash
# 打开 https://github.com/vllm-project/vllm-ascend/compare/main...<你的用户名>:fix/skip-mamba-groups-native-offload
# 粘贴 02_pr_description.md 的内容作为 PR description
# 关联 Issue：在 description 中已含 "Fixes #<issue_number>"
# 提交 PR
```

## 第六步：跟进 review

- 关注 CI 状态（vllm-ascend CI 会在多种硬件上跑测试）
- reviewer 可能要求拆分或合并改动——PR1 只改 offloading_connector.py + 新增测试，改动面小，review 应该快
- 如果 reviewer 要求加更多测试或改设计，参考 README.md 中"评审要点"准备回应

## 注意事项

- **不要**在同一个 PR 里混入 E 组（AscendStore）或 F 组（LMCache）的改动
- **不要**改动 vllm 上游代码（vllm/vllm/ 目录）——那是另一个仓库的事
- commit message 格式遵循 vllm-ascend 现有风格（看 git log --oneline 找规律）
- 如果 vllm-ascend 要求 PR 基于特定版本分支（而非 main），按 maintainer 指示调整
