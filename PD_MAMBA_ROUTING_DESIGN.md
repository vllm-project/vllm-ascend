# PD分离下 Mamba 状态独立通道路由 — 架构设计

## 问题
PD分离(Prefill/Decode分离)下, D节点前缀缓存命中率归零。
根因: `HybridKVCacheCoordinator` 把所有注意力的块大小强行LCM对齐,
Mamba状态在传输中被当作普通KV块处理,但Mamba状态结构不兼容块对齐。

## 架构总览

```
P节点(Prefill)
  │
  ├── KV Cache (块结构,走块通道) ──→ D节点 decode
  └── Mamba State (连续状态,走独立通道) ──→ D节点 decode
                                              │
                                     [控制层 Gate]
                                      ├── 校验两通道都READY → 放行
                                      ├── 各自对齐自己的块/状态
                                      └── 降级: 任意通道失败→全量重算
```

## 三层改动

### Layer 1: 传输层 — 独立 Payload 通道
**文件**: `vllm_ascend/distributed/kv_transfer/.../kv_transfer.py`

**当前**: `ReqMeta` 只传 KV block 的元数据(block_ids, block_hashes)。
**改动**: 在 `ReqMeta` 加 `mamba_state_payload: dict | None`, D节点接收时:
- `mamba_state_payload` 非空 → 标记为混合通道请求
- `mamba_state_payload` 为空 → 纯KV,走原有路径

**关键逻辑**:
```python
# P节点发送
payload = {
    "kv_blocks": block_data,       # 原有KV块
    "mamba_state": mamba_data,     # NEW: Mamba状态独立通道
    "state_ready": False           # 两通道都到齐才置True
}

# D节点接收
# 原子就绪屏障: KV blocks 和 mamba_state 都到达后
# 才通知调度器 TRANSFER_COMPLETE
```

### Layer 2: Coordinator — 跳过Mamba对齐
**文件**: `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`

**当前**: `find_longest_cache_hit()` 对所有组用同一个 `alignment_tokens`。
**改动**: 对齐循环中, `MambaSpec` 组跳过块对齐,改用独立校验。

**关键逻辑**:
```python
for idx, (spec, group_ids, _) in enumerate(self.attention_groups):
    if isinstance(spec, MambaSpec):
        # 跳过块对齐 — Mamba状态已在独立通道对齐
        # 只校验长度一致性
        if not self._validate_mamba_state_length(spec, hit_length):
            curr_hit_length = 0  # 失败→全量重算
        continue  # 不参与LCM
    # 原有的块对齐逻辑不变
```

### Layer 3: 降级开关 — Feature Flag
**文件**: `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`

**新增**:
```python
# 环境变量控制
VLLM_ASCEND_PD_MAMBA_ROUTING = os.environ.get(
    "VLLM_ASCEND_PD_MAMBA_ROUTING", "0"
) == "1"

# 在 coordinator 中
if not VLLM_ASCEND_PD_MAMBA_ROUTING:
    # 走原有路径(纯块对齐, Mamba参与LCM)
else:
    # 走新路径(Mamba独立通道)
```

## 测试方案 (基于Mock, 无需Ascend硬件)

```python
def test_pd_mamba_state_routing():
    # Mock P节点产出
    p_kv_blocks = [MagicMock(block_id=i) for i in range(10)]
    p_mamba_state = torch.randn(4, 512)  # 模拟Mamba连续状态

    # Mock D节点接收
    req = ReqMeta(
        block_ids=[b.block_id for b in p_kv_blocks],
        mamba_state_payload={"layer_0": p_mamba_state}
    )

    # Mock Coordinator
    coord = AscendHybridKVCacheCoordinator(...)
    # 模拟混合attention: full_attn(128) + mamba(无块大小)

    # 验证
    assert coord.find_longest_cache_hit(...) > 0  # 不应归零
    assert req.mamba_state_payload is not None     # 独立通道有效

def test_fallback_on_mamba_failure():
    # Mamba通道失败→回退全量重算
    req = ReqMeta(
        block_ids=[...],
        mamba_state_payload=None  # 模拟传输失败
    )
    coord = AscendHybridKVCacheCoordinator(...)
    # 验证: 不崩溃, 走原有纯KV路径
```

## 实施顺序
1. 先写测试 (基于Mock)
2. 实现 Layer 1 (传输层 payload)
3. 实现 Layer 2 (Coordinator 跳过 Mamba 对齐)
4. 实现 Layer 3 (Feature flag)
5. 提交 PR, 关联 #9247

## 与 #9247 的关系
#9247 修了单节点LCM对齐问题。
本方案是PD分离场景下的扩展——在已有progressive alignment基础上,
增加Mamba状态独立通道。
