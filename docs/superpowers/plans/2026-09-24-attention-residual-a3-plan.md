# Attention Residual A3 适配实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 A3 上以批量向量路径实现 attention residual，修复 64 元素边界，同时通过 A3 小算子组合基线验证精度和性能。

**Architecture:** 保留现有 A3 arch22 reload/resident 数据流和 host tiling ABI，仅替换末轴 Sub/Mul 的低效 per-8 helper；融合入口继续由独立 arch22 kernel 注册。A5 arch35 路径不改。

**Tech Stack:** Ascend C、CANN 9.1、A3 Ascend910、PyTorch/torch_npu、pytest、NPU 设备计时。

**Spec:** `docs/superpowers/specs/2026-09-24-attention-residual-a3-design.md`

## Global Constraints

- A3 目标为 `ascend910_93`，A5 目标为 `ascend950`。
- `raw_prefix`、`materialized`、bank 写回和输入保护必须 bitwise 对齐 A3 小算子参考。
- 最终 output RMSNorm 使用 `rtol=1e-2, atol=1e-2`。
- 输出不得有 NaN 或 Inf。
- 性能基线必须是等价 A3 小算子组合，不能使用有精度缺陷的修复前 A3 包。
- 每次 kernel 修改后清理 build 和 kernel cache，再重新安装测试包。

## Review Focus

- `curColNum == 64`：完整块必须写入 output；由 Task 1 的边界精度测试覆盖。
- `curColNum` 不是 64 倍数：完整块和 tail 不能越界；由 Task 1 的 B=62/63/64/65 覆盖。
- UB 32B 对齐和 repeat stride：批量 API 不得触发 507035；由 Task 2 的独立 helper 回归覆盖。
- `valid=0` 和 NaN padding：无效 bank 不能参与计算；由 Task 3 的 with_add/fused 测试覆盖。
- strided/slice/graph replay：不能通过 contiguous 假设掩盖布局问题；由 Task 3/4 的布局和 replay 测试覆盖。

---

### Task 1: 建立 A3 helper 边界回归

**Files:**
- Modify: `csrc/attention/attn_res_fwd/op_kernel/arch22/reduce_common.h`
- Test: remote direct test script under `/home/z00853276/attn_residual_a3_migration/`

**Interfaces:**
- Consumes: existing `SubLastDimRow1NoBrc`, `MulLastDimRow1NoBrc` signatures.
- Produces: batch-plus-tail behavior covering complete 64-element blocks and one tail call.

- [ ] Inspect existing `BinaryRepeatParams`, `ELEM_PER_REP_FP32`, `ELEM_PER_BLK_FP32`, and A3 alignment constraints before editing.
- [ ] Add failing boundary cases for B=62, 63, 64, 65 with H=128 and reference comparison.
- [ ] Replace per-8 loops with complete-block plus tail calls, preserving output offsets and aligned padding.
- [ ] Build a clean A3 package after deleting build/cache directories.
- [ ] Run B=62/63/64/65 in separate processes and require maxerr=0, no NaN/Inf, and no alignment error.

### Task 2: Tune helper cost and verify no device fault

**Files:**
- Modify: `csrc/attention/attn_res_fwd/op_kernel/arch22/reduce_common.h`
- Test: remote performance helper and direct kernel tests.

**Interfaces:**
- Consumes: Task 1 batch-plus-tail helper.
- Produces: A3 helper with minimal barrier count and valid UB access.

- [ ] Count barriers and vector calls for B=1, 3, 64 before and after the change.
- [ ] Run standalone A3 tests for H=128, 4096, and 7168, including B=1, 3, 64.
- [ ] Inspect device logs for `507035`, UB alignment, illegal memory access, and NaN/Inf.
- [ ] Keep the faster implementation only if all precision cases remain bitwise/reference aligned.

### Task 3: Regress with_add and fused semantics

**Files:**
- Test: `tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py`
- Inspect: `csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_reload.h`
- Inspect: `csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_resident.h`

**Interfaces:**
- Consumes: Task 2 A3 helper and existing fused kernels.
- Produces: precision evidence for add, bank write, materialized, output norm, and invalid-bank masking.

- [ ] Run direct reference cases for with_add with valid blocks 0, 1, 4, and 8, both contiguous and strided.
- [ ] Run fused cases with valid 0, 1, 4, 8, 63, and 64, with and without output norm.
- [ ] Check raw prefix, materialized output, bank writeback, input immutability, dtype, shape, and NaN padding.
- [ ] Run the official pytest file after the test environment loads `_build_info` and the modelscope compatibility module.

### Task 4: Validate prefill and graph replay

**Files:**
- Inspect: `csrc/attention/attn_res_fwd_prefill/op_kernel/attn_res_fwd_prefill.cpp`
- Inspect: `csrc/attention/attn_res_fwd/op_host/attn_res_fwd_tiling.cpp`
- Test: official prefill graph-replay cases.

**Interfaces:**
- Consumes: Task 3 fused semantics.
- Produces: prefill cache correctness and replay evidence.

- [ ] Run valid=0, 1, 2, 4, and 8 with token count 129.
- [ ] Replay the same graph after changing prefix, addend, and output norm weight.
- [ ] Compare raw/materialized/output tensors against the small-op reference.
- [ ] Disable A3 prefill routing if cache mode is slower or fails any semantic case.

### Task 5: Compare against the A3 small-op baseline

**Files:**
- Create: `tests/e2e/nightly/single_node/ops/singlecard_ops/attn_res_a3_perf.py`
- Modify: `docs/features/attn_res_a3_migration_analysis.md`

**Interfaces:**
- Consumes: validated A3 fused kernels and equivalent small-op reference.
- Produces: P50/P90/P99 table and routing decision.

- [ ] Implement identical-input, identical-warmup, identical-repeat timing for small-op composition and fused operator.
- [ ] Cover T=1, 3, 7; B=1, 3, 8, 64; H=128, 256, 4096, 7168; valid=0, 1, 4, 8, 63, 64 where supported.
- [ ] Use NPU events or synchronized device timing and run each package/case in a fresh process.
- [ ] Require fused P50 and P90 no greater than the A3 small-op baseline; record variance and rejected cases.

### Task 6: Final build, review, and report

**Files:**
- Modify: `docs/features/attn_res_a3_migration_analysis.md`
- Test: build/install logs and complete regression report.

**Interfaces:**
- Consumes: Tasks 1-5 evidence.
- Produces: reviewable A3 migration result without unrelated archive files.

- [ ] Build A3 and verify the installed `.so` and custom OPP package paths.
- [ ] Run the complete precision and performance matrix one final time.
- [ ] Run A5 compile/regression checks to verify arch35 behavior is unchanged.
- [ ] Review `git diff`, remove unrelated generated archives from the deliverable, and record exact commands and results.
