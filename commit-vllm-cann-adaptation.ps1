$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\code\vllm-ascend'

$expectedBranch = 'integrate/dsnext-megamoe-vllm-cann'
$expectedBaseline = '0040652a66d31a2b277588ac90d7d37e92757bc5'
if ((git branch --show-current).Trim() -ne $expectedBranch) { throw 'Unexpected branch; nothing was committed.' }
$baseline = (git rev-parse rfc/vllm_cann).Trim()
if ($baseline -ne $expectedBaseline) { throw 'The RFC baseline changed; please have the adaptation checked again.' }
git merge-base --is-ancestor $expectedBaseline HEAD
if ($LASTEXITCODE -ne 0) { throw 'The RFC baseline is not an ancestor of HEAD.' }
if ((git rev-list --count "$expectedBaseline..HEAD").Trim() -ne '3') {
    throw 'Expected three source commits before creating the adaptation commit.'
}
if (@(git diff --cached --name-only).Count -gt 0) {
    throw 'There are already staged changes; please review them before running this script.'
}

$adaptationFiles = @(
    'docs/source/user_guide/configuration/additional_config.md',
    'tests/ut/ops/test_dsnext_megamoe_port.py',
    'tests/ut/ops/test_moe_comm_method.py',
    'tests/ut/test_ascend_forward_context.py',
    'tests/ut/worker/test_model_runner_v1.py',
    'vllm_ascend/ascend_config.py',
    'vllm_ascend/ascend_forward_context.py',
    'vllm_ascend/distributed/parallel_state.py',
    'vllm_ascend/ops/fused_moe/dataclass/fused_experts.py',
    'vllm_ascend/ops/fused_moe/dataclass/shared_experts.py',
    'vllm_ascend/ops/fused_moe/fused_moe.py',
    'vllm_ascend/ops/fused_moe/mega_moe.py',
    'vllm_ascend/ops/fused_moe/moe_comm_method.py',
    'vllm_ascend/ops/fused_moe/prepare_finalize.py',
    'vllm_ascend/ops/fused_moe/routed_experts.py',
    'vllm_ascend/ops/fused_moe/router/fused_topk_router.py',
    'vllm_ascend/ops/fused_moe/shared_experts.py',
    'vllm_ascend/patch/worker/patch_distributed.py',
    'vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py',
    'vllm_ascend/worker/model_runner_v1.py'
)
$unexpectedFiles = @(git diff --name-only HEAD | Where-Object { $_ -notin $adaptationFiles })
if ($unexpectedFiles.Count -gt 0) { throw ('Unexpected tracked changes: ' + ($unexpectedFiles -join ', ')) }

git add -- $adaptationFiles
if ($LASTEXITCODE -ne 0) { throw 'Unable to stage adaptation files.' }
git -c core.whitespace=cr-at-eol diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed.' }

git commit -s -m 'fix(moe): adapt DSNext MegaMoE commits to vllm_cann' -m 'Integrate the three source PRs through the RFC FusedMC2CommImpl and RoutedMoEMilestones interfaces. Preserve DeviceOperator.compute_gate_logits, the RFC quantization weight interface, HCCL lifecycle and KVPP features.' -m 'Validate shared-expert operator arguments and weight layouts, clear cached shared weights after reload, bound A5 input capacity by the execution role, and complete idle-DP metadata propagation. Preserve the hardware-selected vision/hash routing path and standalone shared-expert fallback.' -m 'Validation: 29 host regression tests, compilation of 18 Python files, Ruff lint/format, context-manager AST and Git whitespace checks passed. Confirmed 13 RFC definitions and four complete files retain their baseline behavior. NPU inference, precision, performance and full runtime tests remain pending. bash format.sh ci could not run because of the Windows sandbox shared-memory restriction.'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; please send the error output.' }
if ((git rev-list --count "$expectedBaseline..HEAD").Trim() -ne '4') { throw 'Unexpected final commit count.' }
git log -4 --oneline
Write-Output 'Done: three source commits plus one RFC adaptation commit. Nothing was pushed.'
