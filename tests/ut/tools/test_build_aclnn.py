from pathlib import Path

BUILD_ACLNN_SCRIPT = Path(__file__).parents[3] / "csrc" / "build_aclnn.sh"
TORCH_BINDING_SOURCE = Path(__file__).parents[3] / "csrc" / "torch_binding.cpp"


def _custom_ops_for_soc(script: str, soc_pattern: str) -> set[str]:
    branch_marker = f'elif [[ "$SOC_VERSION" =~ ^{soc_pattern} ]]; then'
    assert branch_marker in script, f"Missing {soc_pattern} build_aclnn.sh branch"

    section = script.split(branch_marker, maxsplit=1)[1]
    for next_branch_marker in ("\nelif ", "\nelse"):
        section = section.split(next_branch_marker, maxsplit=1)[0]

    return {
        line.strip().strip('"')
        for line in section.splitlines()
        if line.strip().startswith('"') and line.strip().endswith('"')
    }


def test_a3_uses_official_rms_norm_dynamic_quant() -> None:
    script = BUILD_ACLNN_SCRIPT.read_text()

    a2_ops = _custom_ops_for_soc(script, "ascend910b")
    a3_ops = _custom_ops_for_soc(script, "ascend910_93")

    assert "rms_norm_dynamic_quant" in a2_ops
    assert "rms_norm_dynamic_quant" not in a3_ops


def test_a3_uses_official_dsv4_sparse_operators() -> None:
    script = BUILD_ACLNN_SCRIPT.read_text()

    a2_ops = _custom_ops_for_soc(script, "ascend910b")
    a3_ops = _custom_ops_for_soc(script, "ascend910_93")

    replaced_ops = {
        "vllm_quant_lightning_indexer",
        "vllm_quant_lightning_indexer_metadata",
        "quant_lightning_indexer_v2",
        "quant_lightning_indexer_v2_metadata",
        "sparse_attn_sharedkv",
        "sparse_attn_sharedkv_metadata",
    }
    assert replaced_ops <= a2_ops
    assert replaced_ops.isdisjoint(a3_ops)


def test_official_rms_norm_dynamic_quant_signature() -> None:
    source = TORCH_BINDING_SOURCE.read_text()
    invocation = (
        "EXEC_NPU_CMD(aclnnRmsNormDynamicQuant, x, gamma, smooth_scale, beta, "
        "epsilon, DST_TYPE_INT8, y_out, scale_out);"
    )

    assert invocation in source
    assert "smooth_scale2" not in source
    assert "y2_out" not in source
    assert "scale2_out" not in source
