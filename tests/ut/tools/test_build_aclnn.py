from pathlib import Path

BUILD_ACLNN_SCRIPT = Path(__file__).parents[3] / "csrc" / "build_aclnn.sh"


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
