from __future__ import annotations

import pytest

from tools.vllm_interface_contracts.analysis_plans import resolve_analysis_plan


def test_main2main_plan_keeps_full_exact_contract_analysis() -> None:
    plan = resolve_analysis_plan("main2main")
    assert plan.relation_types == {"inheritance", "override", "monkey_patch"}
    assert plan.analyze_direct_imports
    assert plan.analyze_direct_calls
    assert plan.include_generator_findings


def test_vllm_interface_plan_excludes_downstream_patch_scope() -> None:
    plan = resolve_analysis_plan("vllm-interface")
    assert plan.relation_types == {"override"}
    assert plan.collect_inheritance
    assert not plan.analyze_inheritance
    assert not plan.collect_monkey_patches
    assert plan.analyze_direct_imports
    assert plan.analyze_direct_calls
    assert not plan.include_generator_findings


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported scenario"):
        resolve_analysis_plan("custom")
