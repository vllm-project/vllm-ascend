from __future__ import annotations

from tests.e2e.vllm_interface.vllm_interface_contracts.analysis_plans import resolve_analysis_plan


def test_vllm_interface_plan_has_one_fixed_ci_scope() -> None:
    plan = resolve_analysis_plan()

    assert plan.scenario == "vllm-interface"
    assert plan.relation_types == {"override"}
    capabilities = plan.capabilities()
    assert capabilities["inheritance_mro"] == {
        "state": "prerequisite",
        "produces_findings": False,
    }
    assert capabilities["override"]["state"] == "analyzed"
    assert capabilities["direct_import"]["state"] == "analyzed"
    assert capabilities["direct_call"]["state"] == "analyzed"
    assert capabilities["monkey_patch"]["state"] == "skipped"
    assert capabilities["generator_findings"]["state"] == "skipped"
