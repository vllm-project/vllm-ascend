# SPDX-License-Identifier: Apache-2.0

from tools.catccos_profiling.summarize_moe_stats import Sample, parse_name, summarize


def test_parse_phase_range_preserves_global_and_rank_m():
    parsed = parse_name(
        "vllm_ascend.moe.catccos.finalize.tp_all_gather[global_M=4096,rank_M=1024,H=2048,topK=8]",
        tp_size=4,
    )

    assert parsed == ("catccos", "finalize.tp_all_gather", 4096, 1024)


def test_parse_legacy_m_range_infers_global_m():
    parsed = parse_name("catccos.a5.binding.kernel_launch_host[M=1024,H=2048,N=1536,topK=8]", tp_size=4)

    assert parsed == ("catccos", "kernel_launch_host", 4096, 1024)


def test_parse_legacy_global_ranges_keeps_global_m():
    parsed = parse_name("vllm_ascend.moe.catccos.host_moe_body[M=4096,H=2048,topK=8]", tp_size=4)

    assert parsed == ("catccos", "host_moe_body", 4096, 4096)


def test_parse_legacy_allgather_pipeline_keeps_global_m():
    parsed = parse_name("vllm_ascend.moe.native_allgather.pipeline_host_scope[M=4096,H=2048,topK=8]", tp_size=4)

    assert parsed == ("native_allgather", "pipeline_host_scope", 4096, 4096)


def test_summarize_reports_component_share_of_body():
    rows = summarize(
        [
            Sample("catccos", "host_moe_body", 4096, 4096, 1000.0, 0.0, "rank0"),
            Sample("catccos", "host_moe_body", 4096, 4096, 1200.0, 0.0, "rank1"),
            Sample("catccos", "finalize.tp_all_gather", 4096, 1024, 250.0, 200.0, "rank0"),
            Sample("catccos", "finalize.tp_all_gather", 4096, 1024, 300.0, 220.0, "rank1"),
        ]
    )

    gather = next(row for row in rows if row["stage"] == "finalize.tp_all_gather")
    assert gather["host_p50_us"] == 275.0
    assert gather["host_share_of_body_pct"] == 25.0
