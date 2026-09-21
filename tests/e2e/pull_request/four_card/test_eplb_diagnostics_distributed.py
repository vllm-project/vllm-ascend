# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four NPU processes: independent PP-like groups and idle DP participation."""

import io
import logging
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("torch_npu")

from vllm_ascend.eplb.diagnostics.config import EplbDiagnosticsConfig
from vllm_ascend.eplb.diagnostics.decision import make_planner
from vllm_ascend.eplb.diagnostics.probe import ExpertLoadProbe
from vllm_ascend.eplb.diagnostics.runtime import DiagnosticsRecorder


@torch.inference_mode()
def _worker(rank, rendezvous, mixed_workload):
    torch.npu.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        members = ([0, 2], [1, 3])
        groups = [dist.new_group(ranks, backend="gloo") for ranks in members]
        stage, local = rank % 2, members[rank % 2].index(rank)
        group = SimpleNamespace(ranks=members[stage], rank_in_group=local, cpu_group=groups[stage])
        probe = ExpertLoadProbe(4, "npu")
        opposite = ExpertLoadProbe(4, "npu")
        source = torch.tensor(3, device="npu")
        probe.source_token_count, probe.source_positions = source, torch.arange(4, device="npu")
        opposite.source_token_count, opposite.source_positions = source, probe.source_positions
        slot = torch.zeros(1, dtype=torch.int64, device="npu")
        for item in (probe, opposite):
            item.history = torch.zeros((2, 6), dtype=torch.int64, device="npu")
            item.history_slot = slot
        ids = torch.tensor([[0], [1], [0], [99]], dtype=torch.int64, device="npu")
        other_ids = torch.tensor([[2], [3], [2], [99]], dtype=torch.int64, device="npu")
        mask = torch.ones(4, dtype=torch.bool, device="npu")
        probe.record_routes(ids, mask)
        opposite.record_routes(other_ids, mask)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            probe.record_routes(ids, mask)
            opposite.record_routes(other_ids, mask)
        graph.replay()
        torch.npu.synchronize()
        layer = SimpleNamespace(
            eplb_diagnostic_probe=probe,
            local_num_experts=2,
            ascend_expert_map=torch.tensor([0, 1, -1, -1] if local == 0 else [-1, -1, 0, 1]),
            moe_config=SimpleNamespace(experts_per_token=1),
        )
        config = EplbDiagnosticsConfig(mode="observe", window_size=2, warmup_steps=0, max_windows=0)
        other_layer = SimpleNamespace(**dict(vars(layer), eplb_diagnostic_probe=opposite))
        from vllm.distributed import parallel_state

        runner = SimpleNamespace(
            model=SimpleNamespace(num_expert_groups=1),
            vllm_config=SimpleNamespace(
                use_v2_model_runner=True, parallel_config=SimpleNamespace(eplb_config=SimpleNamespace(policy="default"))
            ),
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(parallel_state, "get_node_count", lambda: 1)
            planner, policy = make_planner(runner, 2)
        recorder = DiagnosticsRecorder(
            config, [("layer.0", layer), ("layer.1", other_layer)], group, [rank], stage, source, planner, policy
        )
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger = logging.getLogger("vllm.eplb.diagnostics")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            for phase in ("decode", "mixed", "prefill") if mixed_workload else ("decode",) * 3:
                for tokens, dummy in ((3, False), (2, local == 1)):
                    recorder.begin(tokens, dummy)
                    if not dummy:
                        recorder.phases.add((phase if local == 0 else "decode", "FULL"))
                    graph.replay()
                    recorder.end()
            recorder.begin(1, dummy=local == 1)
            graph.replay()
            recorder.end()
            recorder.finish()
            recorder.finish()
            if local == 0:
                assert output.getvalue().count("[EPLB diagnostic]") == 8
                assert output.getvalue().count("[EPLB experts]") == 16
                assert "candidate_layers=['layer.0', 'layer.1']" in output.getvalue()
                assert "cumulative_work=25" in output.getvalue()
                assert output.getvalue().count("final=True") == 1
                assert "valid_assignments=8 rank_work=[8, 0]" in output.getvalue()
                assert "hint=recommend_trial" in output.getvalue()
                assert "timing_evidence=not_collected net_benefit=unknown" in output.getvalue()
                if mixed_workload:
                    assert "mixed" in output.getvalue() and "prefill" in output.getvalue()
                assert recorder.summary.layers["layer.0"].decision.windows == 3
                assert recorder.summary.layers["layer.0"].decision.pairs == 2
                assert recorder.summary.layers["layer.0"].decision.report()["heldout_peak_work_reduction"] > 0
                assert f"stage={stage}" in output.getvalue()
            else:
                assert not output.getvalue()
        finally:
            logger.removeHandler(handler)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mixed_workload", [False, True], ids=["decode", "mixed"])
def test_graph_window_gather_with_idle_dp_and_separate_stages(tmp_path, mixed_workload):
    rendezvous = (tmp_path / "gloo-init").as_uri()
    mp.spawn(_worker, args=(rendezvous, mixed_workload), nprocs=4, join=True)
