# SPDX-License-Identifier: Apache-2.0
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.glm_reduced.compare_prefix_runs import compare_runs
from tools.glm_reduced.prefix_probe import compare_arrays, project_boundary


class Tensor(np.ndarray):
    def clone(self):
        return self.copy()

    def contiguous(self):
        return self.copy()


def test_projection_preserves_original_residual_and_uses_real_head():
    hidden = np.array([[1.0, 2.0], [3.0, 4.0]]).view(Tensor)
    residual = np.ones_like(hidden)

    def norm(h, r):
        assert h.shape == (2, 2)  # Normalize the original token batch before selecting a row.
        h += r  # Model a fused implementation that mutates inputs.
        r[:] = 0
        return h * 2, r

    model = SimpleNamespace(model=SimpleNamespace(norm=norm), compute_logits=lambda x: x * 3)
    output = project_boundary(model, hidden, residual, token_count=2, row_indices=[1], gather_rows=lambda x: x)
    np.testing.assert_array_equal(output["hidden"], [[4, 5]])
    np.testing.assert_array_equal(output["logits"], [[24, 30]])
    np.testing.assert_array_equal(hidden, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(residual, np.ones((2, 2)))


def test_metrics_do_not_invent_a_passing_threshold():
    assert compare_arrays([[1, 2]], [[1, 2]])["status"] == "UNASSESSED"
    assert compare_arrays([[1, 2]], [[1, 2.1]], atol=0.01, rtol=0)["status"] == "FAIL"
    assert compare_arrays([[1, 2]], [[1, 2.1]], atol=0.2, rtol=0)["status"] == "PASS"


@pytest.mark.parametrize("candidate", [[[float("nan"), 2]], [[float("inf"), 2]], [[1]], []])
def test_reject_incomplete_or_nonfinite(candidate):
    with pytest.raises(ValueError):
        compare_arrays([[1, 2]], candidate)


def test_equal_argmax_does_not_hide_numeric_error():
    result = compare_arrays([[1, 10]], [[1, 20]], atol=0, rtol=0)
    assert result["argmax_agreement"] == 1
    assert result["status"] == "FAIL"
    assert result["first_mismatch"] == [0, 1]


def test_run_comparison_requires_complete_logits_and_matching_history(tmp_path):
    metadata = {
        "status": "COLLECTED_NOT_COMPARED",
        "engine": {"tensor_parallel_size": 1},
        "requests": [{"id": "one"}],
        "continuation": [17],
        "runtime": {"revision": "fixed"},
        "scope": "test fixture",
        "probe_sha256": {"fixture": "same"},
        "environment": {},
    }
    paths = [tmp_path / name for name in ("reference", "candidate")]
    for path in paths:
        metadata["role"] = path.name
        directory = path / "one/rank-0"
        directory.mkdir(parents=True)
        (path / "run.json").write_text(json.dumps(metadata))
        np.savez(directory / "step-000.npz", hidden=[[1, 2]], normalized=[[2, 3]], logits=[[3, 4]], position=[0])
        np.savez(directory / "final-000.npz", logits=[[3, 4]])
        np.savez(directory / "actual-norm-000.npz", normalized=[[2, 3]])
    assert compare_runs(*paths)["all_arrays_exact"]
    np.savez(paths[1] / "one/rank-0/step-000.npz", hidden=[[1, 2]], normalized=[[2, 3]], position=[0])
    with pytest.raises(ValueError, match="capture fields"):
        compare_runs(*paths)
    metadata["continuation"] = [18]
    (paths[1] / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="continuation"):
        compare_runs(*paths)


def test_worker_probe_captures_before_forcing_and_restores_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tools.glm_reduced.run_prefix_probe import PrefixProbeWorkerExtension

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(
            get_tp_group=lambda: SimpleNamespace(rank_in_group=0),
            tensor_model_parallel_all_gather=lambda tensor, dim: tensor,
        ),
    )

    class Layer(torch.nn.Module):
        def forward(self, positions, hidden, residual):
            return hidden, residual

    class Norm(torch.nn.Module):
        def forward(self, hidden, residual):
            return hidden + residual, residual

    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=8, layers=[Layer() for _ in range(8)], norm=Norm()),
        compute_logits=lambda hidden: hidden * 2,
    )
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    original = model.compute_logits
    PrefixProbeWorkerExtension.install_prefix_probe(worker, str(tmp_path), 2, [1])
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    residual = torch.ones_like(hidden)
    model.model.layers[7](torch.tensor([0, 1]), hidden, residual)
    assert not (tmp_path / "rank-0/actual-norm-000.npz").exists()
    model.model.norm(hidden, residual)
    with np.load(tmp_path / "rank-0/actual-norm-000.npz") as data:
        np.testing.assert_array_equal(data["normalized"], [[4, 5]])
    forced = model.compute_logits(torch.tensor([[4.0, 5.0]]))
    assert forced[0, 1] == 0 and torch.isneginf(forced[0, 0])
    with np.load(tmp_path / "rank-0/step-000.npz") as data:
        np.testing.assert_array_equal(data["logits"], [[8, 10]])
    assert PrefixProbeWorkerExtension.remove_prefix_probe(worker) == {"steps": 1}
    assert model.compute_logits is original
    assert not model.model.layers[7]._forward_hooks
    assert not model.model.norm._forward_hooks
