# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise production host dispatch for both paired and current Core ABIs."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("current_core", [False, True])
@pytest.mark.parametrize("draft_kind", ["none", "tensor", "replicated_cache", "mapped_cache"])
@pytest.mark.parametrize("block_verification", [False, True])
def test_sampler_dispatch_preserves_row_mapping_and_stats_extent(current_core, draft_kind, block_verification):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py"
    function = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "rejection_sample"
    )
    for argument in function.args.args:
        argument.annotation = None
    function.returns = None
    launches = {}

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches[self.name] = (args, kwargs)

            return launch

    namespace = dict(
        torch=torch,
        triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        _CORE_DRAFT_ROW_MAPPING=current_core,
        _CORE_STATS_USES_NUM_LOGITS=current_core,
        resample=lambda *args, **kwargs: launches.update(resample=(args, kwargs)),
    )
    for name in (
        "_compute_block_stats_kernel",
        "_compute_cumulative_log_p_kernel",
        "_compute_local_residual_mass_kernel",
        "_probabilistic_rejection_kernel",
    ):
        namespace[name] = Kernel(name)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), namespace
    )
    values = torch.empty((3, 3, 257))
    mapping = torch.tensor([2, 0, 1], dtype=torch.int32) if draft_kind == "mapped_cache" else None
    draft = None if draft_kind == "none" else values
    if draft_kind.endswith("cache"):
        draft = SimpleNamespace(values=values, row_mapping=mapping)
    args = (
        torch.empty((8, 257)),
        draft,
        torch.empty(8, dtype=torch.int64),
        torch.tensor([0, 4, 8]),
        torch.arange(8),
        torch.tensor([0, 2]),
        torch.tensor([0] * 4 + [2] * 4),
        torch.tensor([0, 1, 2, 3] * 2),
        torch.zeros(3),
        torch.zeros(3, dtype=torch.int64),
        3,
    )
    if not current_core and mapping is not None:
        with pytest.raises(ValueError, match="do not support mapped draft-cache rows"):
            namespace[function.name](*args, use_block_verification=block_verification)
        assert not launches
        return
    namespace[function.name](*args, use_block_verification=block_verification)
    stats, _ = launches["_compute_block_stats_kernel"]
    assert len(stats) == (21 if current_core else 20)
    if current_core:
        assert stats[15] is mapping
    assert stats[-1] == (8 if current_core else 3)
    kernel = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_probabilistic_rejection_kernel"
    )
    actual_parameters = dict(zip((arg.arg for arg in kernel.args.args), launches[kernel.name][0]))
    assert actual_parameters["draft_logits_row_mapping_ptr"] is mapping
    assert launches["resample"][1]["draft_logits_row_mapping"] is mapping
    if draft_kind != "none":
        assert stats[12] is values
    if block_verification:
        cumulative = launches["_compute_cumulative_log_p_kernel"][0]
        if current_core:
            assert cumulative[11] is mapping
