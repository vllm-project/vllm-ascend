# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.model_executor.warmup.helpers import make_mock_vllm_config
from vllm_ascend.ops.triton import hisparse


def test_single_warmup_key_compiles_without_executing():
    launcher = hisparse.TritonScatterMinRepresentativesKernel()
    keys = launcher.get_warmup_keys(make_mock_vllm_config())
    assert keys == [launcher.CompileKey(block_size=256)]
    kernel = MagicMock(arg_names=["index_ptr", "source_ptr", "representatives_ptr", "num_rows", "BLOCK_SIZE"])
    with patch.object(launcher, "kernel", kernel):
        launcher.compile(keys[0])
    kernel.__getitem__.assert_not_called()
    kernel.warmup.assert_called_once()
    args = kernel.warmup.call_args.kwargs
    assert set(args) == {"index_ptr", "source_ptr", "representatives_ptr", "num_rows", "BLOCK_SIZE", "grid"}
    assert args["num_rows"] == args["BLOCK_SIZE"] == 256
    assert all(args[name].dtype == torch.int64 for name in ("index_ptr", "source_ptr", "representatives_ptr"))


@pytest.mark.parametrize("rows", [0, 257])
def test_public_wrapper_shares_launcher_and_skips_empty(rows):
    def vector(size):
        return SimpleNamespace(ndim=1, dtype=torch.int64, shape=(size,), numel=lambda: size, is_contiguous=lambda: True)

    out, index, source = vector(64), vector(rows), vector(rows)
    with patch.object(hisparse, "_TRITON_SCATTER_MIN_REPRESENTATIVES") as launcher:
        assert hisparse.scatter_min_representatives(out, index, source) is out
    if rows:
        launcher.assert_called_once_with(index, source, out, rows, 256)
    else:
        launcher.assert_not_called()


def test_reject_noncontiguous_inputs_before_launch():
    x = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError, match="contiguous int64"):
        hisparse.scatter_min_representatives(x, x[::2], x[::2])
