# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry for the three-buffer benchmark workspace."""

from vllm_ascend.ops.tilelang.kda import WholeKDA as _OptimizedWholeKDA
from vllm_ascend.ops.tilelang.kda_legacy import whole_program as whole_program


class _Workspace(tuple):
    """Expose the three public buffers and retain the two gate buffers with them."""

    def __new__(cls, buffers):
        workspace = super().__new__(cls, buffers[:3])
        workspace._optimized_buffers = buffers
        return workspace


class WholeKDA:
    """Run the optimized QKV-to-O kernel with the original three-buffer interface."""

    def __init__(self, layer):
        self._optimized = _OptimizedWholeKDA(layer)

    @staticmethod
    def allocate(x):
        return _Workspace(_OptimizedWholeKDA.allocate(x))

    def __call__(self, x, metadata, workspace=None):
        if isinstance(workspace, _Workspace):
            workspace = workspace._optimized_buffers
        return self._optimized(x, metadata, workspace)
