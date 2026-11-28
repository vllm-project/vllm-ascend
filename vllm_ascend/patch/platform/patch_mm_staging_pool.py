#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Platform patch: route multimodal staging/merge pinned allocs through the
fixed staging pool (issue #21094).

Why:
    torch_npu 2.10 pinned allocations go through ``aclrtMallocHostWithCfg``
    (VA flag), binding every allocation to 2MiB base-heap physical pages
    with no small-page fallback. vllm's multimodal input path allocates
    per-request staging temporaries (``_nested_tensors_h2d``) and
    cross-request merge outputs (``MultiModalFlatField._reduce_data``,
    ``MultiModalBatchedField._reduce_data``) via ``new_empty(pin_memory=True)``,
    producing a high-frequency stream of variable-size driver grabs (tiny
    scheduling metadata plus image staging/merges, observed 85MB-1GB).
    Under host memory pressure the 2MiB buddy inventory drains and these
    grabs fail with error 207001/EL0018, crashing prefill serving.

How:
    The three allocation sites are re-pointed to
    ``vllm_ascend.mm_staging_pool`` (a fixed pinned pool allocated once per
    process at first use, full DMA speed, zero runtime driver grabs;
    multi-slot contiguous slices serve oversize merges, event-gated slot
    recycle recorded after the H2D enqueue). Merge outputs flow into
    ``_nested_tensors_h2d`` immediately after ``_reduce_data`` in
    ``BaseMultiModalField.reduce_data``, so their slices are recycled by
    the same post-enqueue release. Every pool interaction is guarded and
    degrades to the original allocation path. Configure via
    ``additional_config.mm_staging_config`` (``pinned_pool_mb`` /
    ``slot_mb``); ``pinned_pool_mb=0`` disables the pool.

    Import-safety note (validated 2026-11-22, gen A/B on issue21094 boxes):
    importing ``vllm.multimodal.inputs`` during platform-plugin registration
    breaks the ApiServer->DP-coordinator->EngineCore request dispatch
    (EngineCores starve on their input rings; requests hang silently).
    Therefore this patch NEVER imports vllm multimodal modules itself: it
    installs a ``sys.meta_path`` import hook that applies the attribute
    patches right AFTER vllm imports ``vllm.multimodal.inputs`` on its own
    (or immediately if the module is already in ``sys.modules``). Processes
    that never import the mm path are never touched.

Related PR (if no, explain why):
    No upstream vllm PR: the failure comes from an NPU-driver-specific
    allocation behavior (ACL_RT_MEM_ATTR_VA_FLAG base-heap pinning) that
    does not exist on CUDA, so the fix lives in the vllm-ascend platform
    patch layer instead of vllm core. The allocator-side alternative is
    tracked separately in torch_npu MR 49638 (``pinned_mem_register``
    gate with plain small-page fallback).

Future Plan:
    Keep while torch_npu lacks a failure fallback for the VA pinned path;
    revisit when the allocator provides small-page fallback natively.
"""

import sys
from typing import cast

import torch
from vllm.logger import init_logger

from vllm_ascend.mm_staging_pool import alloc_view, release_after_enqueue

logger = init_logger(__name__)

_PATCH_DONE = False

# Module attributes this patch requires from vllm.multimodal.inputs. When
# any of them is missing (vLLM version drift) the patch is skipped with a
# warning instead of breaking the import: fail-open to the original path.
_REQUIRED_ATTRS = (
    "_nested_tensors_h2d",
    "json_map_leaves",
    "_is_dense",
    "is_list_of",
    "MultiModalFlatField",
    "MultiModalBatchedField",
)


def _patched_nested_tensors_h2d(_json_map_leaves, _is_dense):
    """Build the pool-routed replacement for ``_nested_tensors_h2d``."""

    def _patched(tensors, device, *, pin_memory=False):
        if device is None:
            return tensors

        def _h2d(x):
            if not isinstance(x, torch.Tensor):
                return x
            if pin_memory and x.is_cpu and not (x.is_pinned() and _is_dense(x)):
                staged = alloc_view(tuple(x.shape), x.dtype)
                if staged is not None:
                    staged.copy_(x)
                    x = staged
                else:
                    # Ensure tensor is pinned and dense for non_blocking H2D copy.
                    x = x.new_empty(x.shape, pin_memory=True).copy_(x)
            out_t = x.to(device=device, non_blocking=True)
            if pin_memory:
                # Release pool slots only after the H2D copy has been
                # enqueued (the event recorded here completes after it).
                release_after_enqueue(x)
            return out_t

        return _json_map_leaves(_h2d, tensors)

    return _patched


def _patched_flat_reduce_data(_is_list_of):
    """Build the pool-routed replacement for ``MultiModalFlatField._reduce_data``.

    ``alloc_view``/``release_after_enqueue`` are resolved from this module's
    globals at call time (late binding) so tests can substitute them.
    """

    def _patched(self, batch, *, pin_memory):
        if len(batch) > 0 and _is_list_of(batch, torch.Tensor, check="all"):
            batch = cast(list[torch.Tensor], batch)
            if len(batch) == 1:
                return batch[0]

            dim = self.dim + (self.dim < 0) * len(batch[0].shape)

            def _shape_before_after(tensor):
                return tensor.shape[:dim], tensor.shape[dim + 1 :]

            first_shape = _shape_before_after(batch[0])

            if all(_shape_before_after(elem) == first_shape for elem in batch):
                shape_before, shape_after = first_shape
                shape_concat = sum(item.shape[dim] for item in batch)
                out = None
                if pin_memory:
                    out = alloc_view((*shape_before, shape_concat, *shape_after), batch[0].dtype)
                if out is None:
                    out = batch[0].new_empty((*shape_before, shape_concat, *shape_after), pin_memory=pin_memory)
                return torch.concat(batch, dim=self.dim, out=out)

            # Variable-length case: non-concat dimensions differ.
            # Use slice-assign approach (more efficient than padding).
            ndim = batch[0].ndim
            max_sizes: list[int] = []
            for d in range(ndim):
                if d == dim:
                    max_sizes.append(sum(t.shape[d] for t in batch))
                else:
                    max_sizes.append(max(t.shape[d] for t in batch))

            out = None
            if pin_memory:
                out = alloc_view(tuple(max_sizes), batch[0].dtype)
                if out is not None:
                    out.zero_()  # preserve torch.zeros semantics
            if out is None:
                out = batch[0].new_zeros(max_sizes, pin_memory=pin_memory)

            concat_offset = 0
            for tensor in batch:
                slices: list[slice] = []
                for d in range(ndim):
                    if d == dim:
                        slices.append(slice(concat_offset, concat_offset + tensor.shape[d]))
                    else:
                        slices.append(slice(0, tensor.shape[d]))
                out[tuple(slices)] = tensor
                concat_offset += tensor.shape[dim]

            return out

        assert self.dim == 0, "dim == 0 is required for nested list"
        return [e for elem in batch for e in elem]

    return _patched


def _patched_batched_reduce_data(_is_list_of):
    """Build the pool-routed replacement for ``MultiModalBatchedField._reduce_data``."""

    def _patched(self, batch, *, pin_memory):
        if len(batch) > 0 and _is_list_of(batch, torch.Tensor, check="all"):
            batch = cast(list[torch.Tensor], batch)
            if len(batch) == 1:
                return batch[0].unsqueeze(0)
            first_shape = batch[0].shape
            if all(elem.shape == first_shape for elem in batch):
                out = None
                if pin_memory:
                    out = alloc_view((len(batch), *first_shape), batch[0].dtype)
                if out is None:
                    out = batch[0].new_empty((len(batch), *first_shape), pin_memory=pin_memory)
                return torch.stack(batch, out=out)

        return batch

    return _patched


def _patch_inputs_module(mm_inputs) -> None:
    """Apply the attribute patches to an already-imported mm inputs module.

    Fail-open: when the module surface does not match (vLLM version drift),
    log once and keep the original behavior instead of breaking the import.
    """
    global _PATCH_DONE
    if _PATCH_DONE or getattr(mm_inputs, "_mm_staging_pool_patched", False):
        return

    missing = [name for name in _REQUIRED_ATTRS if not hasattr(mm_inputs, name)]
    if missing:
        logger.warning(
            "mm staging pool patch skipped: vllm.multimodal.inputs is missing "
            "%s (vLLM version drift); keeping the original allocation path",
            missing,
        )
        return

    mm_inputs._nested_tensors_h2d = _patched_nested_tensors_h2d(mm_inputs.json_map_leaves, mm_inputs._is_dense)
    mm_inputs.MultiModalFlatField._reduce_data = _patched_flat_reduce_data(mm_inputs.is_list_of)
    mm_inputs.MultiModalBatchedField._reduce_data = _patched_batched_reduce_data(mm_inputs.is_list_of)
    mm_inputs._mm_staging_pool_patched = True
    _PATCH_DONE = True
    logger.info(
        "mm staging pool patch applied: h2d + FlatField/BatchedField._reduce_data routed to the fixed pinned pool"
    )


class _MMInputsPatchLoader:
    """Wraps the real loader; applies the patch right after module exec."""

    def __init__(self, real_loader):
        self._real = real_loader

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        try:
            _patch_inputs_module(module)
        except Exception as e:  # never break vllm's own import
            logger.warning("mm staging pool post-import patch failed: %s", e)


class _MMInputsPatchFinder:
    """Delegating finder: intercepts only vllm.multimodal.inputs."""

    _NAME = "vllm.multimodal.inputs"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._NAME or self._NAME in sys.modules:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None and spec.loader is not None:
                spec.loader = _MMInputsPatchLoader(spec.loader)
                return spec
        return None


def _npu_usable() -> bool:
    """Whether this process can create NPU pinned pools and events.

    Checked at patch-install time to keep CPU-only environments (CI unit
    tests) free of import hooks; the real runtime gate remains the pool
    config (``pinned_pool_mb``).
    """
    try:
        return bool(hasattr(torch, "npu") and torch.npu.is_available())
    except Exception:
        return False


def _apply_patch() -> None:
    global _PATCH_DONE
    if _PATCH_DONE or not _npu_usable():
        return
    existing = sys.modules.get("vllm.multimodal.inputs")
    if existing is not None:
        # Already imported by vllm before us: patch in place (no new import).
        try:
            _patch_inputs_module(existing)
        except Exception as e:
            logger.warning("mm staging pool in-place patch failed: %s", e)
        return
    if not any(isinstance(f, _MMInputsPatchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MMInputsPatchFinder())
        logger.info("mm staging pool import hook installed (activates on vllm.multimodal.inputs import)")


_apply_patch()
