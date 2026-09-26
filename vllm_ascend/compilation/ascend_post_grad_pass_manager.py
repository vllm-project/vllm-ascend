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
#
"""PostGradPassManager for the inductor compile-backend track.

The upstream PostGradPassManager's always-on ``FixFunctionalizationPass``
builds its rope target list from CUDA-only ops (``torch.ops._C.rotary_embedding``,
``vllm/compilation/passes/utility/fix_functionalization.py``), which raises
AttributeError on NPU where the vLLM CUDA extension is absent. XPU carries an
early-return guard for the same reason; NPU does not. ``AscendPostGradPassManager``
keeps the upstream manager and swaps that one pass for a no-op (stage
design/stage1/03 R1, fallback plan F1).

Since stage3 it also chains the VA-owned Ascend fusion passes back onto the
track (stage design/stage3/02 §2): ``configure`` injects
AddRMSNormQuantFusionPass / QKNormRopeFusionPass / MulsAddFusionPass per
``ascend_compilation_config.fuse_*`` (default True, aligned with the legacy
GraphFusionPassManager track), plus the torch_npu elide pass wrapped as an
InductorPass (U7: VA-side wrapper, uuid folds the TN function's source hash).
"""

import inspect

import torch
import torch.fx
from vllm.compilation.passes.inductor_pass import InductorPass
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class AscendNoopInductorPass(VllmInductorPass):
    """A no-op pass used to replace passes that are unsafe on NPU.

    ``uuid()`` is inherited from ``InductorPass`` (source hash), so the
    inductor code cache keys change together with this implementation.
    """

    def __call__(self, graph: torch.fx.Graph) -> None:
        logger.debug("Skipping %s: replaced by AscendNoopInductorPass (not supported on NPU).", self.pass_name)


class AscendElideIntFloatIntPass(VllmInductorPass):
    """VA-side InductorPass wrapper for torch_npu's elide pass (stage3 U7).

    torch_npu installs ``_elide_int_float_int_roundtrip_pass`` into the global
    ``inductor_config.post_grad_custom_post_pass`` slot at import time, which
    vLLM's per-compile ``config.patch`` (thread-local) completely shadows — the
    pass never runs on the vLLM inductor track. This wrapper makes it an
    explicit list member instead. The uuid folds the wrapped function's source
    hash so a torch_npu change invalidates the cache (the CallableInductorPass
    gap called out by the stage3 red team).
    """

    def __init__(self, config: VllmConfig, elide_fn) -> None:
        super().__init__(config)
        self.pass_name = "AscendElideIntFloatIntPass"
        self._elide_fn = elide_fn

    def __call__(self, graph: torch.fx.Graph) -> None:
        self.begin()
        self._elide_fn(graph)
        self.end_and_log()

    def uuid(self) -> str:
        return InductorPass.hash_dict(
            {
                "src": super().uuid(),
                "elide_fn_source": inspect.getsource(self._elide_fn),
                "elide_fn_module": self._elide_fn.__module__,
            }
        )

    def is_applicable_for_range(self, compile_range) -> bool:
        return True


def _required_custom_ops_missing(vllm_config: VllmConfig) -> bool:
    """True when fuse-flagged passes depend on custom ops the dispatch excluded.

    ``fuse_norm_quant`` patterns need the oot rms-norm path and
    ``fuse_qknorm_rope`` needs ``rotary_embedding``; with ``custom_ops=none``
    (an explicit user choice or a non-AUTO_ENABLE_CUSTOM_OPS machine) those
    nodes never enter the graph and only the pure-aten muls_add pattern can
    still hit. Warn instead of fail-fast (stage3 D3-3): muls_add survives and
    ``custom_ops=none`` is a legitimate debugging shape.
    """
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        acc = get_ascend_config().ascend_compilation_config
        required = set()
        if acc.fuse_norm_quant:
            required.add("rms_norm")
        if acc.fuse_qknorm_rope:
            required.add("rotary_embedding")
        if not required:
            return False
        ops = set(getattr(vllm_config.compilation_config, "custom_ops", None) or [])
        if "all" in ops:
            return False
        enabled = {o.lstrip("+-") for o in ops if not o.startswith("-")}
        return bool(required - enabled)
    except Exception:  # noqa: BLE001 — a broken config shape must not break configure
        logger.debug("custom_ops mismatch check skipped (config shape unexpected)")
        return False


# torch_npu's bernoulli-family lowering only disables cudagraph_trees
# (V.graph.disable_cudagraphs_reason); the ACLGraph piecewise capture used by
# vLLM never reads it, so a captured RNG op would silently replay the recorded
# seed/offset instead of eager semantics (stage design/stage2/03 R2').
_RNG_OP_MARKER = "bernoulli"


def _warn_if_graph_contains_rng(graph: torch.fx.Graph) -> None:
    """Warn when the post-grad graph contains bernoulli-family RNG ops."""
    for node in graph.nodes:
        if node.op == "call_function" and _RNG_OP_MARKER in str(node.target):
            logger.warning(
                "Post-grad graph contains a RNG op (%s): under graph capture its replay "
                "is NOT eager-equivalent (the captured seed/offset is replayed). Verify "
                "numerical expectations for this model.",
                node.target,
            )
            return


class AscendPostGradPassManager(PostGradPassManager):
    """Upstream PostGradPassManager with NPU-unsafe always-on passes disabled.

    Only ``fix_functionalization`` is replaced today; the remaining always-on
    chain (post_cleanup / ir_lowering / clone_elimination) is platform-neutral
    FX bookkeeping and is kept as-is.
    """

    def __call__(self, graph: torch.fx.Graph) -> None:
        _warn_if_graph_contains_rng(graph)
        super().__call__(graph)

    def configure(self, config) -> None:
        super().configure(config)
        noop = AscendNoopInductorPass(config)
        noop.pass_name = "FixFunctionalizationPass(noop)"
        self.fix_functionalization = noop  # todo: Allen 这里为什么会炸？待定位
        self._inject_ascend_fusion_passes(config)

    def _inject_ascend_fusion_passes(self, config: VllmConfig) -> None:
        """Chain the VA fusion passes back onto the inductor track (stage3 02 §2).

        Mirrors GraphFusionPassManager.configure's conditions and order exactly
        (fuse flags from the validated AscendConfig singleton — NOT the
        upstream pass_config flags of the same names, which the track pins off
        to stay clear of CUDA/ROCm-only fusion classes).
        """
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.utils import is_310p

        try:
            acc = get_ascend_config().ascend_compilation_config
        except RuntimeError:
            # On the live track the AscendConfig singleton is always initialized
            # before compile (worker init). A bare-config caller (unit tests)
            # reaches here: warn and skip injection rather than crash configure.
            logger.warning(
                "AscendConfig not initialized; skip Ascend fusion-pass injection "
                "(fuse flags unreadable)."
            )
            return
        # Stage-4 batch2 #9 (R7): on the inductor track, triton_experimental
        # installs its npu-op decompositions (e.g. npu_rms_norm -> primitive
        # soup) at FIRST COMPILE — i.e. AFTER this configure-time pattern
        # registration. Patterns traced before the table install kept their
        # op anchors and could never match the AOT-decomposed graphs
        # (qknorm_rope was structurally 0-hit on the track; CPU-level leg3
        # single-variable proof in stage4/_notes/t0_probe). Install the table
        # BEFORE tracing the patterns so both sides see the same (decomposed)
        # graph form. Process-level idempotent; anchors on
        # _C_ascend.npu_add_rms_norm(_bias) are NOT decomposed by the table,
        # so the norm-quant patterns (W2, e2e-verified 55 hits) are
        # unaffected.
        try:
            from torch_npu._inductor.decomposition import _register_triton_experimental_decompositions

            _register_triton_experimental_decompositions()
        except Exception:
            logger.debug(
                "triton_experimental decompositions unavailable at fusion-pass "
                "injection; patterns keep their op-anchor form."
            )

        if acc.fuse_norm_quant and not is_310p():
            from .passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

            self.passes.append(
                AddRMSNormQuantFusionPass(config, dynamic_quant_fusion=acc.fuse_norm_quant_dynamic)
            )

        if acc.fuse_qknorm_rope:
            from .passes.qknorm_rope_fusion_pass import QKNormRopeFusionPass

            self.passes.append(QKNormRopeFusionPass(config))

        if acc.fuse_muls_add and not is_310p():
            from .passes.muls_add_pass import MulsAddFusionPass

            self.passes.append(MulsAddFusionPass(config))

        if _required_custom_ops_missing(config):
            logger.warning(
                "Ascend fusion passes are enabled (fuse_norm_quant/fuse_qknorm_rope) but "
                "the required custom_ops (rms_norm / rotary_embedding) are not in "
                "compilation_config.custom_ops — those patterns will not match. Only the "
                "pure-aten muls_add pattern can still fire. Set custom_ops='all' (default "
                "on AUTO_ENABLE_CUSTOM_OPS machines) to enable the fusions."
            )

        # torch_npu elide pass (stage3 U7): the global-slot install is shadowed by
        # vLLM's per-compile config.patch, so chain it explicitly. Degrade to a
        # warning when the TN-side shape is unavailable (old wheel).
        try:
            from torch_npu._inductor.triton_experimental import config as tn_config

            elide_enabled = getattr(tn_config, "elide_int_float_int", False)
        except Exception:  # noqa: BLE001
            elide_enabled = False
        if elide_enabled:
            try:
                from torch_npu._inductor.triton_experimental.fx_passes import (
                    _elide_int_float_int_roundtrip_pass as elide_fn,
                )

                self.passes.append(AscendElideIntFloatIntPass(config, elide_fn))
            except Exception:  # noqa: BLE001
                logger.warning(
                    "torch_npu elide_int_float_int pass found enabled but could not be "
                    "wrapped onto the inductor track; skipping it (no functional impact "
                    "beyond the optimization itself)."
                )
