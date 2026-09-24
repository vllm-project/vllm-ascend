#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
"""T3-3（stage3 项2）：VA 融合 pass 链回 inductor 轨 AscendPostGradPassManager 的 UT。

设计（02 §2）：configure 按 ``get_ascend_config().ascend_compilation_config.fuse_*``
（默认 True，对齐 legacy）注入三条 pass；uuid 因子化；custom_ops 失配 warn；nge
import guard；match_table 观测。轨关时本类不可达（get_pass_manager_cls 切回
GraphFusionPassManager，stage2 rev B 用例已覆盖切换本身）。

Device note: pass 实例化经 pattern 注册创建 NPU 示例张量——依赖真机的用例带
``skipUnless(torch.npu.is_available())``；纯配置用例（warn/guard/flag-off）保持 CPU 安全。
构图须用 ``make_fx(tracing_mode="symbolic")``：symbolic_trace 无节点 meta，
pattern_matcher 全不命中（T0''-2 发现 1）。
"""
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except Exception:
    _NPU_AVAILABLE = False


def _track_vllm_config(fuse_norm_quant=None, fuse_qknorm_rope=None, fuse_muls_add=None):
    """带 fuse flags 的轨上 VllmConfig（bare + SimpleNamespace model_config）。

    轨经上游正门（compilation_config.backend="inductor"）请求；fuse flags
    仍走 additional_config 袋子（真配置语义，非选择机制）。
    """
    ascend: dict = {}
    for k, v in (
        ("fuse_norm_quant", fuse_norm_quant),
        ("fuse_qknorm_rope", fuse_qknorm_rope),
        ("fuse_muls_add", fuse_muls_add),
    ):
        if v is not None:
            ascend[k] = v
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        from vllm.config import VllmConfig

        vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        enforce_eager=False,
        dtype=torch.bfloat16,
        # _model_uses_w4a4_quant 读 vllm_config.quant_config（真实字段，默认 None）
        hf_text_config=SimpleNamespace(routed_scaling_factor=2.5, rms_norm_eps=1e-6),
    )
    vllm_config.quant_config = None
    vllm_config.compilation_config.backend = "inductor"
    vllm_config.additional_config = {"ascend_compilation_config": ascend}
    vllm_config.device_config.device_type = "npu"
    # 跑真实早 hook：12 个上游 pass flag 钉 False（生产时序——
    # 否则上游 PostGradPassManager.configure 会在 pass_config 融合 flag 为真时
    # 读 model_config.using_transformers_backend()，SimpleNamespace 无此方法）
    from vllm_ascend.platform import NPUPlatform

    NPUPlatform._apply_inductor_track_defaults(vllm_config)
    return vllm_config


class FusionChainTestBase(TestBase):
    def setUp(self):
        super().setUp()
        clear_ascend_config()

    def tearDown(self):
        clear_ascend_config()
        super().tearDown()

    def _configure(self, **fuse_kwargs):
        from vllm_ascend.compilation.ascend_post_grad_pass_manager import (
            AscendPostGradPassManager,
        )

        vllm_config = _track_vllm_config(**fuse_kwargs)
        init_ascend_config(vllm_config)
        manager = AscendPostGradPassManager()
        manager.configure(vllm_config)
        return manager, vllm_config


class TestConfigureInjection(FusionChainTestBase):
    @unittest.skipUnless(_NPU_AVAILABLE, "pass 实例化需 NPU 示例张量")
    def test_default_flags_inject_all_three_passes(self):
        """① 默认 True → 三条 pass 注入（与 legacy GraphFusionPassManager 同条件序）。"""
        from vllm_ascend.compilation.passes.muls_add_pass import MulsAddFusionPass
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormQuantFusionPass,
        )
        from vllm_ascend.compilation.passes.qknorm_rope_fusion_pass import (
            QKNormRopeFusionPass,
        )

        with patch("vllm_ascend.utils.is_310p", return_value=False):
            manager, _ = self._configure()
        types = [type(p) for p in manager.passes]
        self.assertIn(AddRMSNormQuantFusionPass, types)
        self.assertIn(QKNormRopeFusionPass, types)
        self.assertIn(MulsAddFusionPass, types)

    def test_all_flags_off_injects_nothing(self):
        """② 三 flag 显式 False → 无融合 pass 注入（CPU 安全；elide wrapper 不受 flag 门控）。"""
        from vllm_ascend.compilation.passes.muls_add_pass import MulsAddFusionPass
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormQuantFusionPass,
        )
        from vllm_ascend.compilation.passes.qknorm_rope_fusion_pass import (
            QKNormRopeFusionPass,
        )

        with patch("vllm_ascend.utils.is_310p", return_value=False):
            manager, _ = self._configure(
                fuse_norm_quant="false", fuse_qknorm_rope="false", fuse_muls_add="false"
            )
        fusion_types = (AddRMSNormQuantFusionPass, QKNormRopeFusionPass, MulsAddFusionPass)
        self.assertFalse([p for p in manager.passes if isinstance(p, fusion_types)])

    @unittest.skipUnless(_NPU_AVAILABLE, "pass 实例化需 NPU 示例张量")
    def test_310p_skips_norm_quant_and_muls_add(self):
        """① 分支：310P 跳过 norm_quant/muls_add（qknorm 无此条件，与 legacy 一致）。"""
        from vllm_ascend.compilation.passes.muls_add_pass import MulsAddFusionPass
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormQuantFusionPass,
        )

        with patch("vllm_ascend.utils.is_310p", return_value=True):
            manager, _ = self._configure()
        types = [type(p) for p in manager.passes]
        self.assertNotIn(AddRMSNormQuantFusionPass, types)
        self.assertNotIn(MulsAddFusionPass, types)


class TestUuidFactorization(FusionChainTestBase):
    @unittest.skipUnless(_NPU_AVAILABLE, "pass 实例化需 NPU 示例张量")
    def test_config_factors_change_uuid(self):
        """③ uuid 因子化：config 因子（scale 等）并入 hash——同图不同配置不得同 uuid。"""
        from vllm.config.utils import Range

        from vllm_ascend.compilation.passes.muls_add_pass import MulsAddFusionPass

        with patch("vllm_ascend.utils.is_310p", return_value=False):
            base = _track_vllm_config()
            scaled = _track_vllm_config()
            scaled.model_config.hf_text_config.routed_scaling_factor = 7.0
        p1 = MulsAddFusionPass(base)
        p2 = MulsAddFusionPass(scaled)
        self.assertNotEqual(p1.uuid(), p2.uuid(), "scale 因子必须进 uuid（防 FxGraphCache 碰撞）")
        # range 谓词语义（红队 §7.7）：当前三 pass 对任意 range 均 True——固化现状
        self.assertTrue(p1.is_applicable_for_range(Range(0, 16)))

    @unittest.skipUnless(_NPU_AVAILABLE, "pass 实例化需 NPU 示例张量")
    def test_manager_aggregate_uuid_changes_with_flags(self):
        """③ 聚合层面：flag 开/关两份 passes 列表 → PostGradPassManager 聚合 uuid 不同。"""
        with patch("vllm_ascend.utils.is_310p", return_value=False):
            on, _ = self._configure()
            off, _ = self._configure(
                fuse_norm_quant="false", fuse_qknorm_rope="false", fuse_muls_add="false"
            )
        self.assertNotEqual(_aggregate_uuid(on), _aggregate_uuid(off))


def _aggregate_uuid(manager) -> str:
    from vllm.compilation.passes.inductor_pass import InductorPass

    parts = [p.uuid() for p in manager.passes]
    return InductorPass.hash_dict({"passes": parts, "off": not parts})


class TestCustomOpsMismatchWarn(FusionChainTestBase):
    def test_warns_when_fuse_on_and_custom_ops_missing(self):
        """④ fuse 开而 custom_ops 缺所需 op → warn（pattern 将失配，仅 muls_add 存活）。"""
        with (
            patch("vllm_ascend.utils.is_310p", return_value=False),
            patch(
                "vllm_ascend.compilation.ascend_post_grad_pass_manager._required_custom_ops_missing",
                return_value=True,
            ),
            self.assertLogs(
                "vllm_ascend.compilation.ascend_post_grad_pass_manager", level="WARNING"
            ) as logs,
        ):
            self._configure()
        self.assertTrue(any("custom_ops" in msg for msg in logs.output))

    def test_no_warn_when_custom_ops_covers(self):
        """④ custom_ops 正常（['all'] 或含所需 op）→ 不告警不炸。"""
        with (
            patch("vllm_ascend.utils.is_310p", return_value=False),
            patch(
                "vllm_ascend.compilation.ascend_post_grad_pass_manager._required_custom_ops_missing",
                return_value=False,
            ),
        ):
            manager, _ = self._configure()
        self.assertIsNotNone(manager)


class TestNgeImportGuard(FusionChainTestBase):
    def test_missing_nge_does_not_crash_and_warns(self):
        """⑤ npugraph_ex/torchair 双缺 → guard：base_pattern 可导入（nge=None，warn）。"""
        import builtins
        import logging

        import vllm_ascend.compilation.passes.base_pattern as bp

        real_import = builtins.__import__
        # reload 会重建模块级 `_registered_patterns` 去重集合——快照以便恢复，
        # 否则后续用例再注册同 pattern 会撞 torch 全局注册表（Duplicate pattern）。
        saved_pattern_ids = set(bp._registered_patterns)

        def fake_import(name, *a, **kw):
            if name.split(".")[0] in ("npugraph_ex", "torchair"):
                raise ImportError(f"blocked for test: {name}")
            return real_import(name, *a, **kw)

        messages: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        # base_pattern 用的是 `from vllm.logger import logger`（模块级默认 logger），
        # 不是 init_logger(__name__)——handler 直接挂 bp.logger 对象最稳。
        handler = _Capture(level=logging.WARNING)
        bp.logger.addHandler(handler)
        builtins.__import__ = fake_import
        try:
            reloaded = importlib.reload(bp)
            self.assertIsNone(getattr(reloaded, "nge", "missing-sentinel"))
            self.assertTrue(
                any("npugraph_ex" in m or "torchair" in m for m in messages),
                f"expected guard warning, got {messages}",
            )
        finally:
            builtins.__import__ = real_import
            bp.logger.removeHandler(handler)
            importlib.reload(bp)  # 恢复真实模块状态（nge 就位）
            bp._registered_patterns.update(saved_pattern_ids)  # 恢复去重集合


class TestMatchTableObservability(FusionChainTestBase):
    @unittest.skipUnless(_NPU_AVAILABLE, "pattern 注册与构图需 NPU")
    def test_muls_add_hit_records_match_table(self):
        """⑥ make_fx(symbolic) 命中图 → match_table 记录（T0''-2 发现 1 的构图法）。"""
        from torch.fx.experimental.proxy_tensor import make_fx

        from vllm.compilation.passes.vllm_inductor_pass import VllmPatternMatcherPass
        from vllm_ascend.compilation.passes.muls_add_pass import MulsAddFusionPass

        with patch("vllm_ascend.utils.is_310p", return_value=False):
            vllm_config = _track_vllm_config()
            fused = MulsAddFusionPass(vllm_config)
        scale = vllm_config.model_config.hf_text_config.routed_scaling_factor

        def f(x, y):
            return x * scale + y

        x = torch.randn(2, 64, device="npu", dtype=torch.bfloat16)
        y = torch.randn(2, 64, device="npu", dtype=torch.bfloat16)
        gm = make_fx(f, tracing_mode="symbolic")(x, y)
        before = VllmPatternMatcherPass.match_table.get("muls_add_fusion_pass", 0)
        fused(gm.graph)
        gm.recompile()
        self.assertEqual(fused.matched_count, 1)
        self.assertIn("muls_add", gm.code)
        # match_table（worker RPC get_compilation_match_table 的取数通道）应累计命中
        self.assertGreaterEqual(
            VllmPatternMatcherPass.match_table.get("muls_add_fusion_pass", 0), before + 1
        )
