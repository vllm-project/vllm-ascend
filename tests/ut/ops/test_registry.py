#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from unittest import mock

from tests.ut.base import TestBase
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


class TestOpsRegistry(TestBase):
    @mock.patch("vllm_ascend.ops.registry.CustomOp")
    def test_register_custom_op(self, mock_customop):
        import vllm.model_executor.custom_op as custom_op_module

        def fake_register_oot(_decorated_op_cls=None, name=None):
            assert name not in custom_op_module.op_registry_oot, f"Duplicate op name: {name}"
            custom_op_module.op_registry_oot[name] = _decorated_op_cls

        mock_customop.register_oot.side_effect = fake_register_oot
        from vllm_ascend.ops.registry import register_custom_op

        with mock.patch.dict(custom_op_module.op_registry_oot, clear=True):
            # downstream custom ops are written as-is (no catalog lookup)
            register_custom_op("OmniCustomOp", int)
            self.assertEqual(mock_customop.register_oot.call_count, 1)
            self.assertIs(custom_op_module.op_registry_oot["OmniCustomOp"], int)

            # registering an already-registered op fails
            with self.assertRaises(AssertionError):
                register_custom_op("OmniCustomOp", str)
            self.assertEqual(mock_customop.register_oot.call_count, 2)
            self.assertIs(custom_op_module.op_registry_oot["OmniCustomOp"], int)

    @mock.patch("vllm_ascend.ops.registry.CustomOp")
    def test_register_custom_ops_exclude(self, mock_customop):
        """exclude= drops names from the full-catalog registration."""
        import vllm.model_executor.custom_op as custom_op_module

        mock_customop.register_oot.side_effect = lambda _decorated_op_cls=None, name=None: (
            custom_op_module.op_registry_oot.__setitem__(name, _decorated_op_cls)
        )
        from vllm_ascend.ops import registry as ops_registry

        with (
            mock.patch(
                "vllm_ascend.ops.registry.get_current_hardware_profile",
                return_value=get_hardware_profile(AscendDeviceType.A2),
            ),
            mock.patch.dict(custom_op_module.op_registry_oot, clear=True),
        ):
            ops_registry.register_custom_ops(exclude={"GateLinear"})
            self.assertEqual(
                mock_customop.register_oot.call_count,
                len(ops_registry._get_ops_base()) - 1,
            )
            self.assertNotIn("GateLinear", custom_op_module.op_registry_oot)

    @mock.patch("vllm_ascend.ops.registry.CustomOp")
    def test_register_all_custom_ops(self, mock_customop):
        import vllm.model_executor.custom_op as custom_op_module

        mock_customop.register_oot.side_effect = lambda _decorated_op_cls=None, name=None: (
            custom_op_module.op_registry_oot.__setitem__(name, _decorated_op_cls)
        )
        from vllm_ascend.ops import registry as ops_registry

        with (
            mock.patch(
                "vllm_ascend.ops.registry.get_current_hardware_profile",
                return_value=get_hardware_profile(AscendDeviceType.A2),
            ),
            mock.patch("vllm_ascend.ops.registry._registered_all_custom_ops", False),
            mock.patch.dict(custom_op_module.op_registry_oot, clear=True),
        ):
            expected_ops = len(ops_registry._get_ops_base()) - 1
            ops_registry.register_all_custom_ops()
            self.assertEqual(mock_customop.register_oot.call_count, expected_ops)

            # ascend custom op is already registered
            ops_registry.register_all_custom_ops()
            self.assertEqual(mock_customop.register_oot.call_count, expected_ops)

    def test_ascend_custom_ops_310p_overrides_base(self):
        """On 310P the merged catalog lets 310P variants win over base ops."""
        from vllm_ascend.ops import registry as ops_registry

        ops_base = ops_registry._get_ops_base()
        ops_310p = ops_registry._get_ops_310p()

        with mock.patch(
            "vllm_ascend.ops.registry.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType._310P),
        ):
            catalog = ops_registry.ascend_custom_ops()
            # 310P variants replace the base classes of the same name
            for name, op_cls in ops_310p.items():
                self.assertIs(catalog[name], op_cls)
            # names without a 310P variant keep the base class
            for name, op_cls in ops_base.items():
                if name not in ops_310p:
                    self.assertIs(catalog[name], op_cls)
