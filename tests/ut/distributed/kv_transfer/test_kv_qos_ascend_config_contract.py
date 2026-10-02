# SPDX-License-Identifier: Apache-2.0
"""CPU regression for the real AscendConfig factory/schema boundary.

Only logging, hardware discovery, and unrelated hardware derivations are mocked.
CLI parsing, QoS policy application, init_ascend_config, and Pydantic schemas
execute their actual source. This is not an engine startup or A5 test.
Run directly with Python as well as through pytest.
"""

import argparse
import contextlib
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AscendQosConfigContract(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules))
        for name in ("vllm", "vllm.utils", "vllm_ascend", "vllm_ascend.device"):
            module = ModuleType(name)
            module.__path__ = [str(ROOT / "vllm_ascend")] if name == "vllm_ascend" else []
            sys.modules[name] = module
        logger: Any = ModuleType("vllm.logger")
        logger.logger = MagicMock()
        sys.modules[logger.__name__] = logger
        math: Any = ModuleType("vllm.utils.math_utils")
        math.cdiv = lambda a, b: -(a // -b)
        sys.modules[math.__name__] = math
        hardware: Any = ModuleType("vllm_ascend.device.hardware_profile")
        hardware.HardwareCapability = NS(NPUGRAPH_EX="fixture")
        hardware.get_current_hardware_profile = lambda: NS(supports=lambda cap: False)
        sys.modules[hardware.__name__] = hardware
        utils: Any = ModuleType("vllm_ascend.utils")
        utils.clear_enable_sp = lambda: None
        sys.modules[utils.__name__] = utils

        load("vllm_ascend.config_utils", ROOT / "vllm_ascend/config_utils.py")
        self.config = load("vllm_ascend.ascend_config", ROOT / "vllm_ascend/ascend_config.py")
        self.ai = load("vllm_ascend.ai_qos", ROOT / "vllm_ascend/ai_qos.py")
        self.policy = load(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos",
            ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/qos.py",
        )
        request = load(
            "vllm_ascend.patch.platform.patch_ai_qos_request",
            ROOT / "vllm_ascend/patch/platform/patch_ai_qos_request.py",
        )
        self.stack.enter_context(patch.object(request, "install_request_validation"))
        self.derive = self.stack.enter_context(patch.object(self.config.AscendConfig, "derive_and_validate"))
        self.stack.enter_context(patch.object(self.config.FinegrainedTPConfig, "_validate_preconditions"))
        self.stack.enter_context(patch.object(self.config.XliteGraphConfig, "_validate_preconditions"))
        self.stack.enter_context(patch.object(self.config, "_is_ascend_config_initialized", return_value=False))
        # Test the normal strict schema, regardless of optional Omni installs.
        find_spec = importlib.util.find_spec
        self.stack.enter_context(
            patch.object(
                importlib.util,
                "find_spec",
                side_effect=lambda name: None if name == "vllm_omni" else find_spec(name),
            )
        )

    def config_for(self, additional, connector="AscendStoreConnector"):
        extra = {"backend": "mooncake"} if connector == "AscendStoreConnector" else {}
        return NS(
            additional_config=additional,
            kv_transfer_config=NS(kv_connector=connector, kv_connector_extra_config=extra),
        )

    def factory(self, config):
        original = copy.deepcopy(config.additional_config)
        self.ai.apply_config(config)
        result = self.config.init_ascend_config(config)
        self.assertEqual(config.additional_config, original)
        self.assertNotIn("ai_qos", vars(result))
        self.assertFalse(result.enable_cpu_binding)
        return result

    def test_cli_fixed_levels_reach_factory_and_policy(self):
        for connector in ("AscendStoreConnector", "MooncakeConnectorV1", "MooncakeLayerwiseConnector"):
            for name, qos in (("low", 0), ("medium", 3), ("high", 7)):
                for reverse in (False, True):
                    with self.subTest(connector=connector, level=name, reverse=reverse):
                        parser = argparse.ArgumentParser()
                        parser.add_argument("--additional-config", type=json.loads, default={})
                        self.ai.register_cli(parser)
                        cli = [
                            "--ai-qos",
                            json.dumps({"kv_transfer": {"request_priority": False, "default_priority": name}}),
                        ]
                        other = ["--additional-config", '{"enable_cpu_binding":false}']
                        args = parser.parse_args(other + cli if reverse else cli + other)
                        config = self.config_for(args.additional_config, connector)
                        self.factory(config)
                        policy = self.policy.KvQosPolicy.from_config(
                            config.kv_transfer_config.kv_connector_extra_config["kv_qos"]
                        )
                        self.assertEqual(policy.resolve_priority({"kv_priority": "high"}), qos)
                        self.assertEqual(policy.select(qos), qos)

    def test_request_override_and_serialization_preserved(self):
        additional = {"enable_cpu_binding": False, "ai_qos": {"kv_transfer": {"request_priority": True}}}
        config = self.config_for(additional)
        self.factory(config)
        restored = self.config_for(json.loads(json.dumps(config.additional_config)))
        self.factory(restored)
        policy = self.policy.KvQosPolicy.from_config(restored.kv_transfer_config.kv_connector_extra_config["kv_qos"])
        self.assertEqual(policy.resolve_priority({"kv_priority": "high"}), 7)
        self.assertEqual(policy.resolve_priority({}), 0)
        self.assertEqual(additional, restored.additional_config)

    def test_disabled_qos_does_not_create_policy(self):
        config = self.config_for({"enable_cpu_binding": False, "ai_qos": {"kv_transfer": {"enabled": False}}})
        self.factory(config)
        self.assertNotIn("kv_qos", config.kv_transfer_config.kv_connector_extra_config)

    def test_no_qos_config_remains_supported(self):
        config = self.config_for({"enable_cpu_binding": False})
        self.factory(config)
        self.assertNotIn("kv_qos", config.kv_transfer_config.kv_connector_extra_config)

    def test_unknown_ascend_key_still_rejected(self):
        config = self.config_for({"enable_cpu_binding": False, "ai_qoss": {}})
        with self.assertRaisesRegex(ValueError, "ai_qoss"):
            self.factory(config)
        self.derive.assert_not_called()

    def test_invalid_qos_still_rejected(self):
        invalid: dict[str, Any]
        for invalid in (
            {"kv_transfer": {"default_priority": "urgent"}},
            {"kv_transfer": {"request_priority": "true"}},
            {"kv_transfer": {"unknown": 1}},
            {"collective_communication": {}},
        ):
            with self.subTest(config=invalid):
                config = self.config_for({"enable_cpu_binding": False, "ai_qos": invalid})
                with self.assertRaises(ValueError):
                    self.factory(config)
        self.derive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
