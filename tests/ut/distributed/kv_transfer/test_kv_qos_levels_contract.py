# SPDX-License-Identifier: Apache-2.0
"""CPU contracts. No test in this file is evidence of A5 execution."""

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BUNDLE))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def payload():
    ai = module(BUNDLE / "vllm_ascend/ai_qos.py", "vllm_ascend.ai_qos")
    name = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos"
    policy = module(BUNDLE / Path(*name.split(".")).with_suffix(".py"), name)
    request = module(
        BUNDLE / "vllm_ascend/patch/platform/patch_ai_qos_request.py", "vllm_ascend.patch.platform.patch_ai_qos_request"
    )
    return ai, policy, request


class LevelContractTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.ai, self.policy, self.request = payload()

    def config(self, value=None, connector="AscendStoreConnector"):
        extra = {
            "backend": "mooncake",
            "kv_qos": {"resource_config": {"comm_resource_config.protocol_desc": ["ub_ctp:device"]}},
        }
        return NS(
            additional_config={"ai_qos": {"kv_transfer": value or {"request_priority": True}}},
            kv_transfer_config=NS(
                kv_connector=connector,
                kv_connector_extra_config=extra,
                get_from_extra_config=lambda key, default: extra.get(key, default),
            ),
        )

    def apply_config(self, config):
        with patch.object(self.request, "install_request_validation"):
            self.ai.apply_config(config)
        return self.policy.KvQosPolicy.from_config(config.kv_transfer_config.kv_connector_extra_config["kv_qos"])

    def test_all_case_variants_monotonic_end_to_end_policy(self):
        for connector in ("AscendStoreConnector", "MooncakeConnectorV1", "MooncakeLayerwiseConnector"):
            policy = self.apply_config(self.config(connector=connector))
            for value, expected in (
                ("low", 0),
                ("LOW", 0),
                ("LoW", 0),
                ("medium", 3),
                ("MEDIUM", 3),
                ("mEdIuM", 3),
                ("high", 7),
                ("HIGH", 7),
                ("HiGh", 7),
            ):
                with self.subTest(connector=connector, value=value):
                    actual = policy.request_priority(NS(kv_transfer_params={"kv_priority": value}))
                    self.assertEqual(actual, expected)
                    self.assertEqual(policy.select(actual), expected)
                    # P serializes the exact canonical value; D accepts it unchanged.
                    wire = json.loads(json.dumps({"kv_priority": actual}))
                    self.assertEqual(policy.request_priority(NS(kv_transfer_params=wire)), expected)

    def test_default_low_and_static_medium_when_request_override_disabled(self):
        policy = self.apply_config(self.config())
        self.assertEqual(policy.request_priority(NS(kv_transfer_params=None)), 0)
        policy = self.apply_config(self.config({"request_priority": False, "default_priority": "MEDIUM"}))
        self.assertEqual(policy.request_priority(NS(kv_transfer_params={"kv_priority": "HIGH"})), 3)

    def test_invalid_labels_and_inversions_rejected(self):
        policy = self.apply_config(self.config())
        value: object
        for value in (None, True, 1, 2, 8, -1, 7.0, "7", "urgent", {}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.request_priority(NS(kv_transfer_params={"kv_priority": value}))
        with self.assertRaisesRegex(ValueError, "identity mapping"):
            self.policy.KvQosPolicy.from_config({"priority_to_qos": {"0": 7, "3": 3, "7": 0}, "level_names": True})

    def test_logging_default_off_and_opt_in(self):
        policy = self.apply_config(self.config())
        with self.assertNoLogs(policy.__class__.__module__, level="INFO"):
            policy.request_priority(NS(request_id="r", kv_transfer_params={"kv_priority": "high"}))
        policy = self.apply_config(self.config({"request_priority": True, "log": True}))
        with self.assertLogs(policy.__class__.__module__, level="INFO") as recorded:
            policy.request_priority(NS(request_id="r", kv_transfer_params={"kv_priority": "HiGh"}))
        self.assertIn("supplied='HiGh' priority=7 lane_qos=7", recorded.output[0])

    def test_legacy_mode_remains_available_but_not_combined(self):
        old = {"priority_to_qos": {"0": 0, "1": 3, "2": 7}}
        policy = self.policy.KvQosPolicy.from_config(old)
        self.assertEqual(policy.request_priority(NS(kv_transfer_params={"kv_priority": 2})), 2)
        self.assertEqual(policy.select(2), 7)
        config = self.config()
        config.kv_transfer_config.kv_connector_extra_config["kv_qos"] = old
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.apply_config(config)

    def test_config_serialization_repeat_and_protocol_preservation(self):
        config = self.config()
        self.apply_config(config)
        self.apply_config(config)  # VllmConfig may be validated more than once.
        encoded = json.loads(json.dumps(config.kv_transfer_config.kv_connector_extra_config))
        self.assertEqual(encoded["kv_qos"]["resource_config"]["comm_resource_config.protocol_desc"], ["ub_ctp:device"])
        self.assertEqual(encoded["kv_qos"]["priority_to_qos"], {"0": 0, "3": 3, "7": 7})

    def test_unimplemented_domains_and_invalid_config_fail_explicitly(self):
        value: dict[str, Any]
        for value in (
            {"op_submit": {"overrides": {"matmul": "medium"}}},
            {"collective": {}},
            {"kv_transfer": {"request_priority": "true"}},
            {"kv_transfer": {"log": 1}},
            {"kv_transfer": {"priority_to_qos": {}}},
            {"other": {}},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.ai.validate_ai_qos(value)
        with self.assertRaises(ValueError):
            self.apply_config(self.config(connector="OtherConnector"))

    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--additional-config", type=json.loads, default={})
        self.ai.register_cli(parser)
        return parser

    def test_cli_both_option_orders_no_state_leak(self):
        args = ["--ai-qos", '{"kv_transfer":{"request_priority":true}}']
        additional = ["--additional-config", '{"other_setting":12}']
        for command in (args + additional, additional + args):
            parsed = self.parser().parse_args(command)
            self.assertEqual(parsed.additional_config["other_setting"], 12)
            self.assertTrue(parsed.additional_config["ai_qos"]["kv_transfer"]["request_priority"])
        parser = self.parser()
        self.ai.register_cli(parser)
        parser.parse_args(args)
        self.assertEqual(parser.parse_args([]).additional_config, {})

    def test_cli_repeated_or_conflicting_entries_fail(self):
        args = ["--ai-qos", '{"kv_transfer":{"request_priority":true}}']
        for command in (
            args + args,
            args + ["--additional-config", '{"ai_qos":{}}'],
            ["--additional-config", '{"ai_qos":{}}'] + args,
            ["--ai-qos", '{"kv_transfer":{},"kv_transfer":{}}'],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parser().parse_args(command)

    def test_input_validation_prevents_bad_label_before_original_processing(self):
        # Load the installed/pinned upstream exception without initializing the
        # engine. VLLMValidationError no longer inherits from plain ValueError.
        exceptions = sys.modules.get("vllm.exceptions")
        if exceptions is None:
            spec = importlib.util.find_spec("vllm")
            path = (
                Path(spec.origin).with_name("exceptions.py")
                if spec and spec.origin
                else BUNDLE.parent / "vllm/vllm/exceptions.py"
            )
            exceptions = module(path, "vllm.exceptions")
        seen = []

        class Input:
            vllm_config: NS

            def _validate_params(self, params, supported_tasks):
                seen.append((params, supported_tasks))
                return "original"

        fake: Any = ModuleType("vllm.v1.engine.input_processor")
        fake.InputProcessor = Input
        sys.modules[fake.__name__] = fake
        self.request.install_request_validation()
        patched = Input._validate_params
        self.request.install_request_validation()
        self.assertIs(patched, Input._validate_params)
        config = self.config()
        self.apply_config(config)
        instance = Input()
        instance.vllm_config = config
        valid = NS(extra_args={"kv_transfer_params": {"kv_priority": "HIGH"}})
        self.assertEqual(instance._validate_params(valid, ("generate",)), "original")
        bad = NS(extra_args={"kv_transfer_params": {"kv_priority": "wrong"}})
        with self.assertRaises(exceptions.VLLMValidationError):
            instance._validate_params(bad, ("generate",))
        self.assertEqual(len(seen), 1)
        self.assertEqual(valid.extra_args["kv_transfer_params"]["kv_priority"], "HIGH")


if __name__ == "__main__":
    unittest.main()
