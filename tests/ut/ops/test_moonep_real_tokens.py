# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for real token bounds at the SDK dispatch/combine boundary."""
import ast
import importlib.util
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/moonep_comm.py"


def dispatch_test_modules():
    """Load pure geometry/enum code and a host-only MX scale dtype marker."""
    root = Path(__file__).resolve().parents[3]
    modules = {}
    for key, path in (
        ("vllm_ascend.quantization.quant_type", root / "vllm_ascend/quantization/quant_type.py"),
        ("ascend_moonep.mxfp8_transport", root.parent / "ascend-moonep/ascend_moonep/mxfp8_transport.py"),
    ):
        spec = importlib.util.spec_from_file_location("cpu_" + key.replace(".", "_"), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[key] = module
    modules["vllm_ascend.device.mxfp_compat"] = SimpleNamespace(FLOAT8_E8M0FNU_DTYPE=getattr(torch, "float8_e8m0fnu", torch.uint8))
    return modules


def dispatch_test_quant(modules, transport):
    quant_type = modules["vllm_ascend.quantization.quant_type"].QuantType
    return SimpleNamespace(
        quant_type=quant_type.W4A8MXFP if transport == "mxfp8" else quant_type.NONE,
        mxfp=SimpleNamespace(act_quant_type=torch.float8_e4m3fn, per_token_scale_dtype=torch.uint8),
    )


def load_method(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "TokenDispatcherWithMoonEP")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "_moonep_dispatch_transport"] if name == "token_dispatch" else []
    exec(compile(ast.Module(body=helpers + [method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class DispatchReached(Exception):
    """Stop after observing the SDK call, before any simulated communication."""


class Metadata(SimpleNamespace):
    pass


class RealTokenBoundsTest(unittest.TestCase):
    def check_dispatch(self, transport, old_slot="0"):
        tokens, capacity, hidden, experts, topk = 2, 4, 1024, 4, 2
        calls = []
        pads = {
            "hidden": torch.full((capacity, hidden), float("nan"), dtype=torch.bfloat16),
            "route": torch.full((capacity, topk), float("nan")),
            "topk": torch.full((capacity, topk), 999, dtype=torch.int32),
            "combine_route": torch.zeros(capacity, 8),
        }

        class Buffer:
            def _require_ctx(self):
                return {"B": 2}

            def dispatch(self, *args, **kwargs):
                calls.append(kwargs)
                raise DispatchReached

            def dispatch_mxfp8(self, *args, **kwargs):
                calls.append(kwargs)
                raise DispatchReached

        buffer = Buffer()
        namespace = {
            "torch": torch, "time": time, "os": os,
            "_EXTRA_CTX": SimpleNamespace(capturing=False),
            "get_moonep_tokens_capacity": lambda: capacity,
            "_select_buffer_for_step": lambda *args: (buffer, capacity),
            "_pads_for": lambda *args: pads,
            "_scratch_for": lambda *args: {"tpe_compat": torch.zeros(experts, dtype=torch.int32)},
            "_MOONEP_RUNTIME": SimpleNamespace(_buffers={(experts, topk, hidden): {capacity: buffer}}),
        }
        dispatch = load_method("token_dispatch", namespace)
        dispatcher = SimpleNamespace(
            num_experts=experts, top_k=topk, _logged_transports=set(),
            _tstats=lambda *args: None,
        )
        modules = dispatch_test_modules()
        inputs = SimpleNamespace(
            hidden_states=torch.ones(tokens, hidden, dtype=torch.bfloat16),
            topk_weights=torch.full((tokens, topk), 0.5),
            topk_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            quant=dispatch_test_quant(modules, transport),
        )
        # Old launchers can still export these names. They must not change the
        # actual token bound or make the dispatcher populate capacity-tail rows.
        environment = {"MOONEP_REAL_T": "0", "MOONEP_PAD_ROTATE": "1",
                       "MOONEP_PAD_EXPERT0": "1", "MOONEP_PAD_CUT": "1",
                       "MOONEP_SLOT_MODE": old_slot}
        with patch.dict(os.environ, environment), patch.dict(sys.modules, modules):
            with self.assertRaises(DispatchReached):
                dispatch(dispatcher, inputs)
        self.assertEqual(calls[0]["num_tokens"], tokens)
        self.assertTrue(torch.equal(pads["topk"][:tokens], inputs.topk_ids))
        self.assertTrue(torch.all(pads["topk"][tokens:] == 999))
        self.assertTrue(torch.isnan(pads["hidden"][tokens:]).all())

    def test_bf16_dispatch_uses_real_tokens_with_legacy_slot_zero(self):
        self.check_dispatch("bf16", "0")

    def test_bf16_slot_dispatch_uses_real_tokens(self):
        self.check_dispatch("bf16", "1")

    def test_mxfp8_slot_dispatch_uses_real_tokens(self):
        self.check_dispatch("mxfp8", "0")

    def test_slot_combine_uses_real_tokens_and_delegates_nan_tail(self):
        tokens, capacity, hidden, rows = 2, 4, 16, 8
        calls = []

        class Buffer:
            def _require_ctx(self):
                return {"NvS": rows, "S": capacity, "H": hidden}

            def combine_weighted(self, **kwargs):
                calls.append(kwargs)
                return torch.ones(capacity, hidden, dtype=torch.bfloat16), None, None

        namespace = {
            "torch": torch, "time": time, "MoonEPCombineMetadata": Metadata,
        }
        combine = load_method("token_combine", namespace)
        metadata = Metadata(
            num_tokens=tokens, hidden_shape=(tokens, hidden), source_topk_weights_sk=torch.ones(capacity, 8),
            w_trim=torch.ones(rows), buffer=Buffer(), tail_masked=True,
            group_list=torch.tensor([2, 1], dtype=torch.int64), plan=object(),
        )
        activations = torch.ones(rows, hidden, dtype=torch.bfloat16)
        activations[3:].fill_(float("nan"))
        with patch.dict(os.environ, {"MOONEP_REAL_T": "0"}):
            result = combine(SimpleNamespace(_tstats=lambda *args: None), activations, metadata)
        self.assertEqual(calls[0]["num_tokens"], tokens)
        self.assertEqual(tuple(result.shape), (tokens, hidden))
        self.assertIs(calls[0]["hidden_nvsh"], activations)
        self.assertIs(calls[0]["row_counts"], metadata.group_list)
        self.assertTrue(calls[0]["fused_publish"])
        self.assertTrue(torch.isnan(activations[3:]).all())


if __name__ == "__main__":
    unittest.main()
