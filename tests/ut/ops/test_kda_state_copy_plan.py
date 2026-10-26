# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops import kda_state_copy_plan as production


class TestKDACompiledLaunchSignatures(TestCase):
    def test_prepared_variants_match_runtime_only_and_constexpr_launchers(self):
        """Cover both launcher ABIs, including specialized runtime strides."""
        for retain_constexpr in (False, True):
            with self.subTest(retain_constexpr=retain_constexpr):
                self._check_launches(retain_constexpr)

    def _check_launches(self, retain_constexpr):
        launches = []
        variants = []
        kernel = MagicMock(pre_run_hooks=[])

        def warmup(*args, **constants):
            signature = dict.fromkeys(
                (
                    "cache_ptr",
                    "packed_ptr",
                    "indices_ptr",
                    "flags_ptr",
                    "cache_rows",
                    "cache_stride_elements",
                    "payload_elements",
                    "index_stride",
                    "flag_stride",
                ),
                "i32",
            )
            expected_constants = ()
            if retain_constexpr:
                # 3.6 also specializes the last two runtime scalars; those are
                # already passed positionally and must not be appended twice.
                signature.update(dict.fromkeys(("index_stride", "flag_stride", *constants), "constexpr"))
                expected_constants = tuple(constants.values())
            index_dtype = args[2].dtype
            to_cache = constants["TO_CACHE"]
            compiled = MagicMock(src=SimpleNamespace(signature=signature))

            def launch(*actual):
                self.assertEqual(len(actual), 9 + len(expected_constants))
                self.assertEqual(actual[2].dtype, index_dtype)
                self.assertEqual(actual[4:9], (7, 24, 24, 1, 0 if to_cache else 1))
                self.assertEqual(actual[9:], expected_constants)
                if to_cache:
                    self.assertIs(actual[3], actual[2])
                else:
                    self.assertEqual(actual[3].dtype, torch.bool)
                launches.append((index_dtype, to_cache))

            compiled.__getitem__.return_value = launch
            variants.append(compiled)
            return compiled

        kernel.__getitem__.return_value = warmup
        state = torch.ones((7, 2, 3, 4), dtype=torch.float32)
        with (
            patch.object(production, "_kda_state_copy_kernel", kernel),
            patch.object(production, "_validate_cache_layout", return_value=(24, 7, 24)),
            patch.object(production, "_configuration", return_value=()),
            patch.object(torch.npu, "device", return_value=nullcontext()),
            patch.object(torch.npu, "current_device", return_value=None),
            patch.object(torch.npu, "synchronize"),
        ):
            plan = production.KDAStateCopyPlan.prepare(state, 5)
            plan.seal()
            self.assertEqual(len(plan._compiled), 4)
            # No signature inspection or JIT entry is permitted after prepare.
            for compiled in variants:
                compiled.src = None
            kernel.__getitem__.side_effect = AssertionError("serving must not reenter JIT")
            for index_dtype in (torch.int32, torch.int64):
                indices = torch.tensor([0, 2, -1, 7, 3], dtype=index_dtype)
                packed = plan.gather(state, indices, None)
                plan.scatter(state, packed, indices)
                empty = indices[:0]
                plan.scatter(state, packed[:0], empty)
                plan.gather(state, empty, None)
        self.assertEqual(
            launches, [(torch.int32, False), (torch.int32, True), (torch.int64, False), (torch.int64, True)]
        )
