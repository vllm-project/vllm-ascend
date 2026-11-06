# SPDX-License-Identifier: Apache-2.0
"""Portable Worker contracts; execute production method bodies without NPU imports.

Run directly with unittest to bypass the NPU-wide pytest conftest. Full worker
integration tests remain in test_worker_v1.py and test_worker_v2.py.
"""

import ast
import copy
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import NoneType, SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "vllm_ascend/worker/worker.py"


def worker_method(name, **bindings):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))
    method.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    scope = dict(bindings)
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), scope)
    return scope[name]


class WorkerAlignmentContract(unittest.TestCase):
    def test_reload_context_and_failure_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                events = []

                @contextmanager
                def config_context(config, events=events):
                    events.append(config)
                    try:
                        yield
                    finally:
                        events.append("exit")

                runner = MagicMock()

                def reload(events=events, fail=fail, **kwargs):
                    self.assertEqual(events, ["config"])
                    self.assertEqual(kwargs, {"weights_path": "checkpoint"})
                    if fail:
                        raise ValueError("load failed")

                runner.reload_weights.side_effect = reload
                worker = SimpleNamespace(vllm_config="config", model_runner=runner)
                method = worker_method("reload_weights", set_current_vllm_config=config_context)
                if fail:
                    with self.assertRaisesRegex(ValueError, "load failed"):
                        method(worker, weights_path="checkpoint")
                else:
                    method(worker, weights_path="checkpoint")
                self.assertEqual(events, ["config", "exit"])

    def test_pp_device_handles_only(self):
        class Intermediate:
            tensors = {"hidden": "buffer"}
            kv_connector_output = None

        class Output:
            pass

        for v2 in (False, True):
            for sp in (False, True):
                for forward in (False, True):
                    with self.subTest(v2=v2, sp=sp, forward=forward):
                        pp, tp = MagicMock(), object()
                        pp.is_first_rank = False
                        pp.is_last_rank = False
                        pp.irecv_tensor_dict.return_value = ({"hidden": "input"}, [], [])
                        metadata, device, previous = MagicMock(), MagicMock(), MagicMock()
                        pp.isend_tensor_dict.return_value = [metadata, device]
                        runner = MagicMock()
                        runner.execute_model.return_value = Intermediate()
                        worker = SimpleNamespace(
                            log_memory_stats=MagicMock(),
                            _pp_send_work=[previous],
                            use_v2_model_runner=v2,
                            model_runner=runner,
                            profiler=None,
                            vllm_config=SimpleNamespace(
                                parallel_config=SimpleNamespace(distributed_executor_backend="mp")
                            ),
                        )
                        execute = worker_method(
                            "execute_model",
                            get_pp_group=lambda pp=pp: pp,
                            get_tp_group=lambda tp=tp: tp,
                            get_ascend_config=lambda: SimpleNamespace(msmonitor_use_daemon=False),
                            enable_sp=lambda _, sp=sp: sp,
                            AsyncIntermediateTensors=lambda *a, **kw: "received",
                            IntermediateTensors=Intermediate,
                            ModelRunnerOutput=Output,
                            AsyncModelRunnerOutput=Output,
                            NoneType=NoneType,
                        )
                        execute(worker, SimpleNamespace(total_num_scheduled_tokens=int(forward)))
                        previous.wait.assert_called_once_with()
                        self.assertEqual(worker._pp_send_work, [device])
                        metadata.wait.assert_not_called()
                        group = None if v2 and sp else tp
                        pp.isend_tensor_dict.assert_called_once_with(Intermediate.tensors, all_gather_group=group)
                        if forward:
                            pp.irecv_tensor_dict.assert_called_once_with(all_gather_group=group)
                        else:
                            pp.irecv_tensor_dict.assert_not_called()

    def test_execution_decorators(self):
        cls = next(
            n for n in ast.parse(WORKER.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "NPUWorker"
        )
        for name in ("execute_model", "sample_tokens"):
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(
                [ast.unparse(d) for d in node.decorator_list], ["torch.inference_mode()", "with_gpu_sync_check"]
            )


if __name__ == "__main__":
    unittest.main()
