# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only DP transport tests, also runnable directly with unittest.

These execute real methods with mocked connections and LLMs; they do not exercise
real subprocesses, NPU execution, or RequestOutput pickling across a process pipe.
"""

import __future__

import ast
import contextlib
import sys
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_runner():
    """Execute the real transport methods without importing the NPU E2E stack."""
    path = Path(__file__).resolve().parents[1] / "e2e/conftest.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_split_data_parallel_indices",
        "_slice_optional_inputs",
        "_slice_list_inputs",
        "_merge_data_parallel_results",
        "_run_vllm_runner_dp_worker",
        "VllmRunner",
        "DPVllmRunner",
    }
    methods = {
        "VllmRunner": {"get_inputs", "_finalize_generate_outputs"},
        "DPVllmRunner": {"_dispatch_prompt_command", "generate_raw", "collective_rpc"},
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.body = [method for method in node.body if getattr(method, "name", None) in methods[node.name]]
    namespace = {
        "TextPrompt": dict,
        "contextlib": contextlib,
        "traceback": traceback,
        "os": SimpleNamespace(environ={}),
        "LLM": Mock(),
        "clear_ascend_config": Mock(),
        "cleanup_dist_env_and_memory": Mock(),
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def _output(index):
    return SimpleNamespace(
        request_id=str(index),
        prompt=f"prompt {index}",
        prompt_token_ids=[10 + index, 20],
        prompt_logprobs=[None, {20: SimpleNamespace(logprob=-0.2)}],
        finished=True,
        outputs=[
            SimpleNamespace(
                text=f"answer {index}",
                token_ids=[42, 43],
                logprobs=[{42: SimpleNamespace(logprob=-1.0)}, {43: SimpleNamespace(logprob=-2.0)}],
                finish_reason="length",
                stop_reason=None,
            )
        ],
    )


class TestDPGenerateRaw(unittest.TestCase):
    def setUp(self):
        self.namespace = _load_runner()
        self.runner = self.namespace["DPVllmRunner"]()
        self.runner._dp_size = 4
        self.runner._dp_request_timeout = 7.0
        self.runner._stop_data_parallel_workers = Mock()
        self.connections = [Mock() for _ in range(4)]
        self.runner._dp_parent_conns = self.connections
        self.sampling_params = object()

    def _responses(self, indices, results):
        for conn, shard_indices, shard_results in zip(self.connections, indices, results):
            conn.poll.return_value = True
            conn.recv.return_value = {"status": "ok", "indices": shard_indices, "result": shard_results}

    def test_shard_order_and_complete_outputs(self):
        outputs = [_output(index) for index in range(5)]
        self._responses([[0, 1], [2], [3], [4]], [outputs[:2], outputs[2:3], outputs[3:4], outputs[4:]])
        prompts = [f"prompt {index}" for index in range(5)]
        images = [f"image {index}" for index in range(5)]
        videos = [f"video {index}" for index in range(5)]
        audios = [f"audio {index}" for index in range(5)]
        result = self.runner.generate_raw(
            prompts, self.sampling_params, images=images, videos=videos, audios=audios, use_tqdm=False
        )
        self.assertEqual(len(result), 5)
        for actual, expected in zip(result, outputs):
            # The transport merge must return each complete object unchanged.
            self.assertIs(actual, expected)
        for conn, expected_indices in zip(self.connections, [[0, 1], [2], [3], [4]]):
            request = conn.send.call_args.args[0]
            self.assertEqual(request["command"], "generate_raw")
            self.assertEqual(request["indices"], expected_indices)
            self.assertIs(request["sampling_params"], self.sampling_params)
            self.assertEqual(request["kwargs"], {"use_tqdm": False})
            for item, index in zip(request["inputs"], expected_indices):
                self.assertEqual(item["prompt"], prompts[index])
                self.assertEqual(
                    item["multi_modal_data"], {"image": images[index], "video": videos[index], "audio": audios[index]}
                )
            conn.poll.assert_called_once_with(7.0)

    def test_prompt_types_and_padding_empty_ranks(self):
        tensor = object()  # get_inputs treats a non-string/non-list prompt as embeddings.
        for prompt, key in [("text", "prompt"), ([11, 12], "prompt_token_ids"), (tensor, "prompt_embeds")]:
            with self.subTest(key=key):
                output = _output(0)
                padding = _output(99)
                self._responses([[0], [], [], []], [[output], [padding], [padding], [padding]])
                self.assertEqual(self.runner.generate_raw([prompt], self.sampling_params), [output])
                for rank, conn in enumerate(self.connections):
                    request = conn.send.call_args.args[0]
                    self.assertEqual(request["indices"], [0] if rank == 0 else [])
                    self.assertIs(request["inputs"][0][key], prompt)

    def test_empty_prompts_do_not_dispatch(self):
        self.assertEqual(self.runner.generate_raw([], self.sampling_params), [])
        for conn in self.connections:
            conn.send.assert_not_called()

    def test_worker_errors_and_timeouts_stop_workers(self):
        for failure in ("error", "timeout", "disconnect"):
            with self.subTest(failure=failure):
                self.setUp()
                self._responses([[0], [1], [2], [3]], [[_output(index)] for index in range(4)])
                failing_conn = self.connections[2]
                if failure == "error":
                    failing_conn.recv.return_value = {"status": "error", "traceback": "kernel failed"}
                    error, message = RuntimeError, "worker 2.*generate_raw"
                elif failure == "timeout":
                    failing_conn.poll.return_value = False
                    error, message = TimeoutError, "worker 2.*generate_raw"
                else:
                    failing_conn.recv.side_effect = EOFError("worker disconnected")
                    error, message = EOFError, "worker disconnected"
                with self.assertRaisesRegex(error, message):
                    self.runner.generate_raw(["a", "b", "c", "d"], self.sampling_params)
                self.runner._stop_data_parallel_workers.assert_called_once()

    def test_incomplete_outputs_fail(self):
        for indices, results, message in [
            ([[0], [1], [2], [3]], [[_output(0)], [_output(1)], [], [_output(3)]], "Mismatched result count"),
            ([[0], [1], [], [3]], [[_output(0)], [_output(1)], [], [_output(3)]], "Some data parallel results"),
        ]:
            with self.subTest(message=message):
                self._responses(indices, results)
                with self.assertRaisesRegex(RuntimeError, message):
                    self.runner.generate_raw(["a", "b", "c", "d"], self.sampling_params)

    def test_collective_rpc_preserves_rank_order_and_kwargs(self):
        results = [[{"dp_rank": rank, "tp_rank": tp_rank} for tp_rank in range(2)] for rank in range(4)]
        self._responses([[], [], [], []], results)
        calls = Mock()
        for rank, conn in enumerate(self.connections):
            calls.attach_mock(conn, f"rank_{rank}")
        kwargs = {"timeout": 3.0, "args": ("eager",), "kwargs": {"reset": True}}
        actual = self.runner.collective_rpc("check_graph_replay", **kwargs)
        self.assertEqual(actual, results)
        for rank, conn in enumerate(self.connections):
            self.assertIs(actual[rank], results[rank])
            conn.send.assert_called_once_with(
                {"command": "collective_rpc", "method": "check_graph_replay", "kwargs": kwargs, "indices": []}
            )
            conn.poll.assert_called_once_with(7.0)
        # Every DP worker must receive its command before waiting on any response.
        self.assertEqual([call[0] for call in calls.mock_calls[:4]], [f"rank_{rank}.send" for rank in range(4)])
        self.runner._stop_data_parallel_workers.assert_not_called()

    def test_collective_rpc_failures_stop_workers(self):
        for rank in range(4):
            for failure in ("send", "error", "timeout", "disconnect"):
                with self.subTest(rank=rank, failure=failure):
                    self.setUp()
                    self._responses([[], [], [], []], [[index] for index in range(4)])
                    failing_conn = self.connections[rank]
                    if failure == "send":
                        failing_conn.send.side_effect = BrokenPipeError("worker pipe closed")
                        error, message = BrokenPipeError, "worker pipe closed"
                    elif failure == "error":
                        failing_conn.recv.return_value = {"status": "error", "traceback": "worker check failed"}
                        error, message = RuntimeError, f"worker {rank}.*check_graph_replay"
                    elif failure == "timeout":
                        failing_conn.poll.return_value = False
                        error, message = TimeoutError, f"worker {rank}.*check_graph_replay"
                    else:
                        failing_conn.recv.side_effect = EOFError("worker disconnected")
                        error, message = EOFError, "worker disconnected"
                    with self.assertRaisesRegex(error, message):
                        self.runner.collective_rpc("check_graph_replay", kwargs={"reset": True})
                    self.runner._stop_data_parallel_workers.assert_called_once()

    def test_worker_collective_rpc_dispatch(self):
        results = [{"tp_rank": 0}, {"tp_rank": 1}]
        llm = self.namespace["LLM"].return_value
        llm.collective_rpc.return_value = results
        conn = Mock()
        kwargs = {"timeout": 3.0, "args": ("eager",), "kwargs": {"reset": True}}
        conn.recv.side_effect = [
            {"command": "collective_rpc", "method": "check_graph_replay", "kwargs": kwargs, "indices": []},
            {"command": "shutdown"},
        ]
        with patch.dict(sys.modules, {"torch": SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 4))}):
            self.namespace["_run_vllm_runner_dp_worker"](conn, {}, 2, 4, 12345)
        llm.collective_rpc.assert_called_once_with("check_graph_replay", **kwargs)
        response = conn.send.call_args_list[1].args[0]
        self.assertEqual(response, {"status": "ok", "rank": 2, "indices": [], "result": results})
        self.assertIs(response["result"], results)
        self.namespace["clear_ascend_config"].assert_called_once()
        self.namespace["cleanup_dist_env_and_memory"].assert_called_once()
        conn.close.assert_called_once()

    def test_worker_generate_exception_reports_traceback_and_cleans_up(self):
        llm = self.namespace["LLM"].return_value
        llm.generate.side_effect = ValueError("injected generation failure")
        conn = Mock()
        conn.recv.return_value = {
            "command": "generate_raw",
            "inputs": [{"prompt_token_ids": [10, 20]}],
            "sampling_params": self.sampling_params,
            "kwargs": {"use_tqdm": False},
            "indices": [2],
        }
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 4))}),
            self.assertRaisesRegex(ValueError, "injected generation failure"),
        ):
            self.namespace["_run_vllm_runner_dp_worker"](conn, {}, 2, 4, 12345)
        self.assertEqual(conn.send.call_args_list[0].args[0], {"status": "ready", "rank": 2})
        self.assertEqual(conn.send.call_count, 2)
        response = conn.send.call_args_list[1].args[0]
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["rank"], 2)
        self.assertIn("ValueError: injected generation failure", response["traceback"])
        self.assertIn("_run_vllm_runner_dp_worker", response["traceback"])
        self.namespace["clear_ascend_config"].assert_called_once()
        self.namespace["cleanup_dist_env_and_memory"].assert_called_once()
        conn.close.assert_called_once()

    def test_worker_raw_dispatch_and_legacy_generate(self):
        outputs = [_output(0)]
        llm = self.namespace["LLM"].return_value
        llm.generate.return_value = outputs
        conn = Mock()
        payload = {
            "inputs": [{"prompt_token_ids": [10, 20]}],
            "sampling_params": self.sampling_params,
            "kwargs": {"use_tqdm": False},
            "indices": [0],
        }
        conn.recv.side_effect = [
            {"command": "generate_raw", **payload},
            {"command": "generate", **payload},
            {"command": "shutdown"},
        ]
        with patch.dict(sys.modules, {"torch": SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 4))}):
            self.namespace["_run_vllm_runner_dp_worker"](conn, {}, 0, 4, 12345)
        raw = conn.send.call_args_list[1].args[0]
        self.assertEqual(raw["status"], "ok")
        self.assertEqual(raw["indices"], [0])
        self.assertIs(raw["result"], outputs)
        self.assertEqual(conn.send.call_args_list[2].args[0]["result"], [([[10, 20, 42, 43]], ["prompt 0answer 0"])])
        llm.generate.assert_called_with(payload["inputs"], sampling_params=self.sampling_params, use_tqdm=False)
        self.namespace["clear_ascend_config"].assert_called_once()
        self.namespace["cleanup_dist_env_and_memory"].assert_called_once()
        conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
