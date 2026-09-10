# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary, deliberately synchronizing SFA diagnostic worker; not production."""

import hashlib
import inspect
import json

import torch

from vllm_ascend.worker.worker import NPUWorker


class SFATraceWorker(NPUWorker):
    def load_model(self):
        super().load_model()
        self.trace_step = 0
        self.trace_batch = None
        self.trace_captures = []
        self.trace_buffers = {}
        state = self.model_runner.model_state
        original = state.prepare_attn
        signature = inspect.signature(original)

        def prepare(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            metadata = original(*args, **kwargs)
            batch = bound.arguments["input_batch"]
            if bound.arguments.get("for_capture", False):
                self.trace_captures.append(metadata)
            elif not getattr(batch, "is_dummy", False):
                self.trace_batch = (batch, metadata)
            return metadata

        state.prepare_attn = prepare
        for name, module in self.model_runner.get_model().named_modules():
            if name.endswith((".self_attn", ".mlp", ".input_layernorm", ".post_attention_layernorm")):
                module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def snapshot(module, inputs, output):
            values = output if isinstance(output, tuple) else (output,)
            for index, value in enumerate(values):
                if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] not in (2, 4):
                    continue
                key = (name, index, tuple(value.shape))
                if key not in self.trace_buffers:
                    self.trace_buffers[key] = torch.empty_like(value)
                # This copy is recorded in the graph; host reads occur only after execution.
                self.trace_buffers[key].copy_(value)

        return snapshot

    def _tensor(self, value):
        cpu = value.detach().cpu().contiguous()
        return {
            "shape": list(value.shape),
            "ptr": value.data_ptr(),
            "values": cpu.flatten()[:64].tolist(),
            "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        }

    def _metadata(self, metadata):
        if not metadata:
            return {}
        first = next(iter(metadata.values()))
        if isinstance(first, dict):
            first = next(iter(first.values()))
        fields = (
            "slot_mapping",
            "pcp_slot_mapping",
            "block_table",
            "seq_lens",
            "positions",
            "cos",
            "sin",
            "query_start_loc",
            "cum_query_lens",
            "seq_lens_cpu",
            "actual_seq_lengths_query",
            "actual_seq_lengths_key",
            "num_actual_tokens",
            "num_decode_tokens",
            "num_decodes",
        )
        return {
            name: self._tensor(value) if isinstance(value, torch.Tensor) else value
            for name in fields
            if (value := getattr(first, name, None)) is not None
        }

    def execute_model(self, scheduler_output):
        result = super().execute_model(scheduler_output)
        if self.trace_batch is None or self.trace_step >= 5:
            return result
        batch, metadata = self.trace_batch
        self.trace_batch = None
        self.trace_step += 1
        batch_fields = (
            "input_ids",
            "positions",
            "seq_lens",
            "query_start_loc",
            "cum_query_lens",
            "seq_lens_cpu",
            "logits_indices",
            "idx_mapping",
            "num_reqs",
            "num_tokens",
            "num_tokens_after_padding",
            "req_ids",
        )
        batch_record = {
            name: self._tensor(value) if isinstance(value, torch.Tensor) else value
            for name in batch_fields
            if (value := getattr(batch, name, None)) is not None
        }
        print(
            "SFA_RUNTIME_TRACE "
            + json.dumps(
                {
                    "rank": self.rank,
                    "step": self.trace_step,
                    "batch": batch_record,
                    "runtime": self._metadata(metadata),
                    "captures": [self._metadata(item) for item in self.trace_captures],
                },
                default=str,
            ),
            flush=True,
        )
        # Prefill snapshots are intentionally omitted: only small decode-shaped copies are captured.
        if batch.num_tokens <= 4:
            for (name, index, shape), value in self.trace_buffers.items():
                cpu = value.detach().cpu().contiguous()
                rows = [
                    {
                        "sha256": hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        "first": row.float()[:8].tolist(),
                        "norm": row.float().norm().item(),
                    }
                    for row in cpu
                ]
                print(
                    "SFA_LAYER_TRACE "
                    + json.dumps(
                        {
                            "rank": self.rank,
                            "step": self.trace_step,
                            "module": name,
                            "output_index": index,
                            "shape": shape,
                            "rows": rows,
                        }
                    ),
                    flush=True,
                )
        return result
