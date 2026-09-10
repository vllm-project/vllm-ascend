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
        self.trace_counters = {}
        self.trace_rows = torch.arange(4, device=self.device)
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
                if not isinstance(value, torch.Tensor) or value.ndim != 2:
                    continue
                key = (name, index)
                if key not in self.trace_buffers:
                    self.trace_buffers[key] = value.new_zeros((4, value.shape[1]))
                    self.trace_counters[key] = value.new_zeros((), dtype=torch.int64)
                # This copy is recorded in the graph; host reads occur only after execution.
                # Keep both copy operands shape-four even when the compiled graph
                # receives fewer tokens. A dynamic slice may be specialized by
                # the no-guards compiler; repeated last rows are diagnostic padding.
                rows = self.trace_rows.clamp_max(value.shape[0] - 1)
                self.trace_buffers[key].copy_(torch.index_select(value, 0, rows))
                self.trace_counters[key].add_(1)

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
        if self.trace_batch is None or self.trace_step >= 12:
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
            "is_prefilling_np",
            "num_scheduled_tokens",
            "num_computed_tokens_np",
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
        if not batch.is_prefilling_np.any():
            for (name, index), value in self.trace_buffers.items():
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
                            "shape": list(value.shape),
                            "updates": int(self.trace_counters[(name, index)].item()),
                            "rows": rows,
                        }
                    ),
                    flush=True,
                )
        return result
