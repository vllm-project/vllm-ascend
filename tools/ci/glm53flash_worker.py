# SPDX-License-Identifier: Apache-2.0
"""Test-only logits capture and graph replay instrumentation for Flash."""

import hashlib

import numpy as np
import torch


class BaselineWorker:
    def start_glm51_replay_count(self):
        self._glm51_replays = 0
        original = torch.npu.NPUGraph.replay
        self._glm51_original_replay = original

        def replay(graph, *args, **kwargs):
            self._glm51_replays += 1
            return original(graph, *args, **kwargs)

        torch.npu.NPUGraph.replay = replay

    def finish_glm51_replay_count(self):
        torch.npu.NPUGraph.replay = self._glm51_original_replay
        return self._glm51_replays

    def fingerprint(self):
        digest = hashlib.sha256()
        total = 0
        for name, value in sorted(self.model_runner.model.named_parameters()):
            cpu = value.detach().cpu().contiguous()
            digest.update(f"{name}|{value.dtype}|{list(value.shape)}".encode())
            raw = cpu.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
            total += raw.nbytes
        return {"rank": self.rank, "sha256": digest.hexdigest(), "bytes": total}

    def begin_capture(self):
        self._captured_logits = []
        model = self.model_runner.model
        self._original_compute_logits = model.compute_logits

        def capture(*args, **kwargs):
            logits = self._original_compute_logits(*args, **kwargs)
            if self.rank == 0 and logits is not None:
                self._captured_logits.append(logits.detach().float().cpu().numpy().copy())
            return logits

        model.compute_logits = capture

    def end_capture(self, path):
        self.model_runner.model.compute_logits = self._original_compute_logits
        if self.rank != 0:
            return None
        shapes = [list(x.shape) for x in self._captured_logits]
        assert len(shapes) >= 8, shapes
        # Single-request collection: discard intermediate chunked-prefill rows.
        data = np.concatenate(self._captured_logits[-8:], axis=0)
        assert data.shape == (8, 154880), data.shape
        assert np.isfinite(data).all()
        np.save(path, data, allow_pickle=False)
        self._captured_logits = []
        return {"path": path, "shape": list(data.shape), "all_call_shapes": shapes}


def prompts(batch, length):
    return [{"prompt_token_ids": [10 + (i + r * 137) % 1000 for i in range(length)]} for r in range(batch)]
