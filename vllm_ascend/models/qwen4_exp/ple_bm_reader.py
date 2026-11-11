# SPDX-License-Identifier: Apache-2.0
"""Graph-lifetime reader for a shared Qwen PLE BM table."""

import torch

from .ple_bm_table import attach_table
from .ple_checkpoint import PLECheckpoint, checkpoint_identity
from .ple_layout import PLELayout


class PLEBMReader(torch.nn.Module):
    """Keep the lease and pointer table alive until model graphs have stopped."""

    def __init__(self, client, checkpoint, key: str, vocab_start: int, vocab_end: int):
        super().__init__()
        self._shared_pool_client = client
        self._closing = False
        try:
            if not 0 <= vocab_start <= vocab_end <= checkpoint.rows:
                raise ValueError("PLE vocabulary range exceeds the checkpoint")
            self.layout = PLELayout(checkpoint.rows, checkpoint.width * 2)
            self.width = checkpoint.width
            self.vocab_start, self.vocab_end = vocab_start, vocab_end
            pointers = attach_table(client, key, self.layout, self.width)
            self.register_buffer("pointers", pointers, persistent=False)
        except BaseException as exc:
            try:
                client.close()
            except Exception as cleanup:
                exc.add_note(f"Shared PLE client cleanup failed: {cleanup}")
            raise
        # The model's loader may skip these names only after attachment succeeds.
        self.ple_external_checkpoint_names = checkpoint.weight_names

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self._closing or self._shared_pool_client is None:
            raise RuntimeError("Shared PLE reader is closed")
        from vllm_ascend.ops.triton.qwen4_exp.ple_shared import gather_ple_rows

        output = torch.empty((*ids.shape, self.width), dtype=torch.bfloat16, device=ids.device)
        gather_ple_rows(
            self.pointers,
            ids,
            output,
            vocab_start=self.vocab_start,
            vocab_end=self.vocab_end,
            rows_per_mapping=self.layout.rows_per_mapping,
        )
        return output

    def close_host_offload(self) -> None:
        """Close after graph teardown; retain mappings on failed cleanup."""
        if self._shared_pool_client is None:
            return
        self._closing = True
        self._shared_pool_client.close()
        self._shared_pool_client = None
        self.pointers = None


def open_bm_reader(model_path, prefix: str, parts: int, width: int, root, device, vocab_start: int, vocab_end: int):
    """Attach to the owner matching this immutable checkpoint and table."""
    from vllm_ascend.models.deepseek_v41.engram.shared_pool import SharedPoolClient

    checkpoint = PLECheckpoint(model_path, prefix, parts, width)
    client = SharedPoolClient(root, checkpoint_identity(model_path, prefix), device=device)
    return PLEBMReader(client, checkpoint, prefix, vocab_start, vocab_end)
