import torch
from vllm.v1.worker.gpu.input_batch import InputBuffers


class AscendInputBuffers(InputBuffers):
    """Input buffers for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        inputs_embeds_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ):
        super().__init__(
            max_num_reqs,
            max_num_tokens,
            inputs_embeds_size,
            vocab_size,
            dtype,
            device,
            pin_memory,
        )
        self.seq_lens_cpu: torch.Tensor = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
