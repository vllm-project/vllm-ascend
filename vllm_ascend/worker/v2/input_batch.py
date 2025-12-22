import numpy as np
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
        # Create seq_lens_cpu and seq_lens_np.
        # npu's attention backend still needs seq_lens on CPU side.
        self.seq_lens_cpu: torch.Tensor = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        # seq_len_np and seq_lens_cpu share the same memory.
        # define seq_lens_np for easier calculation with numpy.
        self.seq_lens_np: np.ndarray = self.seq_lens_cpu.numpy()
