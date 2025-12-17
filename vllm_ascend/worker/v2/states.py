from contextlib import contextmanager

import torch
from vllm.v1.worker.gpu.states import RequestState, UvaBuffer
from vllm.v1.utils import CpuGpuBuffer


class AscendRequestState(RequestState):
    """Request state for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
        pin_memory: bool,
    ):
        with uva_wrapper():
            super().__init__(
                max_num_reqs,
                max_model_len,
                max_num_batched_tokens,
                num_speculative_steps,
                vocab_size,
                device,
                pin_memory,
            )
        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.prefill_token_ids
        # vllm gpu_model_runner_v2 deprecate the seqs_lens_cpu attribute,
        # because they think most attention backends do not need it.
        # However, Ascend attention backend muse uses seqs_lens_cpu,
        # so we keep num_computed_tokens_cpu here, seq_lens_cpu need to be
        # calculated by num_computed_tokens_cpu + decode_token_per_req outside.
        self.num_computed_tokens_cpu: torch.Tensor = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        # NOTE(Ronald1995): Ascend NPUs do not support UVA yet,
        # so we use CpuGpuBuffer to allocate prefill_token_ids buffer.
        self.prefill_token_ids: CpuGpuBuffer = self._make_buffer(  # type: ignore
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32)

    def add_request(
        self,
        req_id,
        prompt_len,
        prefill_token_ids,
        num_computed_tokens,
        sampling_params,
        lora_request,
    ):

        super().add_request(
            req_id,
            prompt_len,
            prefill_token_ids,
            num_computed_tokens,
            sampling_params,
            lora_request,
        )
        req_idx = self.req_id_to_index[req_id]
        self.num_computed_tokens_cpu[req_idx] = num_computed_tokens


@contextmanager
def uva_wrapper():
    """Context manager to disable UVA for Ascend NPUs."""

    class UvaBufferWrapper:

        def __init__(self, *args, **kwargs):
            pass

    # TODO(Ronald1995): rectify this when NPU support uva.
    global UvaBuffer
    ori_class = UvaBuffer
    try:
        UvaBuffer = UvaBufferWrapper
        yield
    finally:
        UvaBuffer = ori_class
