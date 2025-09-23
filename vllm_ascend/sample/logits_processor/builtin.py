import torch
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import MinPLogitsProcessor


class AscendMinPLogitsProcessor(MinPLogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)

        decode_max_num_seqs = getattr(vllm_config.scheduler_config,
                                      'decode_max_num_seqs', 0)
        if decode_max_num_seqs != 0:
            max_num_reqs = max(vllm_config.scheduler_config.max_num_seqs,
                               decode_max_num_seqs)

            self.min_p_count: int = 0

            self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                                dtype=torch.float32,
                                                device="cpu",
                                                pin_memory=is_pin_memory)
            self.min_p_cpu = self.min_p_cpu_tensor.numpy()

            self.use_double_tensor = torch.device(device).type != "cpu"

            if self.use_double_tensor:
                # Pre-allocated device tensor
                self.min_p_device: torch.Tensor = torch.empty(
                    (max_num_reqs, ), dtype=torch.float32, device=device)
            else:
                self.min_p_device = self.min_p_cpu_tensor
            # Current slice of the device tensor
            self.min_p: torch.Tensor = self.min_p_device[:0]
