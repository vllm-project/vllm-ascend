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

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits
        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits.masked_fill_(invalid_token_mask, -float('inf'))
        return logits
