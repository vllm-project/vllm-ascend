# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import torch


@dataclass
class LoRAKernelMetaNPU:
    max_loras: int
    max_num_batched_tokens: int
    device: torch.device | str
    num_active_loras: int = 0
    captured_lora_counts: list[int] = field(default_factory=list)

    @classmethod
    def make(
        cls,
        max_loras: int,
        max_num_batched_tokens: int,
        device: torch.device | str,
        captured_lora_counts: list[int] | None = None,
    ) -> "LoRAKernelMetaNPU":
        obj = cls(
            max_loras=max_loras,
            max_num_batched_tokens=max_num_batched_tokens,
            device=device,
            captured_lora_counts=sorted(captured_lora_counts) if captured_lora_counts else [],
        )
        obj.token_lora_mapping = torch.empty(
            max_num_batched_tokens, dtype=torch.long, device=device
        )
        obj.token_indices_sorted_by_lora_ids = torch.empty(
            max_num_batched_tokens, dtype=torch.long, device=device
        )
        obj.num_tokens_per_lora = torch.zeros(
            max_loras + 1, dtype=torch.long, device=device
        )
        obj.lora_token_start_loc = torch.zeros(
            max_loras + 2, dtype=torch.long, device=device
        )
        obj.active_lora_ids = torch.full(
            (max_loras + 1,), -1, dtype=torch.long, device=device
        )
        obj.no_lora_flag_cpu = torch.zeros(1, dtype=torch.bool, device="cpu")
        return obj

    def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
        num_tokens = token_lora_mapping.numel()
        self.token_lora_mapping[:num_tokens].copy_(token_lora_mapping)

        self.token_indices_sorted_by_lora_ids.zero_()
        self.num_tokens_per_lora.zero_()
        self.lora_token_start_loc.zero_()
        self.active_lora_ids.fill_(-1)
        self.num_active_loras = 0

        if num_tokens == 0:
            self.no_lora_flag_cpu[0] = True
            return

        # Workaround: run sort/unique on CPU to avoid aclnnSort failure on NPU.
        token_lora_mapping_cpu = token_lora_mapping.to("cpu")

        _, token_indices_sorted_cpu = torch.sort(token_lora_mapping_cpu, stable=True)
        token_indices_sorted = token_indices_sorted_cpu.to(token_lora_mapping.device)
        self.token_indices_sorted_by_lora_ids[:num_tokens].copy_(token_indices_sorted)

        lora_ids_cpu, num_tokens_per_lora_cpu = torch.unique(
            token_lora_mapping_cpu, return_counts=True)
        lora_ids = lora_ids_cpu.to(token_lora_mapping.device)
        num_tokens_per_lora = num_tokens_per_lora_cpu.to(token_lora_mapping.device)

        self.active_lora_ids[: lora_ids.numel()].copy_(lora_ids)
        self.num_tokens_per_lora[: num_tokens_per_lora.numel()].copy_(num_tokens_per_lora)

        self.num_active_loras = lora_ids.numel()
        if self.captured_lora_counts and self.num_active_loras > 0:
            idx = bisect.bisect_left(self.captured_lora_counts, self.num_active_loras)
            if idx < len(self.captured_lora_counts):
                self.num_active_loras = self.captured_lora_counts[idx]

        lora_token_start_loc_cpu = torch.cumsum(num_tokens_per_lora_cpu, dim=0)
        lora_token_start_loc = lora_token_start_loc_cpu.to(token_lora_mapping.device)
        self.lora_token_start_loc[1 : 1 + lora_token_start_loc.numel()].copy_(
            lora_token_start_loc
        )

        no_lora = bool((lora_ids < 0).any().item())
        self.no_lora_flag_cpu[0] = no_lora

    def meta_args(
        self,
        token_nums: int,
        specialize_active_lora: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        max_loras = self.active_lora_ids.size(0) - 1
        return (
            self.token_lora_mapping[:token_nums],
            self.token_indices_sorted_by_lora_ids[:token_nums],
            self.num_tokens_per_lora,
            self.lora_token_start_loc,
            self.active_lora_ids,
            self.no_lora_flag_cpu,
            self.num_active_loras if specialize_active_lora else max_loras + 1,
        )
