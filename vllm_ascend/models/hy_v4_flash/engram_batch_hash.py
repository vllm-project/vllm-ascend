# SPDX-License-Identifier: Apache-2.0
import torch


def batched_hashes(
    input_ids, lookback, starts, token_map, multipliers, primes, offsets, pad_id, pad_token_id, visual_ids, output
):
    # Host request boundaries are already available at this caller.
    lengths = [int(b) - int(a) for a, b in zip(starts, starts[1:])]
    count = sum(lengths)
    output.fill_(-1)
    if count == 0:
        return output
    device = input_ids.device
    request = torch.tensor([r for r, n in enumerate(lengths) for _ in range(n)], device=device, dtype=torch.long)
    local = torch.tensor([i for n in lengths for i in range(n)], device=device, dtype=torch.long)
    position = torch.arange(count, device=device)
    current = input_ids[:count]
    rolling = torch.zeros((multipliers.shape[0], count), dtype=torch.int64, device=device)
    blocked = torch.zeros(count, dtype=torch.bool, device=device)
    hashes = []
    for shift in range(multipliers.shape[1]):
        preceding = local - shift
        history_index = (-preceding - 1).clamp(0, lookback.shape[1] - 1)
        tokens = torch.where(
            preceding >= 0, input_ids[(position - shift).clamp_min(0)], lookback[request, history_index]
        ).long()
        blocked = blocked | (tokens < 0) | (tokens >= token_map.numel())
        mapped = token_map[tokens.clamp(0, token_map.numel() - 1)].long()
        mapped = torch.where(blocked, pad_id, mapped)
        rolling = torch.bitwise_xor(rolling, mapped * multipliers[:, shift, None])
        if shift:
            hashes.append(
                rolling.unsqueeze(-1) % primes[:, None, shift - 1] + offsets.reshape_as(primes)[:, None, shift - 1]
            )
    invalid = (current < 0) | (current >= token_map.numel()) | (current == pad_token_id)
    for token in visual_ids:
        invalid = invalid | (current == token)
    output[:, :count].copy_(torch.cat(hashes, dim=-1).masked_fill(invalid[None, :, None], -1).int())
    return output


class DecodeHashGraph:
    """Capture hash arithmetic; keep request boundaries as explicit inputs.

    The graph writes the existing padded output view. Request layout changes
    update only fixed input buffers, including the validity mask for padding.
    CPU-offloaded embedding lookup still consumes the completed hash tensor.
    """

    def __init__(
        self, input_ids, lookback, token_map, multipliers, primes, offsets, pad_id, pad_token_id, visual_ids, output
    ):
        self.capacity = output.shape[1]
        self.ids = torch.zeros(self.capacity, dtype=input_ids.dtype, device=input_ids.device)
        self.history = torch.zeros((self.capacity, lookback.shape[1]), dtype=lookback.dtype, device=lookback.device)
        self.request = torch.zeros(self.capacity, dtype=torch.long, device=input_ids.device)
        self.local = torch.zeros_like(self.request)
        self.live = torch.zeros(self.capacity, dtype=torch.bool, device=input_ids.device)
        self.position = torch.arange(self.capacity, device=input_ids.device)
        self.output = output

        def forward():
            rolling = torch.zeros((multipliers.shape[0], self.capacity), dtype=torch.int64, device=input_ids.device)
            blocked = torch.zeros_like(self.live)
            hashes = []
            for shift in range(multipliers.shape[1]):
                preceding = self.local - shift
                history_index = (-preceding - 1).clamp(0, self.history.shape[1] - 1)
                tokens = torch.where(
                    preceding >= 0,
                    self.ids[(self.position - shift).clamp_min(0)],
                    self.history[self.request, history_index],
                ).long()
                blocked = blocked | (tokens < 0) | (tokens >= token_map.numel())
                mapped = token_map[tokens.clamp(0, token_map.numel() - 1)].long()
                mapped = torch.where(blocked, pad_id, mapped)
                rolling = torch.bitwise_xor(rolling, mapped * multipliers[:, shift, None])
                if shift:
                    hashes.append(
                        rolling.unsqueeze(-1) % primes[:, None, shift - 1]
                        + offsets.reshape_as(primes)[:, None, shift - 1]
                    )
            invalid = (~self.live) | (self.ids < 0) | (self.ids >= token_map.numel()) | (self.ids == pad_token_id)
            for token in visual_ids:
                invalid = invalid | (self.ids == token)
            output.copy_(torch.cat(hashes, dim=-1).masked_fill(invalid[None, :, None], -1).int())

        # Capture only on a cache miss, outside measured steady-state windows.
        for _ in range(2):
            forward()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            forward()

    def run(self, input_ids, lookback, starts):
        lengths = [int(b) - int(a) for a, b in zip(starts, starts[1:])]
        count = sum(lengths)
        if (
            int(starts[0]) != 0
            or min(lengths, default=0) < 0
            or count > self.capacity
            or len(lengths) > self.capacity
            or lookback.shape[0] != len(lengths)
        ):
            raise ValueError("Invalid request layout for captured Engram hashes")
        request = [r for r, n in enumerate(lengths) for _ in range(n)]
        local = [i for n in lengths for i in range(n)]
        request.extend([0] * (self.capacity - count))
        local.extend([0] * (self.capacity - count))
        self.request.copy_(torch.tensor(request, dtype=torch.long))
        self.local.copy_(torch.tensor(local, dtype=torch.long))
        self.live.copy_(torch.arange(self.capacity) < count)
        self.ids[:count].copy_(input_ids[:count])
        self.history[: len(lengths)].copy_(lookback)
        self.graph.replay()
        return self.output
