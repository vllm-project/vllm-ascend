"""Enable deterministic PyTorch state in spawned accuracy-test processes."""

import os
import random

import numpy as np
import torch

_SEED = 0

random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if hasattr(torch, "npu"):
    torch.npu.manual_seed_all(_SEED)
torch.use_deterministic_algorithms(True)

print(
    f"[accuracy-probe][bootstrap] pid={os.getpid()} seed={_SEED} "
    f"torch_deterministic={torch.are_deterministic_algorithms_enabled()}",
    flush=True,
)
