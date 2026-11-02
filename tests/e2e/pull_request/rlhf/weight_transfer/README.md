# RLHF weight-transfer tests

The former one-card IPC and combined two-card HCCL entry points are replaced by
model-family files under `tests/e2e/pull_request/two_card/rlhf/weight_transfer/`.
Keep one family per file so CI can schedule and time each family independently.

## Responsibilities

- `models/`: reduced HF configurations, checkpoint-name adapters and registry selection.
- `runners/`: IPC/HCCL transactions; `common.py` shares server lifecycle and device mapping.
- `utils.py`: deterministic weights, checkpoint serialization and startup-load reference.
- `tests/ut/distributed/weight_transfer/`: tool, metadata and engine regression checks.

Each E2E compares two live updates against an independent startup-loaded reference;
dummy output must differ. Config/tokenizer assets are required, but checkpoint weights
are generated locally. IPC uses TP2 with one producer per card; HCCL uses TP1 inference
and a separate trainer card. All enabled models cover IPC unpacked; Qwen3.5 represents
the full transport matrix. Packed lanes currently skip pending #18763; DeepSeek is skipped.

## Add a model: four steps

1. Define a `WeightUpdateModelCase` in `models/<family>.py`, reducing layers/experts with `hf_overrides`; add a checkpoint adapter only when native HF names differ.
2. Export the family tuple and include it in `models/registry.py`. Leave `full_transfer_matrix=False` for IPC-unpacked-only coverage.
3. Add `ipc/test_<family>_ipc.py` using the template below; add its file budget to `.github/workflows/scripts/estimated_times.yaml` after measurement.
4. Add focused metadata/mapping UT coverage, then run the UTs and the new E2E file. Use existing runners unless the transaction protocol changes.

## Add a backend: three steps

1. Implement `runners/<backend>.py`, reusing `common.py` and the reference/two-update assertions.
2. Extend backend selection in `models/registry.py` and add representative-family entry files under `two_card/rlhf/weight_transfer/<backend>/`.
3. Register the engine if needed, add protocol UTs, validate E2E and record the new file's scheduling budget.

## When to change a runner

| Change | Location |
| --- | --- |
| Model ID, reduced dimensions, names or model-specific server flags | `models/<family>.py` |
| Which family/backend/packing combinations execute | `models/registry.py` and entry files |
| Shared checkpoint generation or reference behavior | `utils.py`; shared server lifecycle in `runners/common.py` |
| Device topology, synchronization, sleep/wake or transaction ordering | Protocol runner; update its regression coverage |

## Run from the repository root

```bash
python -m pytest -sv --durations=0 tests/e2e/pull_request/two_card/rlhf/weight_transfer/ipc/test_minimax_ipc.py
python -m pytest -q tests/ut/distributed/weight_transfer/
```

Use two available NPUs; `ASCEND_RT_VISIBLE_DEVICES` selects physical cards and
`VLLM_RL_TEST_DEVICE_INDEX` selects the first logical card within that visibility.

## Copyable IPC entry template

Create `ipc/test_<family>_ipc.py`; replace `qwen` / `QWEN_CASES` with the family.
For HCCL, use backend `hccl` and runner `run_hccl_weight_transfer_transaction` from `runners.hccl`.

```python
import pytest

from tests.e2e.pull_request.rlhf.weight_transfer.models.base import WeightUpdateModelCase
from tests.e2e.pull_request.rlhf.weight_transfer.models.qwen import QWEN_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.registry import pytest_model_cases
from tests.e2e.pull_request.rlhf.weight_transfer.runners.npu_ipc import run_npu_ipc_weight_transfer_transaction


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=QWEN_CASES))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
```
