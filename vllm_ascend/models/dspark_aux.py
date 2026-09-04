# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-to-draft auxiliary-hidden-state contracts for DSpark."""

from dataclasses import dataclass
from enum import Enum

import torch


class DSparkAuxHiddenFormat(str, Enum):
    """Semantic representation of target auxiliary hidden states."""

    MATERIALIZED = "materialized"
    RAW_PREFIX_SUM = "raw_prefix_sum"

    @property
    def capture_point(self) -> str:
        if self is DSparkAuxHiddenFormat.MATERIALIZED:
            return "pre_layer_materialized"
        return "post_layer_raw_prefix_sum"


@dataclass(frozen=True)
class DSparkAuxHiddenContract:
    """Static contract negotiated when a target and DSpark draft are loaded."""

    format: DSparkAuxHiddenFormat
    layer_ids: tuple[int, ...]
    capture_point: str
    target_hidden_size: int
    dtype: torch.dtype

    @property
    def packed_hidden_size(self) -> int:
        return len(self.layer_ids) * self.target_hidden_size

    def validate_definition(self) -> None:
        if not self.layer_ids:
            raise ValueError("DSpark auxiliary hidden contract requires at least one target layer")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError(f"DSpark auxiliary hidden layer IDs must be unique: {self.layer_ids}")
        if any(layer_id <= 0 for layer_id in self.layer_ids):
            raise ValueError(f"DSpark auxiliary hidden layer IDs must be positive boundaries: {self.layer_ids}")
        if self.target_hidden_size <= 0:
            raise ValueError("DSpark target hidden size must be positive")
        if self.capture_point != self.format.capture_point:
            raise ValueError(
                f"DSpark {self.format.value} auxiliary hidden states must use capture point "
                f"{self.format.capture_point!r}, got {self.capture_point!r}"
            )

    def validate_runtime(
        self,
        aux_hidden_states: list[torch.Tensor] | None,
        *,
        num_target_tokens: int,
        target_device: torch.device,
    ) -> None:
        """Validate shapes and placement without synchronizing device data.

        PCP/PP restoration can pack several selected layers into one tensor, so
        validate the aggregate feature width instead of requiring one tensor per
        layer.
        """
        if not aux_hidden_states:
            raise ValueError(
                f"DSpark draft requires {self.format.value} auxiliary hidden states from target layers {self.layer_ids}"
            )

        packed_hidden_size = 0
        for index, hidden_state in enumerate(aux_hidden_states):
            if hidden_state.ndim != 2:
                raise ValueError(f"DSpark auxiliary hidden state {index} must be 2-D, got {hidden_state.shape}")
            if hidden_state.shape[0] < num_target_tokens:
                raise ValueError(
                    f"DSpark auxiliary hidden state {index} has {hidden_state.shape[0]} token rows, "
                    f"but {num_target_tokens} are required"
                )
            if hidden_state.dtype != self.dtype:
                raise ValueError(
                    f"DSpark auxiliary hidden state {index} has dtype {hidden_state.dtype}, expected {self.dtype}"
                )
            if hidden_state.device != target_device:
                raise ValueError(
                    f"DSpark auxiliary hidden state {index} is on {hidden_state.device}, expected {target_device}"
                )
            packed_hidden_size += hidden_state.shape[-1]

        if packed_hidden_size != self.packed_hidden_size:
            raise ValueError(
                f"DSpark auxiliary hidden width is {packed_hidden_size}, "
                f"expected {self.packed_hidden_size} from layers {self.layer_ids}"
            )


def build_k3_mla_aux_hidden_contract(config, dtype: torch.dtype) -> DSparkAuxHiddenContract:
    """Build the raw-prefix-sum contract declared by a K3 MLA drafter."""
    target_layer_ids = getattr(config, "dspark_target_layer_ids", None) or getattr(
        config,
        "target_layer_ids",
        None,
    )
    if not target_layer_ids:
        raise ValueError("K3 MLA DSpark config must declare target_layer_ids")

    # DSpark checkpoint IDs name zero-based target layers. vLLM captures model
    # boundary IDs, where the output of layer i is boundary i + 1.
    layer_ids = tuple(int(layer_id) + 1 for layer_id in target_layer_ids)
    target_num_hidden_layers = getattr(config, "target_num_hidden_layers", None)
    if target_num_hidden_layers is not None:
        target_num_hidden_layers = int(target_num_hidden_layers)
        if target_num_hidden_layers <= 0:
            raise ValueError("K3 MLA DSpark target_num_hidden_layers must be positive")
        if any(layer_id > target_num_hidden_layers for layer_id in layer_ids):
            raise ValueError(
                f"K3 MLA DSpark target_layer_ids={tuple(target_layer_ids)} exceed "
                f"the declared {target_num_hidden_layers} target layers"
            )
    num_target_layers = int(getattr(config, "num_target_layers", len(layer_ids)))
    if num_target_layers != len(layer_ids):
        raise ValueError(
            f"K3 MLA DSpark num_target_layers={num_target_layers} does not match "
            f"target_layer_ids={tuple(target_layer_ids)}"
        )

    contract = DSparkAuxHiddenContract(
        format=DSparkAuxHiddenFormat.RAW_PREFIX_SUM,
        layer_ids=layer_ids,
        capture_point=DSparkAuxHiddenFormat.RAW_PREFIX_SUM.capture_point,
        target_hidden_size=int(config.target_hidden_size),
        dtype=dtype,
    )
    contract.validate_definition()
    return contract
