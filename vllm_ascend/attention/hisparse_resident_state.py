# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle for one eager resident mapping shared by an index group."""

from dataclasses import dataclass
from typing import Any


def eligible_resident_forward(enforce_eager, graph_mode_none, metadata_present, dummy, resident_flags):
    return (
        enforce_eager
        and graph_mode_none
        and metadata_present
        and not dummy
        and bool(resident_flags)
        and all(resident_flags)
    )


@dataclass
class ResidentForwardState:
    """Owned by one actual index group; renewed by its leader every forward.

    Metadata buffers may be updated in place, including between MTP draft
    forwards. Their pointer/shape signature therefore never serves as an epoch.
    Only a new leader invocation advances the epoch and drops the old result.
    """

    epoch: int = 0
    signature: tuple | None = None
    route: bool | None = None
    result: Any = None

    def begin(self, layer_index, signature):
        if layer_index == 0:
            self.epoch += 1
            self.signature = signature
            self.route = None
            self.result = None
        elif self.epoch == 0 or self.signature != signature:
            raise RuntimeError("Resident follower lacks this forward's leader epoch")

    def choose(self, layer_index, signature, enabled):
        if self.epoch == 0 or self.signature != signature:
            raise RuntimeError("Resident mapping belongs to another forward")
        if layer_index == 0:
            if self.route is not None:
                raise RuntimeError("Resident leader already selected this forward's route")
            self.route = enabled
        elif self.route is None:
            raise RuntimeError("Resident follower ran before its leader selected a route")
        elif self.route != enabled:
            raise RuntimeError("Resident route eligibility changed inside one group forward")
        return self.route

    def publish(self, layer_index, result):
        if layer_index != 0 or self.route is not True or self.result is not None:
            raise RuntimeError("Only this forward's resident leader can publish once")
        self.result = result

    def consume(self):
        if self.route is not True or self.result is None:
            raise RuntimeError("Resident mapping is unavailable for this forward")
        return self.result


def resident_eager_transport_allowed(enforce_eager, is_a5, transfer):
    """Reuse resident mappings only for the qualified direct connector.

    Composite connectors can supply separate current metadata views per layer.
    Their normal resolver remains the owner until shared views are qualified.
    This does not alter resident storage, Host import, or graph execution.
    """
    return bool(enforce_eager and is_a5 and (transfer is None or transfer.kv_connector == "HiSparseConnector"))
