# SPDX-License-Identifier: Apache-2.0
"""Explicit DP ownership and configured producer discovery for P/D.

This module does not assign KV blocks or perform collectives. Each worker keeps
the upstream TP group and owns one DP-local runtime. A consumer selects the
producer replica named by the ticket, independently of its own DP rank.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


@dataclass(frozen=True)
class Replica:
    engine_id: str
    dp_rank: int
    tp_size: int

    def __post_init__(self):
        if not isinstance(self.engine_id, str) or not self.engine_id:
            raise ValueError("An explicit engine instance ID is required")
        _integer(self.dp_rank, "DP rank")
        _integer(self.tp_size, "TP size", 1)

    def request_key(self, request_id: str) -> tuple[str, int, str]:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("An explicit request ID is required")
        return self.engine_id, self.dp_rank, request_id

    def ticket_identity(self) -> dict:
        return {"engine_id": self.engine_id, "dp_rank": self.dp_rank, "tp_size": self.tp_size}


@dataclass(frozen=True)
class ProducerPeer:
    replica: Replica
    host: str
    first_port: int

    def __post_init__(self):
        # Hosts come from deployment configuration, never from request tickets.
        ipaddress.ip_address(self.host)
        _integer(self.first_port, "first control port", 1)
        if self.first_port + self.replica.tp_size - 1 > 65535:
            raise ValueError("Producer control port range exceeds 65535")

    def endpoint(self, tp_rank: int) -> str:
        _integer(tp_rank, "TP rank")
        if tp_rank >= self.replica.tp_size:
            raise ValueError("TP rank outside the configured producer group")
        host = f"[{self.host}]" if ipaddress.ip_address(self.host).version == 6 else self.host
        return f"tcp://{host}:{self.first_port + tp_rank}"


class ProducerDirectory:
    """Allowlisted producers; different producer/consumer DP sizes are valid."""

    def __init__(self, peers: list[ProducerPeer]):
        self._peers = {}
        listeners = set()
        if not peers:
            raise ValueError("Configure at least one producer replica")
        for peer in peers:
            key = peer.replica.engine_id, peer.replica.dp_rank
            if key in self._peers:
                raise ValueError("Duplicate producer replica identity")
            host = str(ipaddress.ip_address(peer.host))
            for port in range(peer.first_port, peer.first_port + peer.replica.tp_size):
                if (host, port) in listeners:
                    raise ValueError("Producer control port ranges overlap")
                listeners.add((host, port))
            self._peers[key] = peer

    def resolve(self, identity: dict, consumer: Replica, tp_rank: int) -> str:
        if not isinstance(identity, dict) or set(identity) != {"engine_id", "dp_rank", "tp_size"}:
            raise ValueError("Missing or unsupported producer identity")
        producer = Replica(**identity)
        peer = self._peers.get((producer.engine_id, producer.dp_rank))
        if peer is None or peer.replica != producer:
            raise ValueError("Ticket names an unconfigured producer replica")
        if producer.tp_size != consumer.tp_size:
            raise ValueError("Cross-TP-size KV resharding is not implemented")
        # consumer.dp_rank deliberately does not participate in peer selection.
        return peer.endpoint(tp_rank)


def contiguous_peers(engine_id: str, hosts: list[str], tp_size: int, base_port: int) -> list[ProducerPeer]:
    """Assign a nonoverlapping port range per producer DP replica."""
    _integer(tp_size, "TP size", 1)
    _integer(base_port, "base control port", 1)
    peers = [
        ProducerPeer(Replica(engine_id, dp_rank, tp_size), host, base_port + dp_rank * tp_size)
        for dp_rank, host in enumerate(hosts)
    ]
    ProducerDirectory(peers)
    return peers
