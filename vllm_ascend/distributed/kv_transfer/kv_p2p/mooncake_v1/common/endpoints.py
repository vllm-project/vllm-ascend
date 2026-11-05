# SPDX-License-Identifier: Apache-2.0
"""Handshake port formulas shared by the mooncake v1 KV connectors.

Two formula families are kept on purpose; they are **not** interchangeable and
must not be unified:

* pull family (``mooncake_connector``), used by both the scheduler and the
  worker::

      base = kv_port + data_parallel_index * tp_size * pp_size * pcp_size
      device_index = (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
      handshake = base + device_index

* layerwise family (``mooncake_layerwise_connector``): the scheduler and the
  worker compute *different* base ports historically::

      scheduler base = kv_port + data_parallel_rank * tp_size
      worker base = kv_port + dp_rank * pcp_size * tp_size + pcp_rank * tp_size
      handshake = base + tp_rank

  The two bases coincide only when the decode-side prefill context parallel size
  is 1 (the layerwise scheduler asserts that for KV consumers). The difference
  is the current behaviour and is locked by golden unit tests, so it is
  preserved as-is until a dedicated behaviour change is requested.

All functions are pure and keyword-only; this module must not import ``vllm``
or ``vllm_ascend``.
"""


def pull_base_port(*, kv_port: int, dp_index: int, tp_size: int, pp_size: int, pcp_size: int) -> int:
    """Base handshake port shared by the pull scheduler and worker.

    ``base = kv_port + dp_index * tp_size * pp_size * pcp_size``
    """
    return kv_port + dp_index * tp_size * pp_size * pcp_size


def pull_device_index(*, pp_rank: int, pcp_size: int, pcp_rank: int, tp_size: int, tp_rank: int) -> int:
    """Per-rank offset added to the pull base port.

    ``device_index = (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank``
    """
    return (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank


def pull_handshake_port(
    *, base_port: int, pp_rank: int, pcp_size: int, pcp_rank: int, tp_size: int, tp_rank: int
) -> int:
    """Handshake port of one pull rank: ``base_port + device_index``."""
    return base_port + pull_device_index(
        pp_rank=pp_rank,
        pcp_size=pcp_size,
        pcp_rank=pcp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


def layerwise_scheduler_base_port(*, kv_port: int, dp_rank: int, tp_size: int) -> int:
    """Base handshake port computed by the layerwise scheduler.

    ``base = kv_port + dp_rank * tp_size`` (scheduler formula, see module docs).
    """
    return kv_port + dp_rank * tp_size


def layerwise_worker_base_port(*, kv_port: int, dp_rank: int, pcp_size: int, tp_size: int, pcp_rank: int) -> int:
    """Base handshake port computed by the layerwise worker.

    ``base = kv_port + dp_rank * pcp_size * tp_size + pcp_rank * tp_size``
    (worker formula, see module docs).
    """
    return kv_port + dp_rank * pcp_size * tp_size + pcp_rank * tp_size


def layerwise_handshake_port(*, base_port: int, tp_rank: int) -> int:
    """Handshake port of one layerwise rank: ``base_port + tp_rank``."""
    return base_port + tp_rank


__all__ = [
    "layerwise_handshake_port",
    "layerwise_scheduler_base_port",
    "layerwise_worker_base_port",
    "pull_base_port",
    "pull_device_index",
    "pull_handshake_port",
]
