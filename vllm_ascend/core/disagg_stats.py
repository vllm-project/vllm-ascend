"""PD disaggregation cached_tokens correction.

In PD (Prefill-Decode) disaggregated deployment, the D-node's connector
reports KV-transfer tokens as prefix-cache hits (``num_external_cached_tokens``
= full prompt length N or N-1).  These tokens were **transferred** from the
P-node, not found in a cache — so reporting them as ``cached_tokens`` inflates
the metric and breaks billing (DeepSeek API bills cache hits at 2-10 % of
miss price).

This module provides:

* :func:`adjust_disagg_prefill_stats` — called in the scheduler WAITING loop
  right after ``request.prefill_stats.set()``.  On the D-side
  (``do_remote_prefill``) it overrides ``num_cached_tokens`` with the sum of
  D-local hits and the P-side's real hits (received via
  ``remote_num_cached_tokens``).  On the P-side (``do_remote_decode``) it
  records the real local prefix-cache hit count for the D-side to read.
* :class:`DisaggPrefillStatsMixin` — mixed into every ascend scheduler so the
  P-side's ``remote_num_cached_tokens`` is propagated through the finish path
  (``_free_request``) and the D-side's ``num_cache_creation_tokens`` is zeroed
  after ``finalize()`` runs (transferred KV is not "cache creation").

Non-PD scenarios (single-node, KV-pool connectors like AscendStore,
preemption-recovery via PreemptOffload, chunked-prefill continuations) are
unaffected: they have no ``do_remote_prefill`` / ``do_remote_decode`` flag,
so the helper is a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.stats import ModelRunnerOutput
    from vllm.v1.request import Request


def adjust_disagg_prefill_stats(
    request: Request,
    num_new_local_computed_tokens: int,
    connector_prefix_cache_hits: int,
) -> int:
    """Adjust ``prefill_stats`` for PD disaggregated requests.

    Call this **immediately after** ``request.prefill_stats.set(...)`` inside
    the ``if request.num_computed_tokens == 0:`` branch of the scheduler
    WAITING loop.

    * **D-side** (``do_remote_prefill``): the external tokens reported by the
      connector are KV-transfer tokens, not prefix-cache hits.  Override
      ``num_cached_tokens`` with ``D-local + P-real`` and mark the stats so
      :meth:`DisaggPrefillStatsMixin.update_from_output` can zero
      ``num_cache_creation_tokens`` afterwards.
    * **P-side** (``do_remote_decode``): record the real local prefix-cache
      hit count into ``kv_transfer_params["remote_num_cached_tokens"]`` so the
      finish path can propagate it to the D-node.
    * **Non-PD**: no-op — returns ``connector_prefix_cache_hits`` unchanged.

    Args:
        request: The request being scheduled.
        num_new_local_computed_tokens: Final local prefix-cache hit count
            (after connector reconciliation).
        connector_prefix_cache_hits: Current value of the
            ``connector_prefix_cache_hits`` scheduler variable.

    Returns:
        The (possibly adjusted) ``connector_prefix_cache_hits`` value.
        Returns ``0`` on the D-side so the D-node's Prometheus
        ``vllm:prefix_cache_hits`` metric reports zero external hits.
    """
    params = request.kv_transfer_params
    if not params:
        return connector_prefix_cache_hits

    if params.get("do_remote_prefill"):
        # D-side: transfer tokens are not prefix-cache hits.
        remote_cached = params.get("remote_num_cached_tokens", 0)
        if request.prefill_stats is not None:
            # D-local hits (shared prefix already in D's cache) + P-real hits.
            request.prefill_stats.num_cached_tokens = min(
                num_new_local_computed_tokens + remote_cached,
                request.num_prompt_tokens,
            )
            # Flag for update_from_output to zero num_cache_creation_tokens.
            request.prefill_stats._disagg_d_side = True  # type: ignore[attr-defined]
        return 0

    # Balance/profiling schedulers re-run this helper for post-preemption
    # re-prefills (their guard lacks the num_preemptions check).  Clear the
    # stale flag from the original remote prefill, or update_from_output
    # would wrongly zero num_cache_creation_tokens for a genuine local
    # re-prefill.
    if request.prefill_stats is not None:
        request.prefill_stats._disagg_d_side = False  # type: ignore[attr-defined]

    if params.get("do_remote_decode"):
        # P-side: record real local prefix cache hits for D-side.
        params["remote_num_cached_tokens"] = num_new_local_computed_tokens

    return connector_prefix_cache_hits


class DisaggPrefillStatsMixin:
    """Mixin for PD disaggregation cached_tokens correction.

    Mixed into every ascend scheduler (:class:`RecomputeScheduler`,
    :class:`BalanceScheduler`, :class:`ProfilingChunkScheduler`,
    :class:`DyntraLBScheduler`) to provide two finish-path hooks:

    1. ``_free_request`` — after the upstream frees the request, copy
       ``remote_num_cached_tokens`` from ``request.kv_transfer_params`` into
       the dict returned to ``EngineCoreOutput.kv_transfer_params`` so the
       proxy can forward it to the D-node.
    2. ``update_from_output`` — after the upstream runs ``finalize()``, zero
       ``num_cache_creation_tokens`` for D-side requests (transferred KV is
       not "cache creation"; leaving it non-zero would overcharge the client
       at cache-creation billing rates).
    """

    def _free_request(self, request: Request, delay_free_blocks: bool = False) -> dict[str, Any] | None:
        kv_xfer_params = super()._free_request(request, delay_free_blocks)  # type: ignore[misc]

        params = request.kv_transfer_params
        if params and params.get("do_remote_decode") and "remote_num_cached_tokens" in params:
            # P-side: propagate real prefix-cache hit count to D-side via the
            # connector's finish dict.  Connectors build their own return dict
            # and do not auto-passthrough this key, so we supplement it here
            # (covers all connector types, including those returning None).
            if kv_xfer_params is None:
                kv_xfer_params = {}
            kv_xfer_params["remote_num_cached_tokens"] = params["remote_num_cached_tokens"]

        return kv_xfer_params

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        outputs = super().update_from_output(scheduler_output, model_runner_output)  # type: ignore[misc]

        # Zero num_cache_creation_tokens on D-side: transferred KV is not
        # "cache creation".  finalize() already ran inside super(), so the
        # value is set — we override it here.
        for ecos in outputs.values():
            for eco in ecos.outputs:
                if eco.prefill_stats is not None and getattr(eco.prefill_stats, "_disagg_d_side", False):
                    eco.prefill_stats.num_cache_creation_tokens = 0

        return outputs
