#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""rare_chars collect / feed mixin for RuntimeGuardProcessor."""

from __future__ import annotations

import contextlib
from typing import Any

from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

logger = init_logger_ascend(__name__)


class RuntimeGuardRareMixin:
    """Guard-side rare_chars stash → Triton collect → async D2H feed."""

    def warmup_detectors(self) -> None:
        """Build detector artifacts that must exist before the first real step.

        Called from ModelRunner warmup / post-``load_model``. Currently builds
        the rare_chars per-tokenizer table when that detector is enabled.
        """
        try:
            tok = self._get_detector_tokenizer()
        except Exception:
            tok = None
        try:
            self.detectors.warmup_rare_chars(tok)
        except Exception:
            logger.exception("[runtime_guard] rare_chars warmup failed")

    def _finish_rare_chars_on_reap(self, reapable: list[str]) -> None:
        """Look-ahead flush for finished reqs whose rare feed pins have drained."""
        if not self.detectors.rare_chars_enabled():
            return
        store = RequestGuardStore.get()
        finishable, incomplete = store.partition_rare_finish_reapable(reapable)
        if incomplete:
            logger.warning(
                "[runtime_guard: rare_chars] skip finish after force-clear (incomplete feed) req_ids=%s",
                incomplete,
            )
        if finishable:
            for alert in self.detectors.finish_rare_chars(finishable):
                self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))

    def check_rare_chars_from_stash(
        self,
        *,
        sampled_token_ids: Any,
        num_sampled: Any = None,
    ) -> None:
        """After-sample rare_chars collect from stashed pre-sample logits."""
        from vllm_ascend.observability.runtime_guard.hooks_rare import take_pre_sample_logits

        if not self.detectors.rare_chars_enabled():
            take_pre_sample_logits(self.runner)
            return

        logits = take_pre_sample_logits(self.runner)
        if logits is None:
            return
        self._schedule_rare_collect(logits, sampled_token_ids, num_sampled=num_sampled)

    @staticmethod
    def _mask_chunked_prefill_tokens(sampled_token_ids: Any, num_sampled: Any) -> Any:
        """Force ``-1`` on rows with ``num_sampled==0`` (chunked-prefill phantoms)."""
        import torch

        if num_sampled is None or not isinstance(sampled_token_ids, torch.Tensor):
            return sampled_token_ids
        if isinstance(num_sampled, torch.Tensor):
            ns = num_sampled.detach().reshape(-1)
        else:
            try:
                ns = torch.as_tensor(num_sampled, device=sampled_token_ids.device).reshape(-1)
            except Exception:
                return sampled_token_ids
        if ns.numel() != sampled_token_ids.shape[0]:
            return sampled_token_ids
        dead = ns <= 0
        if not bool(dead.any().item() if hasattr(dead, "any") else any(dead)):
            return sampled_token_ids
        out = sampled_token_ids.clone()
        out[dead] = -1
        return out

    def _note_rare_stream_gap(self, req_ids: list[str] | None = None) -> None:
        """Reset rare_chars UTF-8 decoder state after a skipped feed step."""
        detector = self.detectors.get("rare_chars")
        note = getattr(detector, "note_stream_gap", None) if detector is not None else None
        if not callable(note):
            return
        ids = req_ids
        if not ids:
            batch = getattr(self, "_last_input_batch", None)
            ids = list(getattr(batch, "req_ids", None) or []) if batch is not None else []
        if ids:
            note(ids)

    def _schedule_rare_collect(
        self,
        logits: Any,
        sampled_token_ids: Any,
        *,
        num_sampled: Any = None,
    ) -> None:
        """Launch (top1, sampled) on main; D2H+feed deferred to RareLogprobFeedQueue."""
        from vllm_ascend.observability.runtime_guard.hooks import is_async_output_rank

        if not is_async_output_rank():
            return
        try:
            if not get_pp_group().is_last_rank:
                return
        except Exception:
            pass
        import torch

        from vllm_ascend.observability.runtime_guard.detector.rare.narrow_collect import (
            compute_top1_and_chosen_logprobs,
        )
        from vllm_ascend.observability.runtime_guard.detector.rare_chars import (
            align_sampled_token_ids_to_logits,
        )

        if not isinstance(logits, torch.Tensor) or not isinstance(sampled_token_ids, torch.Tensor):
            return
        if logits.numel() == 0 or logits.dim() != 2:
            return
        try:
            sampled_token_ids = self._mask_chunked_prefill_tokens(sampled_token_ids, num_sampled)
            cu_num_logits = self._rare_cu_num_logits(num_logit_rows=int(logits.shape[0]))
            num_logit_rows = int(logits.shape[0])
            aligned = align_sampled_token_ids_to_logits(
                sampled_token_ids,
                num_logit_rows,
                cu_num_logits=cu_num_logits,
            )
            if aligned is None:
                # Layout miss: skip this step's collect (fail closed). Rate-limited
                # so MTP/adaptive shape bugs are visible without per-step spam.
                samp_shape = tuple(sampled_token_ids.shape)
                cu_tail = None
                if cu_num_logits is not None:
                    try:
                        cu_list = list(cu_num_logits)
                        cu_tail = (len(cu_list), cu_list[-1] if cu_list else None)
                    except TypeError:
                        cu_tail = type(cu_num_logits).__name__
                logger.warning_once(
                    "[runtime_guard: rare_chars] align miss; skip collect this step "
                    "(enabled but no detection for mismatched layouts). "
                    "logit_rows=%s sampled_shape=%s cu_num_logits=%s",
                    num_logit_rows,
                    samp_shape,
                    cu_tail,
                )
                # Skipped feed creates a UTF-8 stream hole; reset decoder state.
                self._note_rare_stream_gap()
                return
            token_ids, slots_per_req, cu_aligned = aligned
            packed = compute_top1_and_chosen_logprobs(logits, token_ids, packed=True)
            if slots_per_req > 1:
                logger.info_once("[runtime_guard: rare_chars] MTP-aligned collect enabled (slots_per_req>1)")
            elif cu_aligned is not None:
                logger.info_once("[runtime_guard: rare_chars] adaptive/cu_num_logits collect enabled")
            self._stash_rare_batch(
                packed,
                token_ids=token_ids,
                slots_per_req=slots_per_req,
                cu_num_logits=cu_aligned,
            )
        except Exception:
            logger.exception("[runtime_guard: rare_chars] collect failed")

    def _rare_cu_num_logits(self, *, num_logit_rows: int | None = None) -> Any | None:
        """Best-effort ``cu_num_logits`` from the current sample-phase batch.

        Prefer the device tensor when the host ``cu_num_logits_np`` is stale
        (adaptive verification keeps true boundaries on device only).
        """
        batch = getattr(self, "_last_input_batch", None)
        if batch is None:
            batch = getattr(self.runner, "input_batch", None)
        if batch is None:
            return None
        cu_np = getattr(batch, "cu_num_logits_np", None)
        cu_dev = getattr(batch, "cu_num_logits", None)
        if cu_np is not None and num_logit_rows is not None:
            try:
                tail = int(cu_np[-1]) if len(cu_np) else -1
            except Exception:
                tail = -1
            if tail != int(num_logit_rows):
                # Stale host mirror under adaptive verification — use device.
                if cu_dev is not None:
                    return cu_dev
                return None
            return cu_np
        if cu_np is not None:
            return cu_np
        return cu_dev

    def _stash_rare_batch(
        self,
        packed: Any,
        *,
        token_ids: Any | None = None,
        slots_per_req: int = 1,
        cu_num_logits: list[int] | None = None,
    ) -> None:
        """Stash this step's :class:`RareLogprobBatch` for AsyncOutput."""
        import torch

        from vllm_ascend.observability.runtime_guard.rare_logprob_feed import RareLogprobBatch

        leaked = getattr(self, "_step_rare_batch", None)
        if leaked is not None:
            if leaked.ready_event is not None:
                leaked.ready_event.synchronize()
            del leaked
            logger.warning("[runtime_guard: rare_chars] dropped unconsumed step rare batch")
        main_stream = getattr(self.runner, "main_stream", None)
        ready_event = torch.cuda.Event(blocking=True)
        if main_stream is not None:
            ready_event.record(main_stream)
        else:
            ready_event.record()
        tok = None
        if isinstance(token_ids, torch.Tensor):
            tok = token_ids.detach()
        self._step_rare_batch = RareLogprobBatch(
            device_logprobs=packed,
            ready_event=ready_event,
            device_token_ids=tok,
            slots_per_req=int(slots_per_req),
            cu_num_logits=list(cu_num_logits) if cu_num_logits is not None else None,
        )

    def take_step_rare_batch(self) -> Any | None:
        """Detach this step's :class:`RareLogprobBatch` for AsyncOutput."""
        pending = getattr(self, "_step_rare_batch", None)
        self._step_rare_batch = None
        return pending

    def _copy_rare_batch_to_host(self, batch: Any) -> tuple[Any, Any] | None:
        """Wait for collect, then D2H pairs (+ optional aligned token ids)."""
        import torch
        from vllm.v1.worker.gpu.async_utils import stream as copy_stream_ctx

        packed = batch.device_logprobs
        token_ids = getattr(batch, "device_token_ids", None)
        ready_event = batch.ready_event
        if ready_event is not None:
            ready_event.synchronize()
        if packed is None:
            return None
        runner = self.runner
        copy_stream = getattr(runner, "output_copy_stream", None)
        main_stream = getattr(runner, "main_stream", None)
        cpu = torch.empty(packed.shape, dtype=torch.float32, device="cpu", pin_memory=True)
        cpu_tok = None
        if isinstance(token_ids, torch.Tensor):
            cpu_tok = torch.empty(token_ids.shape, dtype=torch.int64, device="cpu", pin_memory=True)
        if copy_stream is None or main_stream is None:
            cpu.copy_(packed)
            if cpu_tok is not None:
                cpu_tok.copy_(token_ids)
            return cpu, cpu_tok
        copy_ev = torch.cuda.Event(blocking=True)
        with copy_stream_ctx(copy_stream, main_stream):
            cpu.copy_(packed, non_blocking=True)
            if cpu_tok is not None:
                cpu_tok.copy_(token_ids, non_blocking=True)
            copy_ev.record(copy_stream)
        copy_ev.synchronize()
        return cpu, cpu_tok

    def feed_rare_logprobs(self, output: Any, *, batch: Any | None = None) -> None:
        """Submit rare logprob D2H + detector feed onto RareLogprobFeedQueue."""
        if batch is None:
            return
        req_ids = [str(rid) for rid in (getattr(output, "req_ids", None) or []) if rid]
        n_req = len(req_ids)
        sampled = getattr(output, "sampled_token_ids", None)
        slots_per_req = int(getattr(batch, "slots_per_req", 1) or 0)
        cu_num_logits = getattr(batch, "cu_num_logits", None)

        if n_req <= 0:
            if batch.ready_event is not None:
                batch.ready_event.synchronize()
            del batch
            return

        pinned = tuple(rid for rid in req_ids[:n_req] if rid)
        store = RequestGuardStore.get()
        pinned_live = False
        try:
            if pinned:
                store.add_rare_feed_jobs(pinned)
                pinned_live = True

            def _finish_feed(
                rare_batch=batch,
                req_ids=req_ids,
                n_req=n_req,
                sampled=sampled,
                pinned_ids=pinned,
                slots_per_req=slots_per_req,
                cu_num_logits=cu_num_logits,
            ) -> None:
                try:
                    copied = self._copy_rare_batch_to_host(rare_batch)
                    del rare_batch
                    if copied is not None and n_req > 0:
                        cpu, cpu_tok = copied
                        self._pack_and_feed_rare(
                            cpu,
                            sampled,
                            req_ids,
                            n_req,
                            host_token_ids=cpu_tok,
                            slots_per_req=slots_per_req,
                            cu_num_logits=cu_num_logits,
                        )
                except Exception:
                    logger.exception("[runtime_guard: rare_chars] async feed finish failed")
                finally:
                    if pinned_ids:
                        RequestGuardStore.get().finish_rare_feed_jobs(pinned_ids)
                    # Idle servers: last pin drop must still flush finish/look-ahead.
                    with contextlib.suppress(Exception):
                        self._reap_finished_requests()

            from vllm_ascend.observability.runtime_guard.rare_logprob_feed import (
                get_rare_logprob_feed_queue,
            )

            ok = bool(get_rare_logprob_feed_queue().submit(_finish_feed))
            if not ok:
                if pinned_live:
                    store.finish_rare_feed_jobs(pinned)
                    pinned_live = False
                if batch.ready_event is not None:
                    batch.ready_event.synchronize()
                del batch
                self._note_rare_stream_gap(req_ids)
                logger.warning(
                    "[runtime_guard: rare_chars] feed queue full; dropped sync+feed req_ids=%s",
                    req_ids,
                )
                return
            pinned_live = False  # ownership transferred to the feed job
        except Exception:
            if pinned_live:
                with contextlib.suppress(Exception):
                    store.finish_rare_feed_jobs(pinned)
            if batch.ready_event is not None:
                with contextlib.suppress(Exception):
                    batch.ready_event.synchronize()
            del batch
            self._note_rare_stream_gap(req_ids)
            logger.exception("[runtime_guard: rare_chars] feed submit failed")

    def _pack_and_feed_rare(
        self,
        cpu: Any,
        sampled: Any,
        req_ids: list[str],
        n_req: int,
        *,
        host_token_ids: Any | None = None,
        slots_per_req: int = 1,
        cu_num_logits: list[int] | None = None,
    ) -> None:
        try:
            from vllm_ascend.observability.runtime_guard.detector.rare_chars import build_rare_feed_arrays

            token_ids, top1, chosen, expanded_req_ids = build_rare_feed_arrays(
                cpu,
                sampled,
                n_req,
                host_token_ids=host_token_ids,
                slots_per_req=slots_per_req,
                req_ids=req_ids,
                cu_num_logits=cu_num_logits,
            )
        except Exception:
            logger.exception("[runtime_guard: rare_chars] feed array materialize failed")
            return
        n = int(token_ids.shape[0])
        if n <= 0:
            return
        feed_req_ids = expanded_req_ids if expanded_req_ids is not None else req_ids[:n]
        detector = self.detectors.get("rare_chars")
        if detector is None:
            return
        try:
            for alert in detector.feed_precomputed(
                req_ids=feed_req_ids[:n],
                token_ids=token_ids,
                top1_logprobs=top1,
                chosen_logprobs=chosen,
            ):
                self._handle_alert(alert, detector=detector)
        except Exception:
            logger.exception("[runtime_guard: rare_chars] feed_precomputed failed")
