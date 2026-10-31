# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Size the vLLM encoder cache by the worst-case per-request residency.

Upstream vLLM sizes the scheduler's encoder cache by the largest single
multimodal item (``compute_mm_encoder_budget``): both the encoder compute
budget and the cache capacity are floored at ``max_tokens_per_item``. A
request whose items' embeddings sum beyond that floor (for example two large
images with ``--limit-mm-per-prompt '{"image": 2}'``) can never hold all of
its items' embeddings at the same time, so scheduling relies on a strict
pipeline: the earlier item's cache entry is only releasable once it is fully
consumed, and the later item can only allocate afterwards.

That pipeline deadlocks when a chunk boundary leaves the earlier item one
token short of being fully consumed. ``Scheduler._try_schedule_encoder_inputs``
schedules only up to ``start_pos - num_prefill_lookahead`` when
``can_allocate`` fails for the later item, so the residue token is never
scheduled (the next step hits the ``num_new_tokens = 0`` branch), the earlier
item's slots are never released, and the request spins in permanent empty
steps (``model_executed=False`` with pending requests) until the client times
out. Observed on Kimi-K3: two 2560x2644 images expand to ~8740 embeddings per
item (17480 total) while the cache is floored at 16817, and the request
retries ~800k empty steps in 20 minutes.

Fix: after ``MultiModalBudget`` computes its budgets, raise
``encoder_cache_size`` to the worst-case single-request residency (sum over
modalities of ``max_tokens_per_item * mm_limit``). The per-step compute budget
is left untouched, so single-step execution size is unchanged. For models that
admit one item per request (``vision_chunk=1`` and similar),
``total_per_request == max_tokens_per_item`` and behavior is unchanged.
Modalities with an unlimited (``None``) ``mm_limit`` keep the legacy
single-item sizing: their worst-case residency is unbounded, so we
conservatively avoid amplifying the cache for them.
"""

from vllm.multimodal.encoder_budget import MultiModalBudget

_orig_init = MultiModalBudget.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    total_per_request = 0
    for modality, max_tokens_per_item in self.mm_max_toks_per_item.items():
        mm_limit = self.mm_limits.get(modality, 1)
        if not mm_limit or mm_limit < 0:
            # None/unlimited: the worst-case residency is unbounded; keep the
            # legacy single-item sizing instead of amplifying without bound.
            mm_limit = 1
        total_per_request += max_tokens_per_item * mm_limit
    if total_per_request > self.encoder_cache_size:
        self.encoder_cache_size = total_per_request


MultiModalBudget.__init__ = _patched_init
