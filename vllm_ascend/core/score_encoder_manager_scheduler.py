from dataclasses import dataclass
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict

from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.logger import init_logger
from vllm.v1.request import Request
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_ascend.ascend_config import ScoreEncoderCacheConfig, get_ascend_config

logger = init_logger(__name__)

@dataclass
class CacheEntry:
    mm_hash: str        # Unique identifier of the multimodal input
    freq: int           # Access frequency
    clock: int          # Clock value used for aging
    num_embeds: int     # Number of slots occupied by this embedding
    cal_cost: int       # Theoretical recomputation cost of this embedding (used for score calculation)


class ScoreEncoderCacheManager(EncoderCacheManager):
    """
    Score-based encoder cache manager.

    The overall structure is a two-level cache:
        NPU cache (fast / small capacity)
        CPU cache (slower / large capacity)

    Core strategy:
    1. Newly generated encoder embeddings are first placed into the CPU cache
    2. If an entry is accessed frequently enough and has a sufficiently high score,
       it can be promoted to the NPU cache
    3. When the NPU cache runs out of space, entries with the lowest scores are evicted
    4. A clock-based aging mechanism is used to prevent stale hot entries
       from occupying the cache for too long
    """
    def __init__(self, cache_size: int, vllm_config: VllmConfig):
        super().__init__(cache_size)
        score_encoder_cache_config = get_ascend_config().score_encoder_cache_config

        # ---------------- NPU cache ----------------
        self.npu_num_free_slots = cache_size    # Empty slots
        self.npu_num_freeable_slots = cache_size    # Reclaimable capacity: reclaimable slots + empty slots

        # ---------------- CPU cache ----------------
        self.cpu_cache_size = score_encoder_cache_config.cpu_cache_slots
        self.cpu_num_free_slots = self.cpu_cache_size
        self.cpu_num_freeable_slots = self.cpu_cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}

        # Actual cache contents
        self.npu_cache: Dict[str, CacheEntry] = {}
        self.cpu_cache: Dict[str, CacheEntry] = {}

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        # Evictable cache entries (entries not referenced by any request)
        self.npu_freeable: Dict[str, CacheEntry] = {}
        self.cpu_freeable: OrderedDict[str, CacheEntry] = OrderedDict()

        # mm_hashes evicted in the previous round; after NPU eviction they may be placed into CPU,
        # and after CPU eviction they may also be recorded here

        self.req_cnt = 0

        self.watermark = score_encoder_cache_config.watermark
        self.promote_percentile = score_encoder_cache_config.promote_percentile
        self.max_clock = score_encoder_cache_config.max_clock
        self.clock_decay_every = score_encoder_cache_config.clock_decay_every

        # Actions to execute in the current round
        self.promoting: list[str] = []             # mm_hashes to be promoted from CPU -> NPU
        self.cpu_get_encoder_mm_hashes: list[str] = []  # mm_hashes whose embeddings need to be prefetched from CPU

        self.attn_heads = vllm_config.model_config.hf_config.vision_config.num_heads
        self.hidden_size = vllm_config.model_config.hf_config.vision_config.hidden_size
        self.feedforward = vllm_config.model_config.hf_config.vision_config.intermediate_size

        # Hardware throughput (FLOPs)
        self.hardware_flops = 4 * 1e14

        # TODO: there may be more kinds of compute ways
        # Coefficients used to estimate the compute cost of encoder embeddings
        self.alpha = 4 * self.hidden_size + 5 * self.attn_heads
        self.beta = self.hidden_size * (8 * self.hidden_size + 6 * self.feedforward + 14)

        self.stats = EmbCacheStats()

    def score(self, ent: CacheEntry) -> float:
        return (ent.freq + ent.clock) * ent.cal_cost

    def evict_from_npu(self, ent: CacheEntry):
        """
        Evict an entry from the NPU cache.
        """
        del self.npu_cache[ent.mm_hash]
        self.freed.append(ent.mm_hash)
        self.npu_num_free_slots += ent.num_embeds

        self.stats.evict_npu += 1
        self.stats.npu_freed_entries += 1

        logger.info(f"evict_from_npu check")
        self._check_invariant()

    def should_promote(self, mm_hash: str) -> bool:
        """
        Determine whether an entry in the CPU cache should be promoted to the NPU cache.

        Logic:
        1. If the NPU has enough free space, promote directly
        2. If space is insufficient, decide based on the score percentile
        3. If needed, evict lower-score entries from the NPU cache
        """
        ent = self.cpu_cache[mm_hash]

        # No reclaimable space on the NPU, promotion is impossible
        if ent.num_embeds > self.npu_num_freeable_slots:
            self.stats.promote_fail_no_space += 1
            return False

        if ent.num_embeds <= self.npu_num_free_slots:
            # The NPU has free space, place it directly
            return True

        ent_value = self.score(ent)
        scored = []
        for cur_hash, cur_ent in self.npu_freeable.items():
            value = self.score(cur_ent)
            scored.append((value, cur_hash, cur_ent))

        scored.sort(key=lambda x: x[0])
        idx = max(0, min(len(scored) - 1, int(len(scored) * self.promote_percentile)))

        threshold = scored[idx][0]
        if ent_value < threshold:
            self.stats.promote_fail_low_score += 1
            return False

        free_slots = max(self.cache_size * self.watermark - self.npu_num_free_slots,
                         ent.num_embeds - self.npu_num_free_slots)

        i = 0
        while free_slots > 0:
            min_hash = scored[i][1]
            ent = self.npu_freeable.pop(min_hash)
            self.evict_from_npu(ent)
            i += 1
            free_slots -= ent.num_embeds

        return True

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """
        Check whether the multimodal embedding corresponding to the current input
        is already cached. If so, update reference tracking, access statistics,
        and hotness information.

        Returns:
            bool:
                True  indicates a cache hit and no need to recompute the encoder output
                False indicates a cache miss and the encoder must be recomputed
        """
        mm_hash = request.mm_features[input_id].identifier

        # Not cached at all
        if mm_hash not in self.cached:
            self.stats.total_requests += 1
            self.on_request()
            self.stats.cache_misses += 1
            return False

        if not self.cached[mm_hash]:
            if mm_hash in self.npu_freeable:
                ent = self.npu_freeable.pop(mm_hash)
                self.npu_num_freeable_slots -= ent.num_embeds
            if mm_hash in self.cpu_freeable:
                ent = self.cpu_freeable.pop(mm_hash)
                self.cpu_num_freeable_slots -= ent.num_embeds
            

        if request.request_id not in self.cached[mm_hash]:
            self.cached[mm_hash].add(request.request_id)
            self.stats.total_requests += 1
            self.stats.cache_hits += 1
            ent = None
            if mm_hash in self.npu_cache:
                ent = self.npu_cache[mm_hash]
                self.stats.npu_hits += 1
            else:
                self.stats.cpu_hits += 1

                if self.should_promote(mm_hash):
                    # Promote
                    ent = self.cpu_cache[mm_hash]
                    self.npu_cache[mm_hash] = ent
                    self.npu_num_free_slots -= ent.num_embeds
                    self.npu_num_freeable_slots -= ent.num_embeds
                    self.promoting.append(mm_hash)

                    self.stats.promote_success += 1
                else:
                    self.cpu_get_encoder_mm_hashes.append(mm_hash)
                    ent = self.cpu_cache[mm_hash]

            self.on_request()
            ent.freq += 1
            ent.clock = self.max_clock

        return True

    def on_request(self):
        self.req_cnt += 1
        if self.req_cnt % self.clock_decay_every == 0:
            for ent in self.npu_cache.values():
                ent.clock = max(0, ent.clock - 1)

        if self.req_cnt % 1 == 0:
            self.emb_log_stats()

        if self.req_cnt % 1 == 0:
            self._check_invariant()

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """
        Determine whether CPU cache space can be allocated for the current input.

        Conditions:
        1. The encoder compute cost of the current input must not exceed the budget of this round
        2. The CPU cache must have enough available or reclaimable space
        3. If free space is insufficient, try evicting entries from CPU freeable

        Returns:
            bool: Whether allocation can be completed
        """

        num_embeds = request.get_num_encoder_embeds(input_id)

        # Not enough compute budget
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        if num_embeds > self.cpu_num_freeable_slots:
            return False

        while num_embeds > self.cpu_num_free_slots:
            mm_hash, ent = self.cpu_freeable.popitem(last=False)
            del self.cached[mm_hash]
            del self.cpu_cache[mm_hash]
            self.freed.append(mm_hash)
            self.cpu_num_free_slots += ent.num_embeds
            self.stats.cpu_evict_due_to_alloc += 1

        return True

    def cal_theory_cost_storage_cost(self, seq_len: int) -> float:
        """
        Compute the theoretical recomputation cost of an encoder output.

        The return value represents:
            A rough estimate of the time required to recompute the embedding
            (derived from FLOPs / hardware_flops)

        Notes:
        - The input parameter uses seq_len as an approximation of embedding size
        - The current formula is a rough theoretical estimate based on the vision encoder
        - b*s[(4h+5a)s +(14h+8h**2 +6h*ffn)]
        """

        cost = 32 * (self.alpha * seq_len + self.beta)
        return cost / self.hardware_flops

    def allocate(self, request: Request, input_id: int) -> None:
        """
        Allocate a CPU cache entry for the current input.

        Notes:
        - Newly computed encoder embeddings are placed into the CPU cache by default
        - This only updates the manager's metadata and does not involve actual tensor storage
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        cache_entry = CacheEntry(
                mm_hash=mm_hash,
                freq=1,
                clock=self.max_clock,
                num_embeds=num_encoder_embeds,
                cal_cost=self.cal_theory_cost_storage_cost(num_encoder_embeds),
            )

        assert self.cpu_num_free_slots >= num_encoder_embeds
        assert self.cpu_num_freeable_slots >= num_encoder_embeds

        self.cpu_num_free_slots -= num_encoder_embeds
        self.cpu_num_freeable_slots -= num_encoder_embeds

        assert mm_hash not in self.cpu_cache, f"mm_hash={mm_hash}"
        self.cpu_cache[mm_hash] = cache_entry

        self.cached[mm_hash].add(request_id)

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        logger.info(f"ZKX TEST: cached:{self.cached}")
        if self.cached[mm_hash]:
            return
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        if mm_hash in self.cpu_cache:
            # cpu_freeable_sum = sum(ent.num_embeds for ent in self.cpu_freeable.values())
            # logger.info(f"cpu freeable sum: {cpu_freeable_sum}, cpu freeable slots:{self.cpu_num_freeable_slots}, free_slots:{self.cpu_num_free_slots}")
            self.cpu_freeable[mm_hash] = self.cpu_cache[mm_hash]
            self.cpu_num_freeable_slots += num_encoder_embeds
        if mm_hash in self.npu_cache:
            npu_freeable_sum = sum(ent.num_embeds for ent in self.npu_freeable.values())
            logger.info(f"npu freeable sum: {npu_freeable_sum}, npu freeable slots:{self.npu_num_freeable_slots}, free_slots:{self.npu_num_free_slots}")
            self.npu_freeable[mm_hash] = self.npu_cache[mm_hash]
            self.npu_num_freeable_slots += num_encoder_embeds
            logger.info(f"cur embeds: {num_encoder_embeds}, free check")
            self._check_invariant()
        del self.cached[mm_hash]

    def get_promoting_mm_hashes(self) -> list[str]:
        promoting = self.promoting
        self.promoting = []
        return promoting

    def get_cpu_get_encoder_mm_hashes(self) -> list[str]:
        cpu_get_encoder_mm_hashes = self.cpu_get_encoder_mm_hashes
        self.cpu_get_encoder_mm_hashes = []
        return cpu_get_encoder_mm_hashes

    def _check_invariant(self):
        """
        Validate internal state.

        Main checks:
        1. Occupied cache slots + free slots = total capacity
        2. Free slots + slots occupied by freeable entries = freeable_slots
        3. Entries in freeable must not be referenced by any request
        """

        # ---------- CPU ----------
        cpu_sum = sum(ent.num_embeds for ent in self.cpu_cache.values())
        assert (cpu_sum + self.cpu_num_free_slots == self.cpu_cache_size), (
            f"cpu_sum + cpu_num_free_slots != cpu_cache_size, "
            f"cpu_sum={cpu_sum}, "
            f"cpu_num_free_slots={self.cpu_num_free_slots}, "
            f"cpu_cache_size={self.cpu_cache_size}"
        )

        cpu_freeable_sum = sum(ent.num_embeds for ent in self.cpu_freeable.values())
        assert (
            self.cpu_num_freeable_slots
            == self.cpu_num_free_slots + cpu_freeable_sum
        ), (
            f"CPU invariant broken: "
            f"freeable={self.cpu_num_freeable_slots}, "
            f"free={self.cpu_num_free_slots}, "
            f"freeable_sum={cpu_freeable_sum}"
        )

        for mm_hash in self.cpu_freeable:
            assert not self.cached.get(mm_hash), (
                f"CPU freeable entry {mm_hash} still referenced: "
                f"{self.cached.get(mm_hash)}"
            )

        # ---------- NPU ----------
        npu_sum = sum(ent.num_embeds for ent in self.npu_cache.values())
        assert (npu_sum + self.npu_num_free_slots == self.cache_size), (
            f"npu_sum + npu_num_free_slots != cache_size, "
            f"npu_sum={npu_sum}, "
            f"npu_num_free_slots={self.npu_num_free_slots}, "
            f"cache_size={self.cache_size}"
        )
        npu_freeable_sum = sum(ent.num_embeds for ent in self.npu_freeable.values())
        assert (
            self.npu_num_freeable_slots
            == self.npu_num_free_slots + npu_freeable_sum
        ), (
            f"NPU invariant broken: "
            f"freeable={self.npu_num_freeable_slots}, "
            f"free={self.npu_num_free_slots}, "
            f"freeable_sum={npu_freeable_sum}"
        )

        for mm_hash in self.npu_freeable:
            assert not self.cached.get(mm_hash), (
                f"NPU freeable entry {mm_hash} still referenced: "
                f"{self.cached.get(mm_hash)}"
            )

    def emb_log_stats(self):
        s = self.stats
        assert s.total_requests == self.req_cnt, f"total_requests={s.total_requests}, req_cnt={self.req_cnt}"

        hit_rate = s.cache_hits * 100 / max(1, s.total_requests)
        npu_hit_rate = s.npu_hits * 100 / max(1, s.total_requests)
        cpu_hit_rate = s.cpu_hits * 100 / max(1, s.total_requests)

        cpu_entries = len(self.cpu_cache)
        npu_entries = len(self.npu_cache)
        cpu_freeable_entries = len(self.cpu_freeable)
        npu_freeable_entries = len(self.npu_freeable)

        logger.info(
            "[EmbCacheStats] "
            "req=%d | hit=%d npu_hit=%d cpu_hit=%d | "
            "hit_rate=%.3f%% npu_hit_rate=%.3f%% cpu_hit_rate=%.3f%% | "
            "promote=%d/%d | "
            "evict(cpu=%d npu2cpu=%d due2alloc=%d freed=%d) | "
            "entries(cpu=%d freeable=%d | npu=%d freeable=%d) | "
            "slots(cpu=%d/%d npu=%d/%d)",
            s.total_requests,
            s.cache_hits,
            s.npu_hits,
            s.cpu_hits,
            hit_rate,
            npu_hit_rate,
            cpu_hit_rate,
            s.promote_success,
            s.promote_attempts,
            s.evict_cpu,
            s.evict_npu_to_cpu,
            s.cpu_evict_due_to_alloc,
            s.freed_entries,
            cpu_entries,
            cpu_freeable_entries,
            npu_entries,
            npu_freeable_entries,
            self.cpu_num_free_slots,
            self.cpu_num_freeable_slots,
            self.npu_num_free_slots,
            self.npu_num_freeable_slots,
        )


@dataclass
class EmbCacheStats:
    # ---- access ----
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cpu_hits: int = 0
    npu_hits: int = 0

    # ---- promote ----
    promote_attempts: int = 0
    promote_success: int = 0
    promote_fail_no_space: int = 0
    promote_fail_low_score: int = 0

    # ---- eviction ----
    evict_npu: int = 0
    evict_cpu: int = 0
    evict_npu_to_cpu: int = 0
    cpu_evict_due_to_alloc: int = 0
    freed_entries: int = 0
    npu_freed_entries: int = 0