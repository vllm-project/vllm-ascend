# SPDX-License-Identifier: Apache-2.0
"""AISBench streaming client for the isolated single-node PD performance cases."""

from uuid import uuid4

from ais_bench.benchmark.models import VLLMCustomAPIChat


class PDCacheChat(VLLMCustomAPIChat):
    def __init__(self, *, stage, prefill_rank=None, replay_tail_tokens=0, **kwargs):
        # A second attempt could turn a cold miss into a cache hit.
        kwargs["retry"] = 1
        super().__init__(**kwargs)
        self.stage = stage
        self.replay_tail_tokens = replay_tail_tokens
        if prefill_rank is not None:
            self.headers["X-data-parallel-rank"] = str(prefill_rank)

    async def get_request_body(self, *args, **kwargs):
        body = await super().get_request_body(*args, **kwargs)
        if self.stage == "prefill":
            # Namespace every request, including identical prompts or shared templates.
            body["cache_salt"] = uuid4().hex
        return body

    async def _parse_usage(self, json_content, output):
        await super()._parse_usage(json_content, output)
        usage = json_content.get("usage")
        if usage:
            output.extra_details_data["prefill_cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            )

    async def stream_infer(self, request_body, output):
        await super().stream_infer(request_body, output)
        if not output.success:
            return
        if output.output_tokens != request_body["max_tokens"]:
            raise ValueError(f"Expected {request_body['max_tokens']} output tokens, got {output.output_tokens}")
        if self.stage == "warmup":
            return
        prompt = output.input_tokens
        cached = output.extra_details_data.get("prefill_cached_tokens")
        if not isinstance(cached, int) or not prompt or not 0 <= cached <= prompt:
            raise ValueError(f"Missing valid Prefill cache usage: {cached=} {prompt=}")
        self.logger.info(
            "PD %s: prefill cached %s/%s prompt tokens, output %s", self.stage, cached, prompt, output.output_tokens
        )
        if self.stage == "prefill" and cached != 0:
            raise ValueError(f"Cold prefill reused {cached}/{prompt} prompt tokens")
        if self.stage == "decode" and (not cached or prompt - cached > self.replay_tail_tokens):
            raise ValueError(
                f"Decode replay missed the warmed Prefill KV: {cached}/{prompt} cached tokens, "
                f"allowed aligned tail {self.replay_tail_tokens}. Check cache capacity and routing."
            )
