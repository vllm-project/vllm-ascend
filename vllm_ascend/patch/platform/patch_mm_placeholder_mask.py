# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility backport of vllm-project/vllm#54548.

The placeholder builder follows upstream commit 60fe831acefe. The serving method
backports its call site onto the supported vLLM main pin ba07e4a48, with v0.27.1
content-parts, output-kind and session-ID differences gated explicitly.
"""

from collections.abc import Mapping
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import PlaceholderRangeInfo
    from vllm.multimodal.inputs import PlaceholderRange


def _wrap_extract_features(original, placeholder_type):
    @wraps(original)
    def extract(engine_input):
        features = original(engine_input)
        if features is None:
            return None
        features.mm_placeholders = {
            modality: [
                placeholder_type(
                    offset=p.offset,
                    length=p.length,
                    is_embed=None if p.is_embed is None else p.is_embed.tolist(),
                )
                for p in ranges
            ]
            for modality, ranges in engine_input["mm_placeholders"].items()
        }
        return features

    return extract


def rebuild_mm_placeholders(
    mm_placeholders: Mapping[str, list["PlaceholderRangeInfo"]],
) -> dict[str, list["PlaceholderRange"]]:
    """Convert rendered placeholders back to ranges, as in vLLM #54548."""
    import torch
    from vllm.multimodal.inputs import PlaceholderRange

    return {
        modality: [
            PlaceholderRange(
                offset=p.offset,
                length=p.length,
                is_embed=None if p.is_embed is None else torch.tensor(p.is_embed, dtype=torch.bool),
            )
            for p in ranges
        ]
        for modality, ranges in mm_placeholders.items()
    }


async def serve_tokens(self, request, raw_request=None):
    # Keep imports lazy: this patch module can load during platform discovery,
    # before vLLM config and serving modules have finished initializing.
    import msgspec
    from vllm.entrypoints.chat_utils import AsyncMultiModalItemTracker
    from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
    from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import decode_mm_kwargs_item
    from vllm.entrypoints.scale_out.token_in_token_out.serving import logger
    from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
    from vllm.inputs import TokensPrompt, mm_input
    from vllm.multimodal.inputs import MultiModalKwargsItems
    from vllm.sampling_params import RequestOutputKind

    from vllm_ascend.utils import vllm_version_is

    is_v0271 = vllm_version_is("0.27.1")
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        logger.error("Error with model %s", error_check_ret)
        return error_check_ret

    # Preserve upstream validation before constructing or scheduling inputs.
    if self.engine_client.errored:
        raise self.engine_client.dead_error

    lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)
    model_name = self.models.model_name(lora_request)
    request_id = f"generate-tokens-{self._base_request_id(raw_request, request.request_id)}"
    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    sampling_params = request.sampling_params
    max_num_seqs = self.engine_client.vllm_config.scheduler_config.max_num_seqs
    if sampling_params.n > max_num_seqs:
        return self.create_error_response(
            f"sampling_params.n must be at most the server's max_num_seqs ({max_num_seqs}), got {sampling_params.n}."
        )
    try:
        msgspec.msgpack.encode(sampling_params)
    except (OverflowError, TypeError, ValueError) as e:
        return self.create_error_response(e)

    if not is_v0271 and request.content_parts:
        tracker = AsyncMultiModalItemTracker(self.model_config)
        mm_parser = tracker.create_parser()
        for part in request.content_parts:
            ptype = part.get("type", "")
            url = part.get("url")
            uuid = part.get("uuid")
            if ptype == "image_url":
                mm_parser.parse_image(url, uuid)
            elif ptype == "audio_url":
                mm_parser.parse_audio(url, uuid)
            elif ptype == "video_url":
                mm_parser.parse_video(url, uuid)
        mm_data, mm_uuids = await tracker.resolve_items()
        prompt = TokensPrompt(prompt_token_ids=request.token_ids)
        if mm_data:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids:
            prompt["multi_modal_uuids"] = mm_uuids
        (engine_input,) = await self.online_renderer.renderer.render_cmpl_async([prompt])
    elif features := request.features:
        mm_placeholders = rebuild_mm_placeholders(features.mm_placeholders)

        # Deserialize tensor data when present; None means an encoder-cache hit.
        mm_kwargs = {}
        if features.kwargs_data is not None:
            for modality, items in features.kwargs_data.items():
                mm_kwargs[modality] = [decode_mm_kwargs_item(item) if item is not None else None for item in items]
        else:
            for modality, hashes in features.mm_hashes.items():
                mm_kwargs[modality] = [None] * len(hashes)

        engine_input = mm_input(
            prompt_token_ids=request.token_ids,
            mm_kwargs=MultiModalKwargsItems(mm_kwargs),
            mm_hashes=features.mm_hashes,
            mm_placeholders=mm_placeholders,
            cache_salt=request.cache_salt,
        )
    else:
        (engine_input,) = await self.online_renderer.preprocess_completion(
            request, prompt_input=request.token_ids, prompt_embeds=None, skip_mm_cache=True
        )

    # Retain upstream defaults, logging, routing and response generators.
    if not request.is_sampling_param_provided("max_tokens"):
        sampling_params.max_tokens = get_max_tokens(
            max_model_len=self.model_config.max_model_len,
            max_tokens=None,
            input_length=self._extract_prompt_len(engine_input),
            default_sampling_params=self.default_sampling_params,
            override_max_tokens=self.override_max_tokens,
        )

    if self.force_no_detokenize:
        sampling_params.detokenize = False
    if request.stream:
        sampling_params.output_kind = RequestOutputKind.DELTA
    elif not is_v0271:
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

    self._log_inputs(request_id, engine_input, params=sampling_params, lora_request=lora_request)
    trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)
    data_parallel_rank = self._get_data_parallel_rank(raw_request)
    # Session routing is not part of the v0.27.1 engine-client interface.
    session_kwargs = {} if is_v0271 else {"session_id": self._get_session_id_from_headers(raw_request)}
    result_generator = self.engine_client.generate(
        engine_input,
        sampling_params,
        request_id,
        lora_request=lora_request,
        trace_headers=trace_headers,
        priority=request.priority,
        data_parallel_rank=data_parallel_rank,
        **session_kwargs,
    )
    assert result_generator is not None
    if request.stream:
        return self.serve_tokens_stream_generator(request, result_generator, request_id, model_name, request_metadata)
    return await self.serve_tokens_full_generator(request, result_generator, request_id, model_name, request_metadata)


def install_mm_placeholder_mask_patch() -> None:
    # Run after platform registration, not during config/plugin module imports.
    from pydantic.fields import FieldInfo
    from vllm.entrypoints.scale_out.derender.serving import ServingDerender
    from vllm.entrypoints.scale_out.render.serving import ServingRender
    from vllm.entrypoints.scale_out.token_in_token_out import protocol
    from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens

    placeholder_type = protocol.PlaceholderRangeInfo
    if "is_embed" in placeholder_type.model_fields:
        return

    # Preserve class identities already imported by serving modules and routers.
    placeholder_type.model_fields["is_embed"] = FieldInfo(annotation=list[bool] | None, default=None)
    for model in (placeholder_type, protocol.MultiModalFeatures, protocol.GenerateRequest):
        model.model_rebuild(force=True)
    for serving in (ServingRender, ServingDerender):
        serving._extract_mm_features = staticmethod(
            _wrap_extract_features(serving._extract_mm_features, placeholder_type)
        )
    ServingTokens.serve_tokens = serve_tokens
