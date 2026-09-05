# SPDX-License-Identifier: Apache-2.0
"""Compatibility backport of vllm-project/vllm#54548."""

from functools import wraps
from types import FunctionType


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


def _wrap_serve_tokens(original):
    @wraps(original)
    async def serve(self, request, raw_request=None):
        if request.features is None or getattr(request, "content_parts", None):
            return await original(self, request, raw_request)

        import torch
        from vllm.multimodal.inputs import PlaceholderRange

        original_mm_input = original.__globals__["mm_input"]

        def mm_input(**kwargs):
            kwargs["mm_placeholders"] = {
                modality: [
                    PlaceholderRange(
                        offset=p.offset,
                        length=p.length,
                        is_embed=None if p.is_embed is None else torch.tensor(p.is_embed, dtype=torch.bool),
                    )
                    for p in ranges
                ]
                for modality, ranges in request.features.mm_placeholders.items()
            }
            return original_mm_input(**kwargs)

        # Bind only this request's input builder without mutating module globals
        # across concurrent awaits or copying the version-specific async handler.
        scoped = FunctionType(
            original.__code__,
            dict(original.__globals__, mm_input=mm_input),
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        scoped.__kwdefaults__ = original.__kwdefaults__
        return await scoped(self, request, raw_request)

    return serve


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
    ServingTokens.serve_tokens = _wrap_serve_tokens(ServingTokens.serve_tokens)
