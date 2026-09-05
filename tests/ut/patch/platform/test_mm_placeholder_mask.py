# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.entrypoints.scale_out.derender.serving import ServingDerender
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest, MultiModalFeatures
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.inputs import mm_input
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange

from vllm_ascend.patch.platform.patch_mm_placeholder_mask import (
    _wrap_serve_tokens,
    install_mm_placeholder_mask_patch,
)


@pytest.mark.parametrize("serving", [ServingRender, ServingDerender])
@pytest.mark.parametrize("mask", [None, [True, False, True, True]])
def test_render_json_preserves_sparse_and_dense_placeholders(serving, mask):
    install_mm_placeholder_mask_patch()
    engine_input = mm_input(
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_kwargs=MultiModalKwargsItems({}),
        mm_hashes={"image": ["image-0"]},
        mm_placeholders={
            "image": [PlaceholderRange(offset=1, length=4, is_embed=None if mask is None else torch.tensor(mask))]
        },
    )
    features = serving._extract_mm_features(engine_input)
    request = GenerateRequest.model_validate_json(
        GenerateRequest(token_ids=[1, 2, 3, 4, 5], sampling_params={}, features=features).model_dump_json()
    )
    assert request.features.mm_placeholders["image"][0].is_embed == mask
    assert serving._extract_mm_features({"type": "token", "prompt_token_ids": [1]}) is None


def test_installation_is_idempotent():
    install_mm_placeholder_mask_patch()
    render = ServingRender._extract_mm_features
    serve = ServingTokens.serve_tokens
    install_mm_placeholder_mask_patch()
    assert ServingRender._extract_mm_features is render
    assert ServingTokens.serve_tokens is serve
    assert "is_embed" in GenerateRequest.model_json_schema()["$defs"]["PlaceholderRangeInfo"]["properties"]


def test_fastapi_request_schema_preserves_embedding_mask():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    install_mm_placeholder_mask_patch()
    app = FastAPI()

    @app.post("/generate")
    def generate(request: GenerateRequest):
        return request.features

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "token_ids": [1, 2, 3, 4],
                "sampling_params": {},
                "features": {
                    "mm_hashes": {"image": ["test"]},
                    "mm_placeholders": {"image": [{"offset": 0, "length": 4, "is_embed": [True, False, True, True]}]},
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["mm_placeholders"]["image"][0]["is_embed"] == [True, False, True, True]
        schema = client.get("/openapi.json").json()
        assert "is_embed" in schema["components"]["schemas"]["PlaceholderRangeInfo"]["properties"]


async def _upstream_handler(self, request, raw_request=None):
    await asyncio.sleep(0)
    return mm_input(
        prompt_token_ids=request.token_ids,
        mm_kwargs=MultiModalKwargsItems({}),
        mm_hashes=request.features.mm_hashes,
        mm_placeholders={"image": [PlaceholderRange(offset=0, length=4)]},
    )


def test_concurrent_requests_rebuild_their_own_embedding_masks():
    install_mm_placeholder_mask_patch()
    original_builder = _upstream_handler.__globals__["mm_input"]
    wrapped = _wrap_serve_tokens(_upstream_handler)

    async def run():
        requests = [
            SimpleNamespace(
                token_ids=[1, 2, 3, 4],
                features=MultiModalFeatures.model_validate(
                    {
                        "mm_hashes": {"image": [str(i)]},
                        "mm_placeholders": {"image": [{"offset": 0, "length": 4, "is_embed": mask}]},
                    }
                ),
            )
            for i, mask in enumerate(([True, False, True, True], [False, True, False, False], None))
        ]
        return await asyncio.gather(*(wrapped(None, request) for request in requests))

    results = asyncio.run(run())
    assert [r["mm_placeholders"]["image"][0].get_num_embeds() for r in results] == [3, 1, 4]
    assert results[0]["mm_placeholders"]["image"][0].is_embed.dtype == torch.bool
    assert _upstream_handler.__globals__["mm_input"] is original_builder


@pytest.mark.parametrize("content_parts", [None, [{"type": "image_url", "url": "image.png"}]])
def test_text_and_content_parts_use_the_original_handler(content_parts):
    async def original(self, request, raw_request):
        return request

    request = SimpleNamespace(features=None if content_parts is None else object(), content_parts=content_parts)
    assert asyncio.run(_wrap_serve_tokens(original)(None, request)) is request


def test_real_handler_passes_mask_to_engine_input_builder():
    install_mm_placeholder_mask_patch()
    request = GenerateRequest.model_validate(
        {
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["test"]},
                "mm_placeholders": {"image": [{"offset": 0, "length": 4, "is_embed": [True, False, True, True]}]},
            },
        }
    )

    async def check_model(request):
        return None

    serving = SimpleNamespace(
        _check_model=check_model,
        engine_client=SimpleNamespace(
            errored=False, vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))
        ),
        _maybe_get_adapters=lambda *a, **k: None,
        models=SimpleNamespace(model_name=lambda _: "test"),
        _base_request_id=lambda *a: "test",
    )

    class ReachedInputBuilder(Exception):
        pass

    def capture(**kwargs):
        assert kwargs["mm_placeholders"]["image"][0].get_num_embeds() == 3
        raise ReachedInputBuilder

    original = getattr(ServingTokens.serve_tokens, "__wrapped__", ServingTokens.serve_tokens)
    with patch.dict(original.__globals__, mm_input=capture), pytest.raises(ReachedInputBuilder):
        asyncio.run(ServingTokens.serve_tokens(serving, request))
