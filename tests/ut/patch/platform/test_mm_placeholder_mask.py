# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from vllm.entrypoints.scale_out.derender.serving import ServingDerender
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.scale_out.token_in_token_out import serving as upstream_serving
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest, MultiModalFeatures
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.inputs import mm_input
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange

from vllm_ascend.patch.platform.patch_mm_placeholder_mask import (
    install_mm_placeholder_mask_patch,
    rebuild_mm_placeholders,
    serve_tokens,
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


def make_request(mask=None, **kwargs):
    install_mm_placeholder_mask_patch()
    return GenerateRequest.model_validate(
        {
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {"max_tokens": 8},
            "features": {
                "mm_hashes": {"image": ["test"]},
                "mm_placeholders": {"image": [{"offset": 0, "length": 4, "is_embed": mask}]},
            },
            **kwargs,
        }
    )


def make_serving():
    async def full_response(request, result, *args):
        await asyncio.sleep(0)
        return result

    return SimpleNamespace(
        _check_model=AsyncMock(return_value=None),
        engine_client=SimpleNamespace(
            errored=False,
            vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4)),
            generate=Mock(side_effect=lambda engine_input, *args, **kwargs: engine_input),
        ),
        _maybe_get_adapters=Mock(return_value=None),
        models=SimpleNamespace(model_name=lambda _: "test"),
        _base_request_id=lambda *a: "test",
        force_no_detokenize=False,
        _log_inputs=Mock(),
        _get_data_parallel_rank=Mock(return_value=2),
        _get_session_id_from_headers=Mock(return_value="session-test"),
        serve_tokens_full_generator=full_response,
        serve_tokens_stream_generator=Mock(side_effect=lambda request, result, *args: result),
        create_error_response=lambda error: error,
        online_renderer=SimpleNamespace(preprocess_completion=AsyncMock(return_value=({"type": "token"},))),
    )


@pytest.mark.parametrize("mask", [None, [True, False, True, True]])
def test_rebuild_mm_placeholders_restores_is_embed(mask):
    request = make_request(mask)
    features = MultiModalFeatures.model_validate_json(request.features.model_dump_json())
    (placeholder,) = rebuild_mm_placeholders(features.mm_placeholders)["image"]
    assert placeholder.offset == 0
    assert placeholder.length == 4
    if mask is None:
        assert placeholder.is_embed is None
    else:
        assert torch.equal(placeholder.is_embed, torch.tensor(mask, dtype=torch.bool))
        assert placeholder.is_embed.device.type == "cpu"
    assert placeholder.get_num_embeds() == (4 if mask is None else sum(mask))


def test_concurrent_requests_rebuild_their_own_embedding_masks():
    original_builder = upstream_serving.mm_input
    serving = make_serving()

    async def run():
        requests = [make_request(mask) for mask in ([True, False, True, True], [False, True, False, False], None)]
        return await asyncio.gather(*(ServingTokens.serve_tokens(serving, request) for request in requests))

    results = asyncio.run(run())
    assert [r["mm_placeholders"]["image"][0].get_num_embeds() for r in results] == [3, 1, 4]
    assert results[0]["mm_placeholders"]["image"][0].is_embed.dtype == torch.bool
    assert upstream_serving.mm_input is original_builder


def test_text_uses_original_preprocessing():
    serving = make_serving()
    request = make_request(features=None)
    assert asyncio.run(serve_tokens(serving, request)) == {"type": "token"}
    serving.online_renderer.preprocess_completion.assert_awaited_once_with(
        request, prompt_input=request.token_ids, prompt_embeds=None, skip_mm_cache=True
    )


@pytest.mark.skipif("content_parts" not in GenerateRequest.model_fields, reason="v0.27.1 has no content_parts")
def test_content_parts_uses_original_rendering():
    serving = make_serving()
    serving.model_config = object()
    serving.online_renderer.renderer = SimpleNamespace(
        render_cmpl_async=AsyncMock(return_value=({"type": "multimodal"},))
    )
    request = make_request(features=None, content_parts=[{"type": "image_url", "url": "image.png"}])
    tracker = Mock()
    tracker.resolve_items = AsyncMock(return_value=({"image": ["pixels"]}, {"image": ["image-id"]}))
    with patch("vllm.entrypoints.chat_utils.AsyncMultiModalItemTracker", return_value=tracker):
        assert asyncio.run(serve_tokens(serving, request)) == {"type": "multimodal"}
    tracker.create_parser.return_value.parse_image.assert_called_once_with("image.png", None)
    serving.online_renderer.renderer.render_cmpl_async.assert_awaited_once_with(
        [
            {
                "prompt_token_ids": request.token_ids,
                "multi_modal_data": {"image": ["pixels"]},
                "multi_modal_uuids": {"image": ["image-id"]},
            }
        ]
    )


@pytest.mark.parametrize("stream", [False, True])
def test_engine_handoff_preserves_routing_and_output_kind(stream):
    from vllm.sampling_params import RequestOutputKind

    serving = make_serving()
    request = make_request([True, False, True, True], stream=stream)
    old_output_kind = request.sampling_params.output_kind
    result = asyncio.run(ServingTokens.serve_tokens(serving, request))
    assert result["mm_placeholders"]["image"][0].get_num_embeds() == 3
    assert result["mm_kwargs"]["image"] == [None]
    kwargs = serving.engine_client.generate.call_args.kwargs
    assert kwargs["data_parallel_rank"] == 2
    if "content_parts" in GenerateRequest.model_fields:
        assert kwargs["session_id"] == "session-test"
        assert request.sampling_params.output_kind == (
            RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY
        )
    else:
        assert "session_id" not in kwargs
        assert request.sampling_params.output_kind == (RequestOutputKind.DELTA if stream else old_output_kind)


def test_model_error_does_not_schedule_request():
    serving = make_serving()
    serving._check_model.return_value = "model error"
    assert asyncio.run(serve_tokens(serving, make_request())) == "model error"
    serving.engine_client.generate.assert_not_called()


def test_sampling_validation_does_not_schedule_request():
    serving = make_serving()
    request = make_request(sampling_params={"n": 5, "max_tokens": 8})
    assert "max_num_seqs" in asyncio.run(serve_tokens(serving, request))
    serving.engine_client.generate.assert_not_called()


def test_default_max_tokens_and_no_detokenize_are_preserved():
    serving = make_serving()
    serving.model_config = SimpleNamespace(max_model_len=128)
    serving.default_sampling_params = {"max_tokens": 23}
    serving.override_max_tokens = None
    serving._extract_prompt_len = lambda _: 4
    serving.force_no_detokenize = True
    request = make_request(sampling_params={})
    asyncio.run(serve_tokens(serving, request))
    assert request.sampling_params.max_tokens == 23
    assert request.sampling_params.detokenize is False


def test_serialized_embedding_data_reaches_engine():
    from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import encode_mm_kwargs_item
    from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalFieldElem, MultiModalKwargsItem

    serving = make_serving()
    request = make_request([True, False, True, True])
    pixels = torch.arange(12).reshape(3, 4)
    item = MultiModalKwargsItem({"pixel_values": MultiModalFieldElem(data=pixels, field=MultiModalBatchedField())})
    request.features.kwargs_data = {"image": [encode_mm_kwargs_item(item)]}
    result = asyncio.run(serve_tokens(serving, request))
    assert torch.equal(result["mm_kwargs"]["image"][0]["pixel_values"].data, pixels)
    assert result["mm_placeholders"]["image"][0].get_num_embeds() == 3


def test_generation_exception_does_not_affect_next_request():
    serving = make_serving()
    serving.engine_client.generate.side_effect = RuntimeError("generation failed")
    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(serve_tokens(serving, make_request([True, False, True, True])))
    result = asyncio.run(serve_tokens(make_serving(), make_request([False, True, False, False])))
    assert result["mm_placeholders"]["image"][0].get_num_embeds() == 1
