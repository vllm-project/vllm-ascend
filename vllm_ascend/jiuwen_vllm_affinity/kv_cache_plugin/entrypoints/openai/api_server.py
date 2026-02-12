from http import HTTPStatus

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.api_server import (
    base,
    chat,
    router,
)
from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import load_aware_call, with_cancellation

from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.entrypoints.openai.protocol import (
    ReleaseKvCacheRequest,
    ReleaseKvCacheResponse,
)


@router.post(
    "/release_kv_cache",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def release_kv_cache(request: ReleaseKvCacheRequest, raw_request: Request):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Chat Completions API")
    try:
        generator = await handler.release_kv_cache(request, raw_request)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)

    elif isinstance(generator, ReleaseKvCacheResponse):
        return JSONResponse(
            content=generator.model_dump(),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")
