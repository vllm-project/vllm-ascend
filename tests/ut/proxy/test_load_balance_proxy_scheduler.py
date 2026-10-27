import asyncio

import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("retry prefill failed"), asyncio.CancelledError()],
    ids=["prefill-error", "request-cancelled"],
)
def test_failed_reassignment_releases_previous_decoder_once(monkeypatch, failure):
    scheduler = proxy.SharedProxyScheduler(
        prefiller_instances=[("127.0.0.1", 19001)],
        decoder_instances=[("127.0.0.1", 19002)],
    )
    runtime = proxy.WorkerRuntime(scheduler)
    monkeypatch.setattr(proxy, "runtime", runtime)

    prefiller_score = 100.0
    decoder_score = 1000.0
    prefiller = scheduler.begin_request(prefiller_score)
    decoder = scheduler.pick_decoder(decoder_score)
    previous_instance = proxy.InstanceInfo(
        request_id="request-0",
        prefiller_key=prefiller["key"],
        prefiller_score=prefiller_score,
        decoder_key=decoder["key"],
        decoder_score=decoder_score,
        decoder_host=decoder["host"],
        decoder_port=decoder["port"],
    )

    async def fail_assignment(*args, **kwargs):
        raise failure

    monkeypatch.setattr(proxy, "assign_instances", fail_assignment)

    async def run_reassignment_and_cleanup():
        with pytest.raises(type(failure)):
            await proxy.reassign_instances(
                "/v1/chat/completions",
                {},
                128,
                previous_instance,
                previous_prefiller_kv_released=True,
            )
        await proxy._finish_instance(runtime, previous_instance, release_prefill_kv=False)

    asyncio.run(run_reassignment_and_cleanup())

    decoder_state = scheduler.decoders[decoder["key"]]
    assert decoder_state.active_tokens == 0.0
    assert scheduler.request_num == 0


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("mode", ["fragmented", "cancel", "recompute", "other-error", "null", "list", "string-error"])
def test_decode_transport_framing_errors_and_close(layerwise, mode):
    import json

    import httpx

    from examples.disaggregated_prefill_v1 import load_balance_proxy_layerwise_server_example

    transport_fn = (
        load_balance_proxy_layerwise_server_example.stream_service_response_with_retry
        if layerwise
        else proxy.stream_service_response
    )
    closed = []
    calls = []
    error = {"error": {"type": "RecomputeRequiredError", "code": 500}, "partial_response": {"choices": []}}
    body = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body[:7]
            yield body[7:]

        async def aclose(self):
            closed.append(True)

    def upstream(request):
        calls.append(request)
        if mode == "recompute":
            return httpx.Response(500, json=error)
        if mode == "other-error":
            return httpx.Response(500, content=b"upstream unavailable")
        if mode in ("null", "list", "string-error"):
            return httpx.Response(
                500, content=json.dumps({"null": None, "list": [], "string-error": {"error": "bad"}}[mode]).encode()
            )
        return httpx.Response(200, stream=Body())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            gen = transport_fn(client, "http://backend/v1/completions", {"stream": True}, "r", max_retries=1)
            if mode in ("other-error", "null", "list", "string-error"):
                with pytest.raises(httpx.HTTPStatusError):
                    await anext(gen)
            elif mode == "cancel":
                assert await anext(gen) == b'data: {"choices":[]}\n\n'
                await gen.aclose()
            else:
                chunks = [chunk async for chunk in gen]
                assert chunks == (
                    [json.dumps(error, separators=(",", ":")).encode()]
                    if mode == "recompute"
                    else [b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"]
                )
        assert len(calls) == 1
        if mode in ("cancel", "fragmented"):
            assert closed == [True]

    asyncio.run(run())


def test_layerwise_transport_setup_failure_releases_request(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from examples.disaggregated_prefill_v1 import load_balance_proxy_layerwise_server_example as layerwise

    state = Mock(
        next_req_id=AsyncMock(return_value="request-0"),
        select_decoder=Mock(return_value=0),
        calculate_decode_scores=Mock(return_value=10),
        decoders=[SimpleNamespace(client=None, url="http://decode")],
        cleanup_request_batch=AsyncMock(),
    )
    request = SimpleNamespace(
        json=AsyncMock(return_value={"prompt": "hello", "stream": True}),
        body=AsyncMock(return_value=b"hello"),
    )
    monkeypatch.setattr(layerwise, "proxy_state", state)
    monkeypatch.setattr(
        layerwise,
        "global_args",
        SimpleNamespace(host="localhost", port=8000, max_retries=1, retry_delay=0),
        raising=False,
    )
    monkeypatch.setattr(layerwise, "stream_service_response_with_retry", Mock(side_effect=RuntimeError("setup failed")))

    async def run():
        response = await layerwise._handle_completions("/completions", request)
        assert [chunk async for chunk in response.body_iterator] == []
        state.release_decoder.assert_called_once_with(0, 10)
        state.cleanup_request_batch.assert_awaited_once_with("request-0")

    asyncio.run(run())
