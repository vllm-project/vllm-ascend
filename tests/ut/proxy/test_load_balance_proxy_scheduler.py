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
