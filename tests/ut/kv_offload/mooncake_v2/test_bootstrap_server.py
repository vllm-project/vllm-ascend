# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi.testclient import TestClient

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.bootstrap_server import AscendMooncakeBootstrapServer


@pytest.fixture
def client():
    server = AscendMooncakeBootstrapServer("127.0.0.1", 0)
    with TestClient(server.app) as client:
        yield client
    server.shutdown()


def payload(**changes):
    return dict(engine_id="p", dp_rank=0, tp_rank=0, pp_rank=0, addr="tcp://p:1234", **changes)


def test_legacy_registration_defaults_and_proxy_envelope(client):
    assert client.post("/register", json=payload()).status_code == 200
    entry = client.get("/query").json()["0"]
    assert entry["engine_id"] == "p"
    assert (entry["pcp_size"], entry["dcp_size"]) == (1, 1)
    assert entry["worker_addr"]["0"]["0"]["0"] == "tcp://p:1234"


def test_context_parallel_workers_do_not_overwrite_each_other(client):
    for pcp in range(2):
        for tp in range(2):
            for pp in range(2):
                data = payload(pcp_size=2, dcp_size=4, pcp_rank=pcp, tp_size=2, pp_size=2)
                data.update(tp_rank=tp, pp_rank=pp, addr=f"tcp://p:{1234 + pcp * 4 + tp * 2 + pp}")
                assert client.post("/register", json=data).status_code == 200
                assert client.post("/register", json=data).status_code == 200
    entry = client.get("/query").json()["0"]
    for pcp in range(2):
        for tp in range(2):
            for pp in range(2):
                endpoint = entry["worker_addr"][str(pcp)][str(tp)][str(pp)]
                assert endpoint == f"tcp://p:{1234 + pcp * 4 + tp * 2 + pp}"


@pytest.mark.parametrize(
    "change", [dict(engine_id="other"), dict(addr="tcp://other:1234"), dict(pcp_size=2), dict(dcp_size=2)]
)
def test_conflict_does_not_modify_registry(client, change):
    original = payload()
    assert client.post("/register", json=original).status_code == 200
    before = client.get("/query").json()
    assert client.post("/register", json=original | change).status_code == 400
    assert client.get("/query").json() == before


@pytest.mark.parametrize(
    "change",
    [dict(pcp_rank=1), dict(pcp_size=0), dict(dcp_size=0), dict(tp_rank=-1), dict(dp_rank=-1)],
)
def test_invalid_ranks_are_rejected(client, change):
    assert client.post("/register", json=payload() | change).status_code == 422
    assert client.get("/query").json() == {}


def test_independent_dp(client):
    data = payload(dcp_size=2)
    assert client.post("/register", json=data).status_code == 200
    assert client.post("/register", json=payload() | {"dp_rank": 1, "engine_id": "p1"}).status_code == 200
    assert set(client.get("/query").json()) == {"0", "1"}
