#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import socket
from types import SimpleNamespace

import pytest

from tests.ut.base import TestBase
from vllm_ascend.dp_rpc_port import (
    _dp_rpc_endpoint,
    check_dp_rpc_port_available,
    find_tcp_listener,
)


def _parallel_config(**overrides):
    config = SimpleNamespace(
        data_parallel_size=4,
        data_parallel_size_local=1,
        data_parallel_rank=0,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_rpc_port=12321,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestDpRpcPort(TestBase):
    def _free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def test_endpoint_skipped_without_data_parallel(self):
        self.assertIsNone(_dp_rpc_endpoint(_parallel_config(data_parallel_size=1)))

    def test_endpoint_skipped_when_all_engines_are_local(self):
        # The local IPC transport is used, so the TCP port is never bound.
        self.assertIsNone(_dp_rpc_endpoint(_parallel_config(data_parallel_size_local=4)))

    def test_endpoint_skipped_for_non_master_rank(self):
        # Only the rank 0 frontend binds the handshake address.
        self.assertIsNone(_dp_rpc_endpoint(_parallel_config(data_parallel_rank=1)))

    def test_endpoint_skipped_for_worker_node_of_a_replica(self):
        # Worker nodes of a multi node replica do not bind the address.
        self.assertIsNone(_dp_rpc_endpoint(_parallel_config(node_rank_within_dp=1)))

    def test_endpoint_returned_for_master_rank(self):
        self.assertEqual(
            _dp_rpc_endpoint(_parallel_config()),
            ("127.0.0.1", 12321),
        )

    def test_check_passes_when_port_is_free(self):
        port = self._free_port()
        check_dp_rpc_port_available(_parallel_config(data_parallel_rpc_port=port))

    def test_check_raises_when_port_is_taken(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            with self.assertRaises(RuntimeError) as ctx:
                check_dp_rpc_port_available(_parallel_config(data_parallel_rpc_port=port))
            self.assertIn(str(port), str(ctx.exception))
            self.assertIn("restart the container", str(ctx.exception))

    def test_check_ignores_a_listener_on_another_address(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            # The endpoint is not a local address, so this host never binds it.
            check_dp_rpc_port_available(
                _parallel_config(
                    data_parallel_master_ip="192.0.2.1",
                    data_parallel_rpc_port=port,
                )
            )

    def test_find_tcp_listener_reports_the_owner(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            holder = find_tcp_listener("127.0.0.1", port)
            self.assertIsNotNone(holder)
            assert holder is not None
            self.assertEqual(holder[0], os.getpid())

    def test_find_tcp_listener_returns_none_for_a_free_port(self):
        self.assertIsNone(find_tcp_listener("127.0.0.1", self._free_port()))


if __name__ == "__main__":
    pytest.main([__file__])
