import unittest
from unittest.mock import MagicMock, patch

import zmq

from vllm_ascend.distributed.utils import (DONE_RECVING_MSG, DONE_SENDING_MSG,
                                           GET_META_MSG, ensure_zmq_recv,
                                           ensure_zmq_send, get_network_utils,
                                           string_to_int64_hash)


class TestDistributedUtils(unittest.TestCase):

    def test_constants(self):
        """Test that the ZMQ constants are defined correctly."""
        self.assertEqual(GET_META_MSG, b"get_meta_msg")
        self.assertEqual(DONE_RECVING_MSG, b"done_recving_msg")
        self.assertEqual(DONE_SENDING_MSG, b"done_sending_msg")

    def test_ensure_zmq_send_success(self):
        """Test successful send operation."""
        mock_socket = MagicMock(spec=zmq.Socket)
        test_data = b"test_data"

        ensure_zmq_send(mock_socket, test_data)

        mock_socket.send.assert_called_once_with(test_data)

    def test_ensure_zmq_send_retry_success(self):
        """Test send operation succeeds after retries."""
        mock_socket = MagicMock(spec=zmq.Socket)
        test_data = b"test_data"

        # Fail once, then succeed
        mock_socket.send.side_effect = [zmq.ZMQError("Test error"), None]

        ensure_zmq_send(mock_socket, test_data)

        self.assertEqual(mock_socket.send.call_count, 2)

    def test_ensure_zmq_send_max_retries(self):
        """Test send operation fails after max retries."""
        mock_socket = MagicMock(spec=zmq.Socket)
        test_data = b"test_data"

        # Always fail
        mock_socket.send.side_effect = zmq.ZMQError("Test error")

        with self.assertRaises(RuntimeError) as context:
            ensure_zmq_send(mock_socket, test_data, max_retries=2)

        self.assertIn("Failed to send data after 2 retries",
                      str(context.exception))
        self.assertEqual(mock_socket.send.call_count, 2)

    def test_ensure_zmq_recv_success(self):
        """Test successful receive operation."""
        mock_socket = MagicMock(spec=zmq.Socket)
        mock_poller = MagicMock(spec=zmq.Poller)
        test_data = b"test_data"

        # Mock successful poll and receive
        mock_poller.poll.return_value = [(mock_socket, zmq.POLLIN)]
        mock_socket.recv.return_value = test_data

        result = ensure_zmq_recv(mock_socket, mock_poller, timeout=1.0)

        self.assertEqual(result, test_data)
        mock_poller.poll.assert_called_once()
        mock_socket.recv.assert_called_once()

    def test_ensure_zmq_recv_timeout(self):
        """Test receive operation fails on timeout."""
        mock_socket = MagicMock(spec=zmq.Socket)
        mock_poller = MagicMock(spec=zmq.Poller)

        # Mock timeout (empty dict from poll)
        mock_poller.poll.return_value = []

        with self.assertRaises(RuntimeError) as context:
            ensure_zmq_recv(mock_socket,
                            mock_poller,
                            timeout=0.1,
                            max_retries=2)

        self.assertIn("Failed to receive data after 2 retries",
                      str(context.exception))
        self.assertEqual(mock_poller.poll.call_count, 2)

    def test_ensure_zmq_recv_retry_success(self):
        """Test receive operation succeeds after retries."""
        mock_socket = MagicMock(spec=zmq.Socket)
        mock_poller = MagicMock(spec=zmq.Poller)
        test_data = b"test_data"

        # Fail once (timeout), then succeed
        mock_poller.poll.side_effect = [[], [(mock_socket, zmq.POLLIN)]]
        mock_socket.recv.return_value = test_data

        result = ensure_zmq_recv(mock_socket, mock_poller, timeout=0.1)

        self.assertEqual(result, test_data)
        self.assertEqual(mock_poller.poll.call_count, 2)
        mock_socket.recv.assert_called_once()

    @patch('vllm_ascend.distributed.utils.vllm_version_is')
    def test_get_network_utils_v0_11_0(self, mock_version_is):
        """Test get_network_utils returns correct functions for v0.11.0."""
        mock_version_is.return_value = True

        with patch('vllm_ascend.distributed.utils.vllm.utils') as mock_utils:
            mock_utils.get_ip = MagicMock()
            mock_utils.make_zmq_path = MagicMock()
            mock_utils.make_zmq_socket = MagicMock()

            get_ip, make_zmq_path, make_zmq_socket = get_network_utils()

            # Verify the imports from vllm.utils were attempted
            mock_version_is.assert_called_once_with("0.11.0")

    @patch('vllm_ascend.distributed.utils.vllm_version_is')
    def test_get_network_utils_newer_version(self, mock_version_is):
        """Test get_network_utils returns correct functions for newer versions."""
        mock_version_is.return_value = False

        with patch('vllm_ascend.distributed.utils.vllm.utils.network_utils'
                   ) as mock_network_utils:
            mock_network_utils.get_ip = MagicMock()
            mock_network_utils.make_zmq_path = MagicMock()
            mock_network_utils.make_zmq_socket = MagicMock()

            get_ip, make_zmq_path, make_zmq_socket = get_network_utils()

            # Verify the imports from vllm.utils.network_utils were attempted
            mock_version_is.assert_called_once_with("0.11.0")

    def test_string_to_int64_hash_consistency(self):
        """Test that string_to_int64_hash produces consistent results."""
        test_string = "test_string"
        hash1 = string_to_int64_hash(test_string)
        hash2 = string_to_int64_hash(test_string)

        # Same input should produce same hash
        self.assertEqual(hash1, hash2)

    def test_string_to_int64_hash_different_strings(self):
        """Test that different strings produce different hashes."""
        hash1 = string_to_int64_hash("string1")
        hash2 = string_to_int64_hash("string2")

        # Different inputs should produce different hashes
        self.assertNotEqual(hash1, hash2)

    def test_string_to_int64_hash_return_type(self):
        """Test that string_to_int64_hash returns an integer."""
        result = string_to_int64_hash("test")

        self.assertIsInstance(result, int)
        # Verify it fits in uint64
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 2**64 - 1)


if __name__ == '__main__':
    unittest.main()
