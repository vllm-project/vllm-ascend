# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from examples.rl import weight_transfer_http_utils as http_utils


_BASE_URL = "http://localhost:8000"


def test_pause_generation_posts_pause_endpoint():
    response = MagicMock()
    with patch.object(http_utils.requests, "post", return_value=response) as post:
        http_utils.pause_generation(_BASE_URL)

    post.assert_called_once_with(f"{_BASE_URL}/pause", timeout=60)
    response.raise_for_status.assert_called_once_with()


def test_resume_generation_posts_resume_endpoint():
    response = MagicMock()
    with patch.object(http_utils.requests, "post", return_value=response) as post:
        http_utils.resume_generation(_BASE_URL)

    post.assert_called_once_with(f"{_BASE_URL}/resume", timeout=60)
    response.raise_for_status.assert_called_once_with()


def test_init_weight_transfer_engine_posts_init_info():
    response = MagicMock()
    init_info = {"master_address": "127.0.0.1", "master_port": 12345}
    with patch.object(http_utils.requests, "post", return_value=response) as post:
        http_utils.init_weight_transfer_engine(_BASE_URL, init_info)

    post.assert_called_once_with(
        f"{_BASE_URL}/init_weight_transfer_engine",
        json={"init_info": init_info},
        timeout=60,
    )
    response.raise_for_status.assert_called_once_with()


def test_start_weight_update_posts_checkpoint_format():
    response = MagicMock()
    with patch.object(http_utils.requests, "post", return_value=response) as post:
        http_utils.start_weight_update(_BASE_URL, is_checkpoint_format=False)

    post.assert_called_once_with(
        f"{_BASE_URL}/start_weight_update",
        json={"is_checkpoint_format": False},
        timeout=60,
    )
    response.raise_for_status.assert_called_once_with()


def test_update_weights_posts_update_info_with_long_timeout():
    response = MagicMock()
    update_info = {"names": ["model.weight"], "dtype_names": ["float32"], "shapes": [[1]], "packed": False}
    with patch.object(http_utils.requests, "post", return_value=response) as post:
        http_utils.update_weights(_BASE_URL, update_info)

    post.assert_called_once_with(
        f"{_BASE_URL}/update_weights",
        json={"update_info": update_info},
        timeout=300,
    )
    response.raise_for_status.assert_called_once_with()


def test_finish_weight_update_posts_finish_endpoint():
    response = MagicMock()
    with patch.object(http_utils.requests, "post", return_value=response) as post:
        http_utils.finish_weight_update(_BASE_URL)

    post.assert_called_once_with(f"{_BASE_URL}/finish_weight_update", timeout=60)
    response.raise_for_status.assert_called_once_with()


def test_get_world_size_returns_response_value():
    response = MagicMock()
    response.json.return_value = {"world_size": 4}
    with patch.object(http_utils.requests, "get", return_value=response) as get:
        world_size = http_utils.get_world_size(_BASE_URL)

    assert world_size == 4
    get.assert_called_once_with(f"{_BASE_URL}/get_world_size", timeout=10)
    response.raise_for_status.assert_called_once_with()
