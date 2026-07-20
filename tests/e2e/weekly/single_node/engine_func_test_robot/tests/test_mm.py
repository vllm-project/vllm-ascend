import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, mm_helper


def _skip_if_capacity_too_small(request, image_count=0, video_count=0, audio_count=0):
    if request.config.getoption("--imageNum") < image_count:
        pytest.skip("model does not support the requested number of images")
    if request.config.getoption("--videoNum") < video_count:
        pytest.skip("model does not support the requested number of videos")
    if request.config.getoption("--audioNum") < audio_count:
        pytest.skip("model does not support the requested number of audios")


def _media_sources(kind, count):
    if kind == "images":
        return mm_helper.get_random_images(count)
    if kind == "videos":
        return mm_helper.get_random_videos(count)
    return mm_helper.get_random_audios(count)


def _send_multimodal_request(api_client, messages, stream=False):
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 128,
        "stream": stream,
    }
    return api_client.post("/v1/chat/completions", json=request_body)


def _assert_multimodal_success(response, stream=False):
    assertion.assert_chat_completion_success(response, stream=stream)


def test_mm_accepts_single_image(api_client, request):
    # A single image covers the base64 image request path.
    _skip_if_capacity_too_small(request, image_count=1)
    images = _media_sources("images", 1)
    if not images:
        pytest.skip("no base64 image fixtures available")

    messages = [mm_helper.build_multimodal_message("Describe the image.", images=images)]
    response = _send_multimodal_request(api_client, messages)
    _assert_multimodal_success(response)


def test_mm_accepts_single_video(api_client, request):
    # A single video covers the base64 video request path.
    _skip_if_capacity_too_small(request, video_count=1)
    videos = _media_sources("videos", 1)
    if not videos:
        pytest.skip("no base64 video fixtures available")

    messages = [mm_helper.build_multimodal_message("Describe the video.", videos=videos)]
    response = _send_multimodal_request(api_client, messages)
    _assert_multimodal_success(response)


def test_mm_accepts_single_audio(api_client, request):
    # Audio-only coverage verifies the audio content shape and model option gate.
    _skip_if_capacity_too_small(request, audio_count=1)
    audios = _media_sources("audios", 1)
    if not audios:
        pytest.skip("no base64 audio fixtures available")

    messages = [mm_helper.build_multimodal_message("Describe the audio.", audios=audios)]
    response = _send_multimodal_request(api_client, messages)
    _assert_multimodal_success(response)


def test_mm_accepts_image_video_audio_combination(api_client, request):
    # The mixed-media path verifies all supported base64 media types in one request.
    _skip_if_capacity_too_small(request, image_count=1, video_count=1, audio_count=1)
    images = _media_sources("images", 1)
    videos = _media_sources("videos", 1)
    audios = _media_sources("audios", 1)
    if not images or not videos or not audios:
        pytest.skip("no complete base64 multimodal fixture set available")

    messages = [
        mm_helper.build_multimodal_message(
            "Describe the image, video, and audio.",
            images=images,
            videos=videos,
            audios=audios,
        )
    ]
    response = _send_multimodal_request(api_client, messages)
    _assert_multimodal_success(response)


def test_mm_accepts_streaming_image_request(api_client, request):
    # One streaming case is enough to cover SSE response handling for multimodal requests.
    _skip_if_capacity_too_small(request, image_count=1)
    images = mm_helper.get_random_images(1)
    if not images:
        pytest.skip("no base64 image fixtures available")

    messages = [mm_helper.build_multimodal_message("Describe the image.", images=images)]
    response = _send_multimodal_request(api_client, messages, stream=True)
    _assert_multimodal_success(response, stream=True)


def test_mm_accepts_multi_turn_image_request(api_client, request):
    # Multi-turn coverage verifies that multimodal content can coexist with prior assistant messages.
    _skip_if_capacity_too_small(request, image_count=1)
    images = mm_helper.get_random_images(1)
    if not images:
        pytest.skip("no base64 image fixtures available")

    messages = [
        mm_helper.build_multimodal_message("Describe the image.", images=images),
        {"role": "assistant", "content": "The image contains visible objects."},
        {"role": "user", "content": "Summarize it in one sentence."},
    ]
    response = _send_multimodal_request(api_client, messages)
    _assert_multimodal_success(response)


@pytest.mark.parametrize(
    "content",
    [
        {"type": "image_url", "image_url": {"url": "not_a_valid_url"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,invalid-base64"}},
        {"type": "video_url", "video_url": {"url": "not_a_valid_url"}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,invalid-base64"}},
        {"type": "audio_url", "audio_url": {"url": "not_a_valid_url"}},
        {"type": "audio_url", "audio_url": {"url": "data:audio/mpeg;base64,invalid-base64"}},
        {"type": "image_url", "image_url": {"url": ""}},
    ],
    ids=[
        "bad_image_url",
        "bad_image_base64",
        "bad_video_url",
        "bad_video_base64",
        "bad_audio_url",
        "bad_audio_base64",
        "empty_image_url",
    ],
)
def test_mm_rejects_invalid_media_content(api_client, content):
    # Invalid media cases are collapsed to one representative URL/base64 failure per media type.
    messages = [{"role": "user", "content": [{"type": "text", "text": "Describe this."}, content]}]
    response = _send_multimodal_request(api_client, messages)
    assertion.assert_validation_error_response(response)
