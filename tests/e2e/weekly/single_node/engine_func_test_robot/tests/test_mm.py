import base64
import io
import struct
import wave
import zlib

import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion

AUDIO_SAMPLE_RATE = 16_000
AUDIO_DURATION_SECONDS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
IMAGE_SIZE = 32
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CAPACITY_OPTIONS = {
    "image": "--imageNum",
    "video": "--videoNum",
    "audio": "--audioNum",
}


def _png_chunk(chunk_type, data):
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _png_bytes(color):
    """Build a small RGB PNG with standard-library primitives."""
    header = struct.pack(">IIBBBBB", IMAGE_SIZE, IMAGE_SIZE, 8, 2, 0, 0, 0)
    scanline = b"\x00" + bytes(color) * IMAGE_SIZE
    pixels = zlib.compress(scanline * IMAGE_SIZE)
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


def _base64(data):
    return base64.b64encode(data).decode("ascii")


def _image_data_url():
    return f"data:image/png;base64,{_base64(_png_bytes((220, 40, 40)))}"


def _video_data_url():
    # vLLM accepts comma-separated image frames as an in-memory video payload.
    frames = [_png_bytes((220, 40, 40)), _png_bytes((40, 80, 220))]
    return f"data:video/png;base64,{','.join(_base64(frame) for frame in frames)}"


def _audio_data_url():
    audio_buffer = io.BytesIO()
    frame_count = AUDIO_SAMPLE_RATE * AUDIO_DURATION_SECONDS
    with wave.open(audio_buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(AUDIO_SAMPLE_WIDTH_BYTES)
        audio_file.setframerate(AUDIO_SAMPLE_RATE)
        audio_file.writeframes(b"\x00\x00" * frame_count)
    return f"data:audio/wav;base64,{_base64(audio_buffer.getvalue())}"


@pytest.fixture(scope="module")
def generated_media():
    """Generate deterministic media payloads without files or network access."""
    return {
        "image": _image_data_url(),
        "video": _video_data_url(),
        "audio": _audio_data_url(),
    }


def _build_multimodal_message(prompt, **media):
    content = [{"type": "text", "text": prompt}]
    for media_type, data_url in media.items():
        content_type = f"{media_type}_url"
        content.append({"type": content_type, content_type: {"url": data_url}})
    return {"role": "user", "content": content}


def _skip_unsupported_media(request, *media_types):
    for media_type in media_types:
        if request.config.getoption(CAPACITY_OPTIONS[media_type]) < 1:
            pytest.skip(f"model does not support {media_type} input")


def _send_multimodal_request(api_client, messages, stream=False):
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 128,
        "stream": stream,
    }
    return api_client.post("/v1/chat/completions", json=request_body)


@pytest.mark.parametrize(
    "media_type",
    ["image", "video", "audio"],
    ids=["image", "video", "audio"],
)
def test_mm_accepts_single_media(api_client, request, generated_media, media_type):
    # Each media type covers one generated, in-memory Base64 request path.
    _skip_unsupported_media(request, media_type)
    messages = [
        _build_multimodal_message(
            f"Describe the {media_type}.",
            **{media_type: generated_media[media_type]},
        )
    ]
    response = _send_multimodal_request(api_client, messages)
    assertion.assert_chat_completion_success(response)


def test_mm_accepts_image_video_audio_combination(api_client, request, generated_media):
    # Keep one mixed request for multimodal integration coverage.
    _skip_unsupported_media(request, "image", "video", "audio")
    messages = [
        _build_multimodal_message(
            "Describe the image, video, and audio.",
            image=generated_media["image"],
            video=generated_media["video"],
            audio=generated_media["audio"],
        )
    ]
    response = _send_multimodal_request(api_client, messages)
    assertion.assert_chat_completion_success(response)


def test_mm_accepts_streaming_image_request(api_client, request, generated_media):
    # One streaming case covers SSE handling for multimodal requests.
    _skip_unsupported_media(request, "image")
    messages = [_build_multimodal_message("Describe the image.", image=generated_media["image"])]
    response = _send_multimodal_request(api_client, messages, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


def test_mm_accepts_multi_turn_image_request(api_client, request, generated_media):
    # Verify that generated media can coexist with conversation history.
    _skip_unsupported_media(request, "image")
    messages = [
        _build_multimodal_message("Describe the image.", image=generated_media["image"]),
        {"role": "assistant", "content": "The image contains a solid color."},
        {"role": "user", "content": "Summarize it in one sentence."},
    ]
    response = _send_multimodal_request(api_client, messages)
    assertion.assert_chat_completion_success(response)


@pytest.mark.parametrize(
    ("media_type", "media_url"),
    [
        ("image", "not_a_valid_url"),
        ("image", "data:image/png;base64,invalid-base64"),
        ("video", "not_a_valid_url"),
        ("video", "data:video/png;base64,invalid-base64"),
        ("audio", "not_a_valid_url"),
        ("audio", "data:audio/wav;base64,invalid-base64"),
    ],
    ids=[
        "bad_image_url",
        "bad_image_base64",
        "bad_video_url",
        "bad_video_base64",
        "bad_audio_url",
        "bad_audio_base64",
    ],
)
def test_mm_rejects_invalid_media_content(api_client, media_type, media_url):
    # URL syntax and Base64 decoding are separate validation paths.
    messages = [_build_multimodal_message("Describe this.", **{media_type: media_url})]
    response = _send_multimodal_request(api_client, messages)
    assertion.assert_validation_error_response(response)
