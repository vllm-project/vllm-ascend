import base64
from pathlib import Path

import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion

IMAGE_EDIT_URI = "/v1/images/edits"
IMAGE_EDIT_MODEL = "qwen_image_edit_2511"
PROMPT = "Convert this image to watercolor style."
ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@pytest.fixture(scope="module")
def sample_images(tmp_path_factory):
    project_images = _project_test_images()
    if project_images:
        return project_images

    # Keep a tiny fallback image so the multipart tests remain runnable in checkouts without data/images assets.
    image_path = tmp_path_factory.mktemp("image_edit") / "sample.png"
    image_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
    return [image_path]


def _project_test_images():
    try:
        from tests.e2e.weekly.single_node.engine_func_test_robot.utility import image_helper
    except ImportError:
        return []

    try:
        return [Path(path) for path in image_helper.get_all_test_images()]
    except FileNotFoundError:
        return []


def _project_image_urls(count=1):
    try:
        from tests.e2e.weekly.single_node.engine_func_test_robot.utility import image_helper
    except ImportError:
        return []

    try:
        return image_helper.get_random_test_image_urls(count)
    except FileNotFoundError:
        return []


def _pick_images(images, count):
    return (images * ((count // len(images)) + 1))[:count]


def _image_files(paths):
    opened_files = []
    files = []
    for path in paths:
        file_obj = Path(path).open("rb")
        opened_files.append(file_obj)
        files.append(("image[]", (Path(path).name, file_obj, "image/png")))
    return files, opened_files


def _send_image_edit_request(api_client, data, image_paths=None):
    files = None
    opened_files = []
    if image_paths:
        files, opened_files = _image_files(image_paths)
    try:
        return api_client.post(IMAGE_EDIT_URI, data=data, files=files)
    finally:
        for file_obj in opened_files:
            file_obj.close()


def _base_request(**overrides):
    data = {
        "model": IMAGE_EDIT_MODEL,
        "prompt": PROMPT,
        "size": "1024x1024",
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"size": "512x512"},
        {"output_format": "png"},
        {"output_format": "jpeg", "output_compression": 80},
        {"seed": 12345},
        {"negative_prompt": "blur"},
        {"response_format": "b64_json"},
    ],
    ids=["basic", "size", "format", "compression", "seed", "negative_prompt", "response_format"],
)
def test_image_edit_accepts_representative_single_image_options(api_client, sample_images, overrides):
    # Representative success cases for the main optional fields; full value matrices are intentionally avoided.
    response = _send_image_edit_request(api_client, _base_request(**overrides), _pick_images(sample_images, 1))
    assertion.assert_image_edit_response_fields(response)


@pytest.mark.parametrize("image_count", [2, 5], ids=["dual_images", "multi_images"])
def test_image_edit_accepts_multiple_images(api_client, sample_images, image_count):
    # Prefer project image fixtures; reuse them only when the fixture corpus has fewer images than this case needs.
    response = _send_image_edit_request(api_client, _base_request(), _pick_images(sample_images, image_count))
    assertion.assert_image_edit_response_fields(response)


def test_image_edit_accepts_image_url_input(api_client):
    # URL mode is separate from multipart image upload; prefer project URL fixtures when available.
    image_urls = _project_image_urls(1) or ["https://httpbin.org/image/png"]
    data = _base_request(**{"url[]": image_urls[0]})
    response = _send_image_edit_request(api_client, data)
    assertion.assert_image_edit_response_fields(response)


@pytest.mark.parametrize(
    "data",
    [
        {"prompt": PROMPT},
        {"model": IMAGE_EDIT_MODEL},
        {"model": "", "prompt": PROMPT},
        {"model": IMAGE_EDIT_MODEL, "prompt": ""},
    ],
    ids=["missing_model", "missing_prompt", "empty_model", "empty_prompt"],
)
def test_image_edit_rejects_missing_or_empty_required_fields(api_client, sample_images, data):
    # Required-field validation is covered by one missing and one empty value for each required field.
    response = _send_image_edit_request(api_client, data, _pick_images(sample_images, 1))
    assertion.assert_validation_error_response(response)


@pytest.mark.parametrize(
    "overrides",
    [
        {"size": "invalid"},
        {"size": "0x0"},
        {"size": "-1x1024"},
        {"output_compression": -1, "output_format": "jpeg"},
        {"output_compression": 101, "output_format": "jpeg"},
        {"seed": "not-an-int"},
        {"response_format": "invalid"},
        {"user": {"name": "invalid"}},
    ],
    ids=[
        "bad_size_format",
        "zero_size",
        "negative_size",
        "compression_low",
        "compression_high",
        "bad_seed",
        "bad_response_format",
        "bad_user",
    ],
)
def test_image_edit_rejects_invalid_option_values(api_client, sample_images, overrides):
    # Invalid options are grouped by validation class instead of testing every equivalent literal.
    response = _send_image_edit_request(api_client, _base_request(**overrides), _pick_images(sample_images, 1))
    assertion.assert_validation_error_response(response)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not_a_valid_url",
        "ftp://example.com/image.png",
        "https://httpbin.org/status/404",
    ],
    ids=["empty", "malformed", "unsupported_protocol", "not_found"],
)
def test_image_edit_rejects_invalid_urls(api_client, url):
    # URL validation covers empty, malformed, unsupported protocol, and remote fetch failure.
    response = _send_image_edit_request(api_client, _base_request(**{"url[]": url}))
    assertion.assert_validation_error_response(response)


def test_image_edit_rejects_mixed_file_and_url_inputs(api_client, sample_images):
    # The API accepts either multipart images or URL input; mixing both should be rejected.
    image_urls = _project_image_urls(1) or ["https://httpbin.org/image/png"]
    response = _send_image_edit_request(
        api_client,
        _base_request(**{"url[]": image_urls[0]}),
        _pick_images(sample_images, 1),
    )
    assertion.assert_validation_error_response(response)


def test_image_edit_rejects_missing_image_input(api_client):
    # A syntactically valid request still needs at least one image source.
    response = _send_image_edit_request(api_client, _base_request())
    assertion.assert_validation_error_response(response)
