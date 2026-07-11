"""Helpers for image edit and image generation API tests."""

import base64
import random
import struct
from pathlib import Path

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
IMAGE_URLS_FILE = IMAGES_DIR / "image_urls.txt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def get_all_test_images() -> list:
    """Return all local image fixtures, or an empty list when no fixture directory exists."""
    if not IMAGES_DIR.exists():
        return []
    return [
        str(path) for path in sorted(IMAGES_DIR.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_test_image_urls() -> list:
    """Return image URLs listed in data/images/image_urls.txt."""
    if not IMAGE_URLS_FILE.exists():
        return []
    return [
        line.strip()
        for line in IMAGE_URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _repeat_to_count(items, count):
    if not items:
        return []
    if len(items) >= count:
        return random.sample(items, count)
    return (items * ((count // len(items)) + 1))[:count]


def get_random_test_image_urls(count: int = 1) -> list:
    """Return up to count image URLs, reusing fixtures when fewer URLs are available."""
    return _repeat_to_count(get_test_image_urls(), count)


def send_image_edit_request(api_client, data: dict, image_files: list = None):
    """Send a multipart image edit request and close opened files."""
    files = []
    opened_files = []
    try:
        for image_path in image_files or []:
            path = Path(image_path)
            file_obj = path.open("rb")
            opened_files.append(file_obj)
            mime_type = MIME_TYPES.get(path.suffix.lower(), "image/png")
            files.append(("image[]", (path.name, file_obj, mime_type)))
        return api_client.post("/v1/images/edits", data=data, files=files or None)
    finally:
        for file_obj in opened_files:
            file_obj.close()


def send_image_generation_request(api_client, data: dict):
    """Send a JSON image generation request."""
    return api_client.post("/v1/images/generations", json=data)


def decode_b64_image(b64_json: str) -> bytes:
    """Decode a base64 image string, accepting optional data URI prefixes."""
    if "," in b64_json and ";base64," in b64_json:
        b64_json = b64_json.split(",", 1)[1]
    return base64.b64decode(b64_json)


def get_image_dimensions(image_data: bytes) -> tuple:
    """Return image dimensions for PNG, JPEG, or WebP bytes."""
    if image_data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">I", image_data[16:20])[0], struct.unpack(">I", image_data[20:24])[0]

    if image_data[:2] == b"\xff\xd8":
        pos = 2
        while pos < len(image_data):
            if image_data[pos] != 0xFF:
                break
            marker = image_data[pos + 1]
            if marker in (0xC0, 0xC2):
                height = struct.unpack(">H", image_data[pos + 5 : pos + 7])[0]
                width = struct.unpack(">H", image_data[pos + 7 : pos + 9])[0]
                return width, height
            if marker in (0xD8, 0xD9):
                pos += 2
            elif marker == 0xFF:
                pos += 1
            else:
                pos += 2 + struct.unpack(">H", image_data[pos + 2 : pos + 4])[0]

    if image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
        chunk_type = image_data[12:16]
        if chunk_type == b"VP8 ":
            return (
                struct.unpack("<H", image_data[26:28])[0] & 0x3FFF,
                struct.unpack("<H", image_data[28:30])[0] & 0x3FFF,
            )
        if chunk_type == b"VP8L":
            bits = struct.unpack("<I", image_data[21:25])[0]
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk_type == b"VP8X":
            width = struct.unpack("<I", b"\x00" + image_data[24:27])[0] + 1
            height = struct.unpack("<I", b"\x00" + image_data[27:30])[0] + 1
            return width, height

    return None, None


def parse_size_string(size: str) -> tuple:
    """Parse a WxH size string."""
    if not size:
        return None, None
    parts = size.lower().split("x")
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def assert_image_generation_response_fields(response, msg=""):
    """Verify image generation response fields and return the JSON body."""
    assertion.assert_status_code_200(response, msg)
    response_json = response.json()
    assertion.check.is_true("data" in response_json, f"{msg}Response should contain data")
    assertion.check.is_true(response_json.get("data"), f"{msg}data should contain at least one result")

    output_format = response_json.get("output_format")
    if output_format is not None:
        assertion.check.is_in(output_format.lower(), ["png", "jpeg", "webp"], f"{msg}Invalid output_format")

    if "size" in response_json:
        width, height = parse_size_string(response_json["size"])
        assertion.check.is_true(width is not None and height is not None, f"{msg}Invalid size format")

    for idx, item in enumerate(response_json["data"]):
        has_b64 = bool(item.get("b64_json"))
        has_url = bool(item.get("url"))
        assertion.check.is_true(has_b64 or has_url, f"{msg}data[{idx}] should contain b64_json or url")
        if has_b64:
            width, height = get_image_dimensions(decode_b64_image(item["b64_json"]))
            assertion.check.is_true(width and height, f"{msg}data[{idx}] image dimensions are invalid")

    return response_json
