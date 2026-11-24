# SPDX-License-Identifier: Apache-2.0
"""Local file:// image path validation for the EPD load-balance proxy.

CPU-only regression for get_hw_from_local(): the probe must follow the
directory the request names, never a same-named file elsewhere in the root
(basename-collision regression), and must reject anything that does not
resolve inside the configured --local-image-root.
"""

import math
import os
from pathlib import Path

import pytest
from PIL import Image

from examples.epd_disaggregated import epd_load_balance_proxy_layerwise_server_example as proxy

# The file:// URLs under test follow POSIX semantics (the proxy runs on Linux
# NPU servers), and the suite creates a symlink for the escape case.
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX file:// URL semantics and symlink creation required")


def _write_image(path: Path, size: tuple[int, int], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path, format=fmt)


@pytest.fixture()
def image_root(tmp_path: Path) -> Path:
    """Root with three images sharing the filename `same.png`, plus a JPEG
    whose name needs percent-encoding and an outside image that shares the
    name of an allowed one."""
    root = tmp_path / "images"
    _write_image(root / "same.png", (4, 4), "PNG")
    _write_image(root / "a" / "same.png", (64, 48), "PNG")
    _write_image(root / "b" / "same.png", (7, 11), "PNG")
    _write_image(root / "photo name.jpg", (5, 9), "JPEG")
    _write_image(tmp_path / "outside" / "same.png", (13, 17), "PNG")
    return root


def _url(root: Path, relative: str) -> str:
    return "file://" + str(root / relative)


def test_probe_follows_requested_directory_not_basename(image_root: Path):
    # Two sub-directories hold different images named `same.png`, colliding
    # with the root-level one; the probe must read the requested file.
    assert proxy.get_hw_from_local(_url(image_root, "a/same.png"), image_root=image_root) == (64, 48)
    assert proxy.get_hw_from_local(_url(image_root, "b/same.png"), image_root=image_root) == (7, 11)
    assert proxy.get_hw_from_local(_url(image_root, "same.png"), image_root=image_root) == (4, 4)


def test_probe_rejects_path_outside_root(image_root: Path, tmp_path: Path):
    # The outside file exists and shares its name with an allowed image: it
    # must be rejected, not silently probed through the same-named file.
    with pytest.raises(ValueError, match="inside"):
        proxy.get_hw_from_local(_url(tmp_path / "outside", "same.png"), image_root=image_root)


def test_probe_rejects_traversal(image_root: Path):
    with pytest.raises(ValueError, match="inside"):
        proxy.get_hw_from_local(_url(image_root, "../outside/same.png"), image_root=image_root)


def test_probe_rejects_symlink_escape(image_root: Path, tmp_path: Path):
    os.symlink(tmp_path / "outside" / "same.png", image_root / "escape.png")
    with pytest.raises(ValueError, match="inside"):
        proxy.get_hw_from_local(_url(image_root, "escape.png"), image_root=image_root)


def test_probe_rejects_non_file_and_remote_urls(image_root: Path):
    for url in ("https://example.com/a.png", "http://example.com/a.png", "file://otherhost/x.png"):
        with pytest.raises(ValueError):
            proxy.get_hw_from_local(url, image_root=image_root)


def test_probe_accepts_localhost_authority(image_root: Path):
    url = "file://localhost" + str(image_root / "same.png")
    assert proxy.get_hw_from_local(url, image_root=image_root) == (4, 4)


def test_probe_decodes_percent_encoded_names(image_root: Path):
    assert proxy.get_hw_from_local(_url(image_root, "photo%20name.jpg"), image_root=image_root) == (5, 9)


def test_probe_rejects_missing_file(image_root: Path):
    with pytest.raises(ValueError, match="existing file"):
        proxy.get_hw_from_local(_url(image_root, "missing.png"), image_root=image_root)


def test_calculate_messages_size_uses_validated_root(image_root: Path):
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _url(image_root, "a/same.png")}},
                    {"type": "text", "text": "hi"},
                ],
            }
        ]
    }
    stats = proxy.calculate_messages_size(request, b"body", image_root=image_root)
    assert stats["mul_token"] == math.ceil(48 / 32) * math.ceil(64 / 32)
    assert stats["text_char_count"] == 2
