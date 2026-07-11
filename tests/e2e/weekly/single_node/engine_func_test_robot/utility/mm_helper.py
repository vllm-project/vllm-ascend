"""Helpers for multimodal chat completion tests."""

import base64
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}
AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}


def get_all_files(directory: Path, extensions: set) -> list:
    """Return local fixture files with matching extensions."""
    if not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != ".gitkeep" and path.suffix.lower() in extensions
    ]


def _repeat_to_count(items, count):
    if not items:
        return []
    if len(items) >= count:
        return random.sample(items, count)
    return (items * ((count // len(items)) + 1))[:count]


def get_all_images() -> list:
    return get_all_files(IMAGES_DIR, IMAGE_EXTENSIONS)


def get_all_videos() -> list:
    return get_all_files(VIDEOS_DIR, VIDEO_EXTENSIONS)


def get_all_audios() -> list:
    return get_all_files(AUDIO_DIR, AUDIO_EXTENSIONS)


def get_random_images(count: int = 1) -> list:
    return _repeat_to_count(get_all_images(), count)


def get_random_videos(count: int = 1) -> list:
    return _repeat_to_count(get_all_videos(), count)


def get_random_audios(count: int = 1) -> list:
    return _repeat_to_count(get_all_audios(), count)


def get_urls_from_file(file_path: Path) -> list:
    """Return non-empty, non-comment URLs from a fixture URL file."""
    if not file_path.exists():
        return []
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def get_image_urls() -> list:
    return get_urls_from_file(IMAGES_DIR / "image_urls.txt")


def get_video_urls() -> list:
    return get_urls_from_file(VIDEOS_DIR / "video_urls.txt")


def get_audio_urls() -> list:
    return get_urls_from_file(AUDIO_DIR / "audio_urls.txt")


def get_random_image_urls(count: int = 1) -> list:
    return _repeat_to_count(get_image_urls(), count)


def get_random_video_urls(count: int = 1) -> list:
    return _repeat_to_count(get_video_urls(), count)


def get_random_audio_urls(count: int = 1) -> list:
    return _repeat_to_count(get_audio_urls(), count)


def _data_url(path: Path, mime_types: dict, default_mime_type: str) -> str:
    mime_type = mime_types.get(path.suffix.lower(), default_mime_type)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_source(source, source_type: str, mime_types: dict, default_mime_type: str) -> str:
    if source_type == "url":
        return str(source)
    if isinstance(source, Path):
        return _data_url(source, mime_types, default_mime_type)
    if isinstance(source, str) and source.startswith("data:"):
        return source
    if isinstance(source, str) and ";base64," in source:
        return f"data:{source}"
    return f"data:{default_mime_type};base64,{source}"


def build_image_content(image_source, source_type: str = "url") -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": _normalize_source(image_source, source_type, IMAGE_MIME_TYPES, "image/jpeg"),
        },
    }


def build_video_content(video_source, source_type: str = "url") -> dict:
    return {
        "type": "video_url",
        "video_url": {
            "url": _normalize_source(video_source, source_type, VIDEO_MIME_TYPES, "video/mp4"),
        },
    }


def build_audio_content(audio_source, source_type: str = "url") -> dict:
    return {
        "type": "audio_url",
        "audio_url": {
            "url": _normalize_source(audio_source, source_type, AUDIO_MIME_TYPES, "audio/mpeg"),
        },
    }


def build_multimodal_message(
    prompt: str,
    images: list = None,
    videos: list = None,
    audios: list = None,
    source_type: str = "url",
) -> dict:
    """Build one user message containing text plus optional image/video/audio content."""
    content = [{"type": "text", "text": prompt}]
    content.extend(build_image_content(image, source_type) for image in images or [])
    content.extend(build_video_content(video, source_type) for video in videos or [])
    content.extend(build_audio_content(audio, source_type) for audio in audios or [])
    return {"role": "user", "content": content}
