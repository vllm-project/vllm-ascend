"""多模态对话测试辅助函数"""
import os
import base64
import random
import json
from pathlib import Path


# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"


def get_all_files(directory: Path, extensions: list = None) -> list:
    """获取目录下所有指定扩展名的文件"""
    if not directory.exists():
        return []
    
    files = []
    for f in directory.iterdir():
        if f.is_file() and f.name != ".gitkeep" and not f.name.endswith("_urls.txt"):
            if extensions is None or f.suffix.lower() in extensions:
                files.append(f)
    return files


def get_all_images() -> list:
    """获取所有测试图片"""
    return get_all_files(IMAGES_DIR, [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"])


def get_all_videos() -> list:
    """获取所有测试视频"""
    return get_all_files(VIDEOS_DIR, [".mp4", ".avi", ".mov", ".mkv", ".webm"])


def get_all_audios() -> list:
    """获取所有测试音频"""
    return get_all_files(AUDIO_DIR, [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"])


def get_random_images(count: int = 1) -> list:
    """随机获取指定数量的测试图片（支持复用）"""
    all_images = get_all_images()
    if not all_images:
        return []
    
    if len(all_images) >= count:
        return random.sample(all_images, count)
    else:
        return (all_images * ((count // len(all_images)) + 1))[:count]


def get_random_videos(count: int = 1) -> list:
    """随机获取指定数量的测试视频（支持复用）"""
    all_videos = get_all_videos()
    if not all_videos:
        return []
    
    if len(all_videos) >= count:
        return random.sample(all_videos, count)
    else:
        return (all_videos * ((count // len(all_videos)) + 1))[:count]


def get_random_audios(count: int = 1) -> list:
    """随机获取指定数量的测试音频（支持复用）"""
    all_audios = get_all_audios()
    if not all_audios:
        return []
    
    if len(all_audios) >= count:
        return random.sample(all_audios, count)
    else:
        return (all_audios * ((count // len(all_audios)) + 1))[:count]


def get_urls_from_file(file_path: Path) -> list:
    """从URL文件中获取所有URL（每行一个）"""
    if not file_path.exists():
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


def get_image_urls() -> list:
    """获取所有图片URL"""
    return get_urls_from_file(IMAGES_DIR / "image_urls.txt")


def get_video_urls() -> list:
    """获取所有视频URL"""
    return get_urls_from_file(VIDEOS_DIR / "video_urls.txt")


def get_audio_urls() -> list:
    """获取所有音频URL"""
    return get_urls_from_file(AUDIO_DIR / "audio_urls.txt")


def get_random_image_urls(count: int = 1) -> list:
    """随机获取指定数量的图片URL（支持复用）"""
    all_urls = get_image_urls()
    if not all_urls:
        return []
    
    if len(all_urls) >= count:
        return random.sample(all_urls, count)
    else:
        return (all_urls * ((count // len(all_urls)) + 1))[:count]


def get_random_video_urls(count: int = 1) -> list:
    """随机获取指定数量的视频URL（支持复用）"""
    all_urls = get_video_urls()
    if not all_urls:
        return []
    
    if len(all_urls) >= count:
        return random.sample(all_urls, count)
    else:
        return (all_urls * ((count // len(all_urls)) + 1))[:count]


def get_random_audio_urls(count: int = 1) -> list:
    """随机获取指定数量的音频URL（支持复用）"""
    all_urls = get_audio_urls()
    if not all_urls:
        return []
    
    if len(all_urls) >= count:
        return random.sample(all_urls, count)
    else:
        return (all_urls * ((count // len(all_urls)) + 1))[:count]


def file_to_base64(file_path: Path) -> str:
    """将文件转换为base64编码"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_base64_with_mime(image_path: Path) -> tuple:
    """获取图片的base64编码和MIME类型"""
    suffix = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }
    mime_type = mime_map.get(suffix, "image/jpeg")
    b64_data = file_to_base64(image_path)
    return b64_data, mime_type


def get_video_base64_with_mime(video_path: Path) -> tuple:
    """获取视频的base64编码和MIME类型"""
    suffix = video_path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm"
    }
    mime_type = mime_map.get(suffix, "video/mp4")
    b64_data = file_to_base64(video_path)
    return b64_data, mime_type


def get_audio_base64_with_mime(audio_path: Path) -> tuple:
    """获取音频的base64编码和MIME类型"""
    suffix = audio_path.suffix.lower()
    mime_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4"
    }
    mime_type = mime_map.get(suffix, "audio/mpeg")
    b64_data = file_to_base64(audio_path)
    return b64_data, mime_type


def build_image_content(image_source, source_type: str = "url") -> dict:
    """构建图片消息内容
    
    Args:
        image_source: 图片路径(Path)或URL字符串或base64字符串
        source_type: "url" 或 "base64"
    
    Returns:
        dict: 图片消息内容
    """
    if source_type == "url":
        return {
            "type": "image_url",
            "image_url": {"url": image_source}
        }
    else:
        # 如果是字符串类型，直接使用（可能是已包含MIME类型的base64数据或纯base64）
        if isinstance(image_source, str):
            # 如果已经包含data:前缀，直接使用
            if image_source.startswith("data:"):
                return {
                    "type": "image_url",
                    "image_url": {"url": image_source}
                }
            # 如果包含MIME类型前缀但缺少前缀，添加data:前缀
            elif ";base64," in image_source:
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_source}"}
                }
            else:
                # 纯base64数据，添加data:前缀和默认MIME类型
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_source}"}
                }
        else:
            # Path类型，读取文件并转换
            b64_data, mime_type = get_image_base64_with_mime(image_source)
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
            }


def build_video_content(video_source, source_type: str = "url") -> dict:
    """构建视频消息内容
    
    Args:
        video_source: 视频路径(Path)或URL字符串或base64字符串
        source_type: "url" 或 "base64"
    
    Returns:
        dict: 视频消息内容
    """
    if source_type == "url":
        return {
            "type": "video_url",
            "video_url": {"url": video_source}
        }
    else:
        # 如果是字符串类型，直接使用（可能是已包含MIME类型的base64数据或纯base64）
        if isinstance(video_source, str):
            # 如果已经包含前缀，直接使用
            if video_source.startswith(""):
                return {
                    "type": "video_url",
                    "video_url": {"url": video_source}
                }
            # 如果包含MIME类型前缀但缺少前缀，添加前缀
            elif ";base64," in video_source:
                return {
                    "type": "video_url",
                    "video_url": {"url": f"{video_source}"}
                }
            else:
                # 纯base64数据，添加前缀和默认MIME类型
                return {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_source}"}
                }
        else:
            # Path类型，读取文件并转换
            b64_data, mime_type = get_video_base64_with_mime(video_source)
            return {
                "type": "video_url",
                "video_url": {"url": f"data:{mime_type};base64,{b64_data}"}
            }


def build_audio_content(audio_source, source_type: str = "url") -> dict:
    """构建音频消息内容
    
    Args:
        audio_source: 音频路径(Path)或URL字符串或base64字符串
        source_type: "url" 或 "base64"
    
    Returns:
        dict: 音频消息内容
    """
    if source_type == "url":
        return {
            "type": "audio_url",
            "audio_url": {"url": audio_source}
        }
    else:
        # 如果是字符串类型，直接使用（可能是已包含MIME类型的base64数据或纯base64）
        if isinstance(audio_source, str):
            # 如果已经包含前缀，直接使用
            if audio_source.startswith("data:"):
                return {
                    "type": "audio_url",
                    "audio_url": {"url": audio_source}
                }
            # 如果包含MIME类型前缀但缺少前缀，添加前缀
            elif ";base64," in audio_source:
                return {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:{audio_source}"}
                }
            else:
                # 纯base64数据，添加前缀和默认MIME类型
                return {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/mpeg;base64,{audio_source}"}
                }
        else:
            # Path类型，读取文件并转换
            b64_data, mime_type = get_audio_base64_with_mime(audio_source)
            return {
                "type": "audio_url",
                "audio_url": {"url": f"data:{mime_type};base64,{b64_data}"}
            }


def build_multimodal_message(prompt: str, images: list = None, videos: list = None, 
                              audios: list = None, source_type: str = "url") -> dict:
    """构建多模态消息
    
    Args:
        prompt: 文本提示
        images: 图片列表（Path列表或URL字符串列表）
        videos: 视频列表（Path列表或URL字符串列表）
        audios: 音频列表（Path列表或URL字符串列表）
        source_type: "url" 或 "base64"
    
    Returns:
        dict: 多模态消息
    """
    content = [{"type": "text", "text": prompt}]
    
    if images:
        for img in images:
            content.append(build_image_content(img, source_type))
    
    if videos:
        for video in videos:
            content.append(build_video_content(video, source_type))
    
    if audios:
        for audio in audios:
            content.append(build_audio_content(audio, source_type))
    
    return {"role": "user", "content": content}


def attach_multimodal_request(request_body: dict, name: str = "请求体"):
    """Allure记录多模态请求体
    
    对于包含base64的数据，隐藏base64内容，只显示长度信息
    """
    with allure.step(name):
        # 深拷贝请求体，避免修改原始数据
        display_body = json.loads(json.dumps(request_body))
        
        # 遍历messages，隐藏base64内容
        if "messages" in display_body:
            for message in display_body["messages"]:
                if "content" in message and isinstance(message["content"], list):
                    for item in message["content"]:
                        # 处理图片URL
                        if item.get("type") == "image_url" and "image_url" in item:
                            url = item["image_url"].get("url", "")
                            if url and (";base64," in url or url.startswith("data:")):
                                item["image_url"]["url"] = f"<base64 data, length={len(url)}>"
                        # 处理视频URL
                        if item.get("type") == "video_url" and "video_url" in item:
                            url = item["video_url"].get("url", "")
                            if url and ";base64," in url:
                                item["video_url"]["url"] = f"<base64 data, length={len(url)}>"
                        # 处理音频URL
                        if item.get("type") == "audio_url" and "audio_url" in item:
                            url = item["audio_url"].get("url", "")
                            if url and (";base64," in url or url.startswith("data:")):
                                item["audio_url"]["url"] = f"<base64 data, length={len(url)}>"
        
        allure.attach(
            json.dumps(display_body, indent=2, ensure_ascii=False),
            name=name,
            attachment_type=allure.attachment_type.JSON,
            extension="json"
        )


def attach_response_body(response, name: str = "响应体"):
    """Allure记录响应体"""
    import requests
    response.encoding = 'utf-8'
    content_type = response.headers.get("Content-Type", "")

    with allure.step(name):
        if "application/json" in content_type:
            allure.attach(
                json.dumps(response.json(), indent=2, ensure_ascii=False),
                name=name,
                attachment_type=allure.attachment_type.JSON,
                extension="json"
            )
        else:
            allure.attach(
                response.text.encode("utf-8"),
                name=name,
                attachment_type=allure.attachment_type.TEXT
            )


def attach_source_files(images: list = None, videos: list = None, audios: list = None, 
                        source_type: str = "url", name: str = "源文件信息"):
    """Allure记录源文件信息，方便问题复现
    
    Args:
        images: 图片列表（Path列表或URL字符串列表或base64字符串列表）
        videos: 视频列表（Path列表或URL字符串列表或base64字符串列表）
        audios: 音频列表（Path列表或URL字符串列表或base64字符串列表）
        source_type: "url" 或 "base64"
        name: allure附件名称
    """
    source_info = {
        "source_type": source_type,
        "images": [],
        "videos": [],
        "audios": []
    }
    
    if images:
        for i, img in enumerate(images):
            if isinstance(img, Path):
                source_info["images"].append({
                    "index": i,
                    "type": "file",
                    "name": img.name,
                    "path": str(img)
                })
            elif isinstance(img, str):
                if img.startswith("http://") or img.startswith("https://"):
                    source_info["images"].append({
                        "index": i,
                        "type": "url",
                        "url": img
                    })
                else:
                    # base64 字符串
                    source_info["images"].append({
                        "index": i,
                        "type": "base64",
                        "length": len(img),
                        "preview": img[:100] + "..." if len(img) > 100 else img
                    })
    
    if videos:
        for i, video in enumerate(videos):
            if isinstance(video, Path):
                source_info["videos"].append({
                    "index": i,
                    "type": "file",
                    "name": video.name,
                    "path": str(video)
                })
            elif isinstance(video, str):
                if video.startswith("http://") or video.startswith("https://"):
                    source_info["videos"].append({
                        "index": i,
                        "type": "url",
                        "url": video
                    })
                else:
                    # base64 字符串
                    source_info["videos"].append({
                        "index": i,
                        "type": "base64",
                        "length": len(video),
                        "preview": video[:100] + "..." if len(video) > 100 else video
                    })
    
    if audios:
        for i, audio in enumerate(audios):
            if isinstance(audio, Path):
                source_info["audios"].append({
                    "index": i,
                    "type": "file",
                    "name": audio.name,
                    "path": str(audio)
                })
            elif isinstance(audio, str):
                if audio.startswith("http://") or audio.startswith("https://"):
                    source_info["audios"].append({
                        "index": i,
                        "type": "url",
                        "url": audio
                    })
                else:
                    # base64 字符串
                    source_info["audios"].append({
                        "index": i,
                        "type": "base64",
                        "length": len(audio),
                        "preview": audio[:100] + "..." if len(audio) > 100 else audio
                    })
    
    with allure.step(name):
        allure.attach(
            json.dumps(source_info, indent=2, ensure_ascii=False),
            name=name,
            attachment_type=allure.attachment_type.JSON,
            extension="json"
        )