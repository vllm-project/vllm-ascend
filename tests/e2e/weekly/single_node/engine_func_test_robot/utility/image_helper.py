"""图像编辑/生成接口辅助函数"""
import os
import base64
import json
import struct
from pathlib import Path


# 测试图片目录
IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"
# 图片URL文件
IMAGE_URLS_FILE = IMAGES_DIR / "image_urls.txt"


def get_test_image_path(image_name: str) -> str:
    """获取测试图片的完整路径"""
    return str(IMAGES_DIR / image_name)


def get_all_test_images() -> list:
    """获取所有测试图片路径列表"""
    images = []
    for f in IMAGES_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
            images.append(str(f))
    return images


def get_test_image_urls() -> list:
    """获取所有测试图片URL列表

    Returns:
        list: 图片URL列表，每行一个URL
    """
    if not IMAGE_URLS_FILE.exists():
        return []

    urls = []
    with open(IMAGE_URLS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            url = line.strip()
            if url and not url.startswith('#'):  # 跳过空行和注释行
                urls.append(url)
    return urls


def get_random_test_image_url() -> str:
    """随机获取一个测试图片URL

    Returns:
        str: 随机图片URL，如果没有可用URL则返回None
    """
    import random
    urls = get_test_image_urls()
    if not urls:
        return None
    return random.choice(urls)


def get_random_test_image_urls(count: int = 1) -> list:
    """随机获取指定数量的测试图片URL（支持复用）

    Args:
        count: 需要获取的URL数量

    Returns:
        list: 随机图片URL列表，如果没有可用URL则返回空列表
    """
    import random
    urls = get_test_image_urls()
    if not urls:
        return []

    # 如果URL不足，通过复用现有URL达到所需数量
    if len(urls) >= count:
        return random.sample(urls, count)
    else:
        return (urls * ((count // len(urls)) + 1))[:count]


def image_to_base64(image_path: str) -> str:
    """将图片文件转换为base64字符串（带data URI前缀）"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    # 根据文件扩展名确定MIME类型
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/png')
    b64_str = base64.b64encode(image_data).decode('utf-8')
    return f"{mime_type};base64,{b64_str}"


def send_image_edit_request(api_client, data: dict, image_files: list = None):
    """发送图像编辑请求（multipart/form-data格式）

    Args:
        api_client: HTTP客户端实例
        data: 表单数据字典
        image_files: 图片文件路径列表

    Returns:
        response: 响应对象
    """
    files = None
    if image_files:
        # 构建multipart/form-data的files参数
        files = []
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            files.append(('image[]', (img_name, open(img_path, 'rb'), 'image/png')))

    return api_client.post("/v1/images/edits", files=files, data=data, headers=None)


def attach_multipart_request(data: dict = {}, image_files: list = None, name: str = "请求体"):
    """Allure记录multipart请求体"""
    with allure.step(name):
        request_info = {"form_data": data}
        if image_files:
            request_info["images"] = [os.path.basename(f) for f in image_files]
        allure.attach(
            json.dumps(request_info, indent=2, ensure_ascii=False),
            name=name,
            attachment_type=allure.attachment_type.JSON,
            extension="json"
        )


def attach_response(response, name: str = "响应体"):
    """Allure记录响应体"""
    response.encoding = 'utf-8'
    content_type = response.headers.get("Content-Type", "")

    with allure.step(name):
        if "application/json" in content_type:
            # 对于图片响应，隐藏b64_json内容（太长）
            resp_json = response.json()
            if "data" in resp_json:
                for item in resp_json.get("data", []):
                    if "b64_json" in item and item["b64_json"]:
                        item["b64_json"] = f"<base64 data, length={len(item['b64_json'])}>"
            allure.attach(
                json.dumps(resp_json, indent=2, ensure_ascii=False),
                name=name,
                attachment_type=allure.attachment_type.JSON,
                extension="json"
            )
        else:
            allure.attach(
                response.text[:2000].encode("utf-8"),  # 限制长度
                name=name,
                attachment_type=allure.attachment_type.TEXT
            )


def decode_b64_image(b64_json: str) -> bytes:
    """解码base64图片数据"""
    # 移除可能的data URI前缀
    if b64_json.startswith('image'):
        b64_json = b64_json.split(',', 1)[1]
    return base64.b64decode(b64_json)


def get_image_dimensions(image_data: bytes) -> tuple:
    """获取图片尺寸（宽，高）

    支持PNG、JPEG、WebP格式
    """
    # PNG格式
    if image_data[:8] == b'\x89PNG\r\n\x1a\n':
        # PNG IHDR chunk包含宽高信息
        width = struct.unpack('>I', image_data[16:20])[0]
        height = struct.unpack('>I', image_data[20:24])[0]
        return width, height

    # JPEG格式
    if image_data[:2] == b'\xff\xd8':
        # JPEG需要解析SOI后的各个marker
        pos = 2
        while pos < len(image_data):
            if image_data[pos] != 0xff:
                break
            marker = image_data[pos + 1]
            if marker == 0xc0 or marker == 0xc2:  # SOF0 or SOF2
                height = struct.unpack('>H', image_data[pos + 5:pos + 7])[0]
                width = struct.unpack('>H', image_data[pos + 7:pos + 9])[0]
                return width, height
            # 跳过当前segment
            if marker == 0xd8 or marker == 0xd9:  # SOI or EOI
                pos += 2
            elif marker == 0xff:  # 填充字节
                pos += 1
            else:
                seg_len = struct.unpack('>H', image_data[pos + 2:pos + 4])[0]
                pos += 2 + seg_len

    # WebP格式
    if image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
        # WebP VP8/VP8L/VP8X格式解析
        chunk_type = image_data[12:16]
        if chunk_type == b'VP8 ':
            # VP8 bitstream
            width = struct.unpack('<H', image_data[26:28])[0] & 0x3fff
            height = struct.unpack('<H', image_data[28:30])[0] & 0x3fff
            return width, height
        elif chunk_type == b'VP8L':
            # VP8L (lossless)
            bits = struct.unpack('<I', image_data[21:25])[0]
            width = (bits & 0x3fff) + 1
            height = ((bits >> 14) & 0x3fff) + 1
            return width, height
        elif chunk_type == b'VP8X':
            # VP8X (extended)
            width = struct.unpack('<I', b'\x00' + image_data[24:27])[0] + 1
            height = struct.unpack('<I', b'\x00' + image_data[27:30])[0] + 1
            return width, height

    # 无法解析，返回None
    return None, None


def parse_size_string(size_str: str) -> tuple:
    """解析size字符串，返回(宽, 高)元组

    Args:
        size_str: 格式如 "1024x1024" 或 "1024X1024"

    Returns:
        (width, height) 元组
    """
    if not size_str:
        return None, None
    parts = size_str.lower().split('x')
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    return None, None


# ============== 图像生成接口相关函数 ==============

def send_image_generation_request(api_client, data: dict):
    """发送图像生成请求（JSON格式）

    Args:
        api_client: HTTP客户端实例
        data: 请求数据字典

    Returns:
        response: 响应对象
    """
    return api_client.post("/v1/images/generations", json=data)


def attach_generation_request(data: dict, name: str = "请求体"):
    """Allure记录图像生成请求体"""
    with allure.step(name):
        allure.attach(
            json.dumps(data, indent=2, ensure_ascii=False),
            name=name,
            attachment_type=allure.attachment_type.JSON,
            extension="json"
        )


def assert_image_generation_response_fields(response, msg=""):
    """校验图像生成接口响应字段完整性

    Args:
        response: HTTP响应对象
        msg: 错误消息前缀

    Returns:
        resp_json: 响应JSON
    """
    from engine_func_test_robot.utility import assertion

    resp_json = response.json()

    # 校验顶层字段
    assertion.check.is_true("created" in resp_json, f"{msg}响应应包含created字段")
    assertion.check.is_true("data" in resp_json, f"{msg}响应应包含data字段")

    # 校验output_format字段
    assertion.check.is_true("output_format" in resp_json, f"{msg}响应应包含output_format字段")
    output_format = resp_json.get("output_format")
    assertion.check.is_in(
        output_format.lower() if output_format else "",
        ["png", "jpeg", "webp"],
        f"{msg}output_format必须为png/jpeg/webp之一，实际为{output_format}"
    )

    # 校验size字段
    assertion.check.is_true("size" in resp_json, f"{msg}响应应包含size字段")
    size = resp_json.get("size")
    width, height = parse_size_string(size) if size else (None, None)
    assertion.check.is_true(
        width is not None and height is not None,
        f"{msg}size格式必须为WxH（如1024x1024），实际为{size}"
    )

    # 校验data数组
    data = resp_json.get("data", [])
    assertion.check.is_true(len(data) > 0, f"{msg}data应包含至少一个结果")

    # 校验data数组元素字段
    for idx, item in enumerate(data):
        has_b64 = "b64_json" in item and item["b64_json"]
        has_url = "url" in item and item["url"]
        assertion.check.is_true(has_b64 or has_url, f"{msg}data[{idx}]应包含b64_json或url字段")
        # revised_prompt可以为null
        assertion.check.is_true("revised_prompt" in item, f"{msg}data[{idx}]应包含revised_prompt字段")

    return resp_json


def decode_and_validate_image(b64_json: str, expected_format: str = None, expected_size: tuple = None, msg: str = ""):
    """解码base64图片并验证格式和尺寸

    Args:
        b64_json: base64编码的图片数据
        expected_format: 期望的格式（png/jpeg/webp）
        expected_size: 期望的尺寸元组(width, height)
        msg: 错误消息前缀

    Returns:
        tuple: (image_data, width, height)
    """
    # 解码base64
    img_data = decode_b64_image(b64_json)

    # 获取尺寸
    width, height = get_image_dimensions(img_data)

    if expected_size:
        exp_width, exp_height = expected_size
        if exp_width and width:
            assert width == exp_width, f"{msg}图片实际宽度{width}与期望宽度{exp_width}不一致"
            assert height == exp_height, f"{msg}图片实际高度{height}与期望高度{exp_height}不一致"

    if expected_format:
        # 根据文件头验证格式
        if expected_format.lower() in ['png']:
            assert img_data[:8] == b'\x89PNG\r\n\x1a\n', f"{msg}图片格式不是PNG"
        elif expected_format.lower() in ['jpg', 'jpeg']:
            assert img_data[:2] == b'\xff\xd8', f"{msg}图片格式不是JPEG"
        elif expected_format.lower() == 'webp':
            assert img_data[:4] == b'RIFF' and img_data[8:12] == b'WEBP', f"{msg}图片格式不是WebP"

    return img_data, width, height
