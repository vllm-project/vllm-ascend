"""Ascend社区文档搜索技能 - 提供Ascend社区文档搜索功能"""
import base64
import json
import logging
from urllib.parse import urljoin

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AscendSearchSkill:
    """Ascend社区搜索技能主类"""

    def __init__(self, base_url="https://www.hiascend.com/"):
        self.base_url = base_url
        self.search_endpoint = "/ascendgateway/ascendservice/intelligent/search"
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": base_url,
        })
        self._csrf_token = None

    def search_documents(self,
                         keyword,
                         lang="zh",
                         doc_type="DOC",
                         page_num=1,
                         page_size=10,
                         sort=1,
                         ignore_correction=False,
                         search_type=True):
        """搜索Ascend社区文档"""
        min_page_num = 1
        max_page_num = 100
        min_page_size = 1
        max_page_size = 10

        is_keyword_empty = not keyword or not keyword.strip()
        if is_keyword_empty:
            return self._error_response("关键词参数是必需的")

        is_page_num_invalid = page_num < min_page_num or page_num > max_page_num
        if is_page_num_invalid:
            return self._error_response("页码必须在1到100之间")

        is_page_size_invalid = page_size < min_page_size or page_size > max_page_size
        if is_page_size_invalid:
            return self._error_response("页面大小必须在1到10之间")

        params = {
            "keyword": base64.b64encode(keyword.strip().encode('utf-8')).decode('utf-8'),
            "lang": lang,
            "type": doc_type,
            "pageNum": page_num,
            "pageSize": page_size,
            "sort": sort,
            "ignoreCorrection": ignore_correction,
            "searchType": search_type
        }

        try:
            self._ensure_session()
            return self._do_search(params)
        except requests.exceptions.RequestException as e:
            logger.error("API请求失败", exc_info=True)
            return self._error_response(f"API请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error("JSON响应解析失败", exc_info=True)
            return self._error_response(f"无效的JSON响应: {str(e)}")
        except Exception as e:
            logger.error("未知错误", exc_info=True)
            return self._error_response(f"未知错误: {str(e)}")

    def _ensure_session(self):
        """获取 session cookie 和 CSRF token"""
        if self._csrf_token:
            return

        init_url = urljoin(self.base_url, "/ascendgateway/ascendservice/commons/currentTime")
        try:
            resp = self._session.get(init_url, headers={"x-request-type": "machine"}, timeout=10)
            token = resp.headers.get("next-token")
            if not token:
                raise RuntimeError("未能获取 next-token")
            self._csrf_token = token
            logger.info("Session 初始化成功")
        except requests.exceptions.RequestException as e:
            logger.error("Session 初始化失败", exc_info=True)
            raise RuntimeError(f"Session 初始化失败: {e}") from e

    def _do_search(self, params):
        """执行搜索请求并解析响应"""
        full_url = urljoin(self.base_url, self.search_endpoint)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-request-type": "machine",
            "x-csrf-token": self._csrf_token,
            "locale": "zh-cn",
        }
        response = self._session.post(full_url, json=params,
                                      headers=headers, timeout=10)
        response.raise_for_status()

        result_data = response.json()
        is_success = result_data.get("success")
        has_data = "data" in result_data

        if not (is_success and has_data):
            return result_data

        data_content = result_data["data"]
        if not isinstance(data_content, dict) or "data" not in data_content:
            return result_data

        document_data = data_content["data"]
        if not isinstance(document_data, list):
            return result_data

        formatted_data = []
        for item in document_data:
            formatted_item = {
                "title": item.get("docTitle", ""),
                "summary": item.get("docContent", ""),
                "url": item.get("docUrl", ""),
                "version": item.get("version", ""),
                "publishTime": item.get("publishTime", "")
            }
            formatted_data.append(formatted_item)

        formatted_data = self._transform_urls(formatted_data)
        return {
            "success": True,
            "message": result_data.get("msg", "success"),
            "data": formatted_data
        }

    def _transform_urls(self, data):
        """将内部URL转换为公共URL"""
        transformed_data = []
        source_prefix = "/source/"
        document_prefix = "/document/detail/"

        for item in data:
            has_url = "url" in item and item["url"]
            if has_url:
                transformed_url = item["url"].replace(source_prefix,
                                                      document_prefix)
                is_relative_url = transformed_url.startswith("/")
                if is_relative_url:
                    transformed_url = urljoin(self.base_url, transformed_url)
                item["url"] = transformed_url

            transformed_data.append(item)

        return transformed_data

    def _error_response(self, message):
        """创建标准错误响应"""
        return {"success": False, "message": message, "data": []}
