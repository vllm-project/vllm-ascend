"""
Ascend文档内容获取器 - 完全替代WebFetch功能

功能：
- 获取Ascend社区文档的详细内容
- 解析HTML提取结构化信息
- 支持API文档、教程、示例等各类文档
"""
import json
import logging
import re
import html
from urllib.parse import urljoin
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AscendContentFetcher:
    """Ascend文档内容获取器"""

    def __init__(self, base_url="https://www.hiascend.com/"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    def fetch_document(self, url: str) -> Dict:
        """
        获取文档完整内容

        Args:
            url: 文档URL

        Returns:
            包含文档结构化信息的字典
        """
        try:
            logger.info(f"开始获取文档: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 提取文档信息
            document_info = {
                "url": url,
                "title": self._extract_title(soup),
                "description": self._extract_description(soup),
                "content_type": self._detect_content_type(soup),
                "main_content": self._extract_main_content(soup),
                "code_examples": self._extract_code_examples(soup),
                "api_details": self._extract_api_details(soup),
                "tables": self._extract_tables(soup),
                "links": self._extract_links(soup),
                "images": self._extract_images(soup),
                "metadata": self._extract_metadata(soup)
            }

            logger.info(f"文档获取成功: {document_info['title']}")
            return {
                "success": True,
                "message": "文档获取成功",
                "data": document_info
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {url}", exc_info=True)
            return self._error_response(f"请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"处理文档失败: {url}", exc_info=True)
            return self._error_response(f"处理失败: {str(e)}")

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取文档标题"""
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text().strip()
            title = re.sub(r'[-|]\s*昇腾社区$', '', title)
            return title

        h1_elem = soup.find('h1')
        if h1_elem:
            return h1_elem.get_text().strip()

        return ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """提取文档描述"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content'].strip()
            return self._clean_text(desc)  # 使用清理后的文本

        first_p = soup.find('p')
        if first_p:
            text = first_p.get_text().strip()
            if len(text) > 20:
                return self._clean_text(text)

        return ""

    def _detect_content_type(self, soup: BeautifulSoup) -> str:
        """检测内容类型"""
        title = self._extract_title(soup).lower()
        content = soup.get_text().lower()

        if any(keyword in title or keyword in content
               for keyword in ['api', '接口', '函数', '算子']):
            return "API文档"
        elif any(keyword in title or keyword in content
                for keyword in ['教程', 'guide', '入门', '学习']):
            return "教程"
        elif any(keyword in title or keyword in content
                for keyword in ['示例', 'example', 'demo', '样例']):
            return "示例"
        elif any(keyword in title or keyword in content
                for keyword in ['错误', '故障', '问题', '解决']):
            return "故障排除"
        else:
            return "文档"

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """提取主要内容"""
        content_selectors = [
            '.main-content',
            '.content',
            '.article-content',
            '#content',
            'main',
            'article',
            '.doc-content',
            '.document-content',
            '.api-content',  # API页面专用
            '.api-doc-content'
        ]

        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # 移除脚本、样式、导航、页脚等无关元素
                for elem in content_elem.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button']):
                    elem.decompose()

                text = content_elem.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    return self._clean_text(text)

        # 备用方案：提取body内容但过滤无关区域
        body = soup.find('body')
        if body:
            # 移除导航、页脚、脚本等区域
            for elem in body.find_all(['nav', 'footer', 'header', 'aside', 'script', 'style', 'form', 'button']):
                elem.decompose()

            text = body.get_text(separator=' ', strip=True)
            return self._clean_text(text)

        return ""

    def _extract_code_examples(self, soup: BeautifulSoup) -> List[Dict]:
        """提取代码示例"""
        code_examples = []

        code_elements = soup.find_all(['pre', 'code'])
        for i, elem in enumerate(code_elements):
            code_text = elem.get_text().strip()
            if code_text and len(code_text) > 10:
                language = self._detect_code_language(code_text)

                code_examples.append({
                    "id": i + 1,
                    "language": language,
                    "code": code_text,
                    "context": self._get_code_context(elem)
                })

        return code_examples

    def _extract_api_details(self, soup: BeautifulSoup) -> Dict:
        """提取API详细信息"""
        api_details = {
            "function_name": "",
            "description": "",
            "parameters": [],
            "return_value": "",
            "usage_example": ""
        }

        headers = soup.find_all(['h1', 'h2', 'h3'])
        for header in headers:
            text = header.get_text().strip()
            if re.search(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(', text):
                api_details["function_name"] = text
                break

        tables = soup.find_all('table')
        for table in tables:
            headers = [th.get_text().strip().lower() for th in table.find_all('th')]
            headers_text = ' '.join(headers)

            if any(keyword in headers_text for keyword in ['参数', 'parameter']):
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        param_name = cells[0].get_text().strip()
                        param_desc = cells[1].get_text().strip()
                        if param_name:
                            api_details["parameters"].append({
                                "name": param_name,
                                "description": param_desc
                            })

            elif any(keyword in headers_text for keyword in ['返回值', 'return']):
                rows = table.find_all('tr')[1:]
                if rows:
                    cells = rows[0].find_all('td')
                    if cells:
                        api_details["return_value"] = cells[0].get_text().strip()

        return api_details

    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """提取表格数据"""
        tables_data = []

        for i, table in enumerate(soup.find_all('table')):
            table_data = {
                "id": i + 1,
                "headers": [],
                "rows": []
            }

            headers = table.find_all('th')
            if headers:
                table_data["headers"] = [th.get_text().strip() for th in headers]

            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells and not all(cell.name == 'th' for cell in cells):
                    row_data = []
                    for cell in cells:
                        # 检查单元格中是否有链接
                        links_in_cell = cell.find_all('a')
                        cell_text = cell.get_text(strip=True)

                        if links_in_cell:
                            # 提取所有链接
                            cell_links = []
                            for link in links_in_cell:
                                href = link.get('href', '')
                                if href:
                                    if href.startswith('/'):
                                        href = urljoin(self.base_url, href)
                                    cell_links.append({
                                        "text": link.get_text(strip=True),
                                        "href": href,
                                        "title": link.get('title', '')
                                    })

                            if cell_links:
                                # 如果有链接，存储链接信息
                                row_data.append({
                                    "text": cell_text,
                                    "links": cell_links
                                })
                            else:
                                row_data.append(cell_text)
                        else:
                            row_data.append(cell_text)

                    table_data["rows"].append(row_data)

            if table_data["headers"] or table_data["rows"]:
                tables_data.append(table_data)

        return tables_data

    def _extract_links(self, soup: BeautifulSoup) -> List[Dict]:
        """提取文档中的所有链接"""
        links = []

        for i, a_tag in enumerate(soup.find_all('a')):
            href = a_tag.get('href', '')
            text = a_tag.get_text(strip=True)
            title = a_tag.get('title', '')

            if not href:
                continue

            # 处理相对URL
            if href.startswith('/'):
                href = urljoin(self.base_url, href)
            elif href.startswith('#'):
                # 页面内锚点链接，跳过或处理
                continue

            # 过滤常见的不需要的外部链接
            if href.startswith('javascript:'):
                continue

            links.append({
                "id": i + 1,
                "text": text,
                "href": href,
                "title": title,
                "context": self._get_link_context(a_tag)
            })

        return links

    def _get_link_context(self, a_tag) -> str:
        """获取链接上下文信息"""
        # 获取父元素或前一个元素作为上下文
        parent = a_tag.find_parent(['p', 'li', 'td', 'th', 'div'])
        if parent:
            # 获取父元素文本，但移除当前链接文本
            parent_text = parent.get_text(strip=True)
            if a_tag.text in parent_text:
                # 简单提取链接前后的文字
                text = parent.get_text()
                link_text = a_tag.text
                if link_text and text:
                    idx = text.find(link_text)
                    if idx != -1:
                        start = max(0, idx - 50)
                        end = min(len(text), idx + len(link_text) + 50)
                        return text[start:end].strip()
        return ""

    def _extract_images(self, soup: BeautifulSoup) -> List[Dict]:
        """提取图片信息（已过滤，返回空列表）"""
        # 根据用户需求，过滤掉所有图片信息
        return []

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """提取元数据"""
        metadata = {}

        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content

        return metadata

    def _detect_code_language(self, code_text: str) -> str:
        """检测代码语言"""
        code_lower = code_text.lower()

        if re.search(r'#include\s*<', code_text):
            return "C++"
        elif re.search(r'def\s+\w+\s*\(|import\s+\w+', code_text):
            return "Python"
        elif re.search(r'function\s+\w+\s*\(|var\s+\w+\s*=', code_text):
            return "JavaScript"
        elif re.search(r'public\s+class|private\s+\w+', code_text):
            return "Java"
        else:
            return "Unknown"

    def _get_code_context(self, code_elem) -> str:
        """获取代码上下文"""
        prev_elem = code_elem.find_previous(['h1', 'h2', 'h3', 'h4', 'p'])
        if prev_elem:
            return prev_elem.get_text().strip()[:100]
        return ""

    def _clean_text(self, text: str) -> str:
        """清理文本 - 增强版，保留技术文档中的代码结构"""
        # 1. 移除脚本、样式标签和XML声明（多行内容）
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\?xml[^>]*\?>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # 2. 解码HTML实体（优先处理，将&lt;等转换为<）
        text = html.unescape(text)

        # 3. 移除剩余的完整HTML标签，但保留简单的尖括号对（可能是C++模板）
        # 匹配完整的HTML标签：以<开头，包含字母，可能有属性，以>结尾
        # 但不匹配简单的<word>形式（可能是C++模板）
        # 先移除常见的HTML标签
        html_tags = [
            'div', 'span', 'p', 'a', 'img', 'br', 'hr', 'ul', 'ol', 'li',
            'table', 'tr', 'td', 'th', 'form', 'input', 'button', 'nav',
            'footer', 'header', 'aside', 'section', 'article', 'main',
            'meta', 'link', 'title', 'head', 'body', 'html', 'strong',
            'em', 'b', 'i', 'u', 'code', 'pre', 'blockquote'
        ]
        for tag in html_tags:
            text = re.sub(f'</?{tag}[^>]*>', '', text, flags=re.IGNORECASE)

        # 4. 替换常见实体（双重保险）
        entities = {
            '&nbsp;': ' ', '&lt;': '<', '&gt;': '>', '&amp;': '&',
            '&quot;': '"', '&#39;': "'", '&apos;': "'", '&cent;': '¢',
            '&pound;': '£', '&yen;': '¥', '&euro;': '€', '&copy;': '(c)',
            '&reg;': '(R)', '&trade;': '(TM)', '&times;': '×', '&divide;': '÷',
            '&mdash;': '—', '&ndash;': '–', '&hellip;': '...'
        }
        for entity, replacement in entities.items():
            text = text.replace(entity, replacement)

        # 5. 移除十六进制和十进制实体
        text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)

        # 6. 修复C++模板中的多余空格：< int64_t > -> <int64_t>, < srcType , false> -> <srcType, false>
        # 匹配尖括号内有空格的情况，但避免影响其他情况
        def fix_template_spaces(match):
            inner = match.group(1)
            # 移除内部开头和结尾的空格
            inner = re.sub(r'^\s+|\s+$', '', inner)
            # 修复逗号周围的空格：移除逗号前的空格，保留逗号后的一个空格
            inner = re.sub(r'\s*,\s*', ', ', inner)
            # 合并多个连续空格为一个
            inner = re.sub(r'\s+', ' ', inner)
            return f'<{inner}>'

        # 应用修复：匹配<...>，但不匹配已经紧凑的模板
        text = re.sub(r'<\s*([^>]+?)\s*>', fix_template_spaces, text)

        # 7. 合并空白字符
        text = re.sub(r'\s+', ' ', text)

        # 8. 移除控制字符和特殊空白
        text = re.sub(r'[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f]', '', text)

        return text.strip()

    def _error_response(self, message: str) -> Dict:
        """创建错误响应"""
        return {
            "success": False,
            "message": message,
            "data": {}
        }

def main():
    """命令行入口函数"""
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("用法: python ascend_content_fetcher.py <URL>")
        print("示例: python ascend_content_fetcher.py https://www.hiascend.com/document/detail/zh/ModelZoo/Documentation/qs/qs_0002.html")
        sys.exit(1)
    
    url = sys.argv[1]
    fetcher = AscendContentFetcher()
    
    try:
        result = fetcher.fetch_document(url)
        
        # 直接输出JSON格式的结果，不保存文件
        print(json.dumps(result, ensure_ascii=False, indent=2))
            
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()