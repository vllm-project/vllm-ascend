#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
Markdown 文件清理脚本
用于清理 Markdown 文件中的冗余 HTML 内容
"""

import os
import re
import argparse
import shutil
from pathlib import Path
from html.parser import HTMLParser
from html.entities import name2codepoint


class HTMLTableParser(HTMLParser):
    """解析 HTML 表格并转换为 Markdown 格式"""

    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_row = False
        self.in_cell = False
        self.is_header = False
        self.current_row = []
        self.rows = []
        self.header_row = None
        self.cell_content = ""
        self.result = ""
        self.list_depth = 0  # 追踪列表嵌套深度
        self.in_p = False  # 追踪是否在段落内
        self.has_thead = False  # 追踪是否有 thead 标签

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.result = ""
            self.has_thead = False
        elif tag == "thead":
            self.in_thead = True
            self.has_thead = True
        elif tag == "tbody":
            self.in_tbody = True
        elif tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in ("th", "td"):
            self.in_cell = True
            self.is_header = (tag == "th")
            self.cell_content = ""
            # 处理 colspan 和 rowspan（简化处理）
            self.colspan = 1
            self.rowspan = 1
            for name, value in attrs:
                if name == "colspan":
                    self.colspan = int(value)
                elif name == "rowspan":
                    self.rowspan = int(value)
        elif tag == "ul" or tag == "ol":
            if self.in_cell:
                self.list_depth += 1
                # 如果单元格已有内容，添加换行
                if self.cell_content.strip():
                    self.cell_content += "<br>"
        elif tag == "li":
            if self.in_cell:
                # 使用 <br> 作为行内换行标记，后续会处理
                indent = "  " * (self.list_depth - 1) if self.list_depth > 0 else ""
                self.cell_content += f"<br>{indent}- "
        elif tag == "p":
            self.in_p = True
            # 段落内容开始时无需特殊处理

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
            # 生成 Markdown 表格
            # 如果没有 thead，把第一行作为表头
            if not self.header_row and self.rows:
                self.header_row = self.rows.pop(0)
            if self.header_row:
                self.result = self._generate_markdown_table()
        elif tag == "thead":
            self.in_thead = False
            if self.current_row:
                self.header_row = self.current_row
                self.current_row = []
        elif tag == "tbody":
            self.in_tbody = False
        elif tag == "tr":
            self.in_row = False
            if self.current_row:
                if self.in_thead and not self.header_row:
                    self.header_row = self.current_row
                else:
                    self.rows.append(self.current_row)
                self.current_row = []
        elif tag in ("th", "td"):
            self.in_cell = False
            # 清理单元格内容
            content = self.cell_content.strip()
            # 处理 colspan
            cells = [content] + [""] * (self.colspan - 1)
            self.current_row.extend(cells)
            self.cell_content = ""
        elif tag == "ul" or tag == "ol":
            if self.in_cell and self.list_depth > 0:
                self.list_depth -= 1
        elif tag == "p":
            self.in_p = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell_content += data

    def _generate_markdown_table(self):
        """生成 Markdown 表格格式"""
        if not self.header_row:
            return ""

        # 计算列数
        col_count = len(self.header_row)

        # 生成表头
        header = "| " + " | ".join(self.header_row) + " |"
        separator = "| " + " | ".join(["---"] * col_count) + " |"

        # 生成数据行
        data_rows = []
        for row in self.rows:
            # 补齐列数
            while len(row) < col_count:
                row.append("")
            # 处理单元格内的换行，将 <br> 转换为实际的换行转义
            processed_row = []
            for cell in row:
                # 将 <br> 替换为 HTML 实体的换行，以便在 Markdown 表格中显示
                cell = cell.replace('<br>', '<br/>')
                processed_row.append(cell)
            data_rows.append("| " + " | ".join(processed_row) + " |")

        return "\n".join([header, separator] + data_rows)


def remove_anchor_tags(content: str) -> str:
    """移除锚点标签 <a name="..."></a>"""
    # 移除空锚点
    content = re.sub(r'<a\s+name="[^"]*">\s*</a>', '', content, flags=re.IGNORECASE)
    # 移除锚点开始标签（非空的情况）
    content = re.sub(r'<a\s+name="[^"]*">', '', content, flags=re.IGNORECASE)
    # 移除多余的 </a>
    # 注意：要小心不要移除正常的链接
    return content


def convert_html_tables(content: str) -> str:
    """将 HTML 表格转换为 Markdown 表格"""

    def replace_table(match):
        table_html = match.group(0)
        parser = HTMLTableParser()
        try:
            parser.feed(table_html)
            if parser.result:
                return "\n" + parser.result + "\n"
        except Exception:
            pass
        return table_html  # 解析失败则保留原样

    # 匹配整个 table 标签（包括嵌套）
    pattern = r'<table[^>]*>.*?</table>'
    return re.sub(pattern, replace_table, content, flags=re.DOTALL | re.IGNORECASE)


def remove_term_tags(content: str) -> str:
    """移除术语标签 <term id="...">...</term>，保留内容"""
    # 移除 term 标签
    content = re.sub(r'<term\s+id="[^"]*">', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</term>', '', content, flags=re.IGNORECASE)
    return content


def remove_paragraph_id_tags(content: str) -> str:
    """移除段落和 span 的 id 属性标签"""
    # 移除 p 标签的 id 属性（保留标签本身）
    content = re.sub(r'<p\s+id="[^"]*">', '<p>', content, flags=re.IGNORECASE)

    # 移除 span 标签（保留内容）
    content = re.sub(r'<span\s+[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</span>', '', content, flags=re.IGNORECASE)

    # 移除 div 标签（保留内容）
    content = re.sub(r'<div\s+[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</div>', '', content, flags=re.IGNORECASE)

    return content


def clean_html_entities(content: str) -> str:
    """清理 HTML 实体"""
    # &nbsp; 替换为空格
    content = content.replace('&nbsp;', ' ')
    content = content.replace('&#160;', ' ')

    # 其他常见实体
    content = content.replace('&lt;', '<')
    content = content.replace('&gt;', '>')
    content = content.replace('&amp;', '&')
    content = content.replace('&quot;', '"')
    content = content.replace('&apos;', "'")

    return content


def remove_html_attributes(content: str) -> str:
    """移除残留的 HTML 属性"""
    # 移除 style 属性
    content = re.sub(r'\s+style="[^"]*"', '', content, flags=re.IGNORECASE)
    content = re.sub(r"\s+style='[^']*'", '', content, flags=re.IGNORECASE)

    # 移除 class 属性
    content = re.sub(r'\s+class="[^"]*"', '', content, flags=re.IGNORECASE)
    content = re.sub(r"\s+class='[^']*'", '', content, flags=re.IGNORECASE)

    # 移除 id 属性（在非特定标签上）
    content = re.sub(r'\s+id="[^"]*"', '', content, flags=re.IGNORECASE)
    content = re.sub(r"\s+id='[^']*'", '', content, flags=re.IGNORECASE)

    # 移除 width/height 属性
    content = re.sub(r'\s+width="[^"]*"', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\s+height="[^"]*"', '', content, flags=re.IGNORECASE)

    return content


def clean_extra_newlines(content: str) -> str:
    """清理多余的空行（3个及以上替换为2个）"""
    # 将 3 个及以上连续换行替换为 2 个
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 清理行尾空白
    content = re.sub(r'[ \t]+\n', '\n', content)

    # 清理文件末尾多余空白
    content = content.strip() + '\n'

    return content


def remove_other_html_tags(content: str) -> str:
    """移除其他常见的冗余 HTML 标签"""

    # 移除 <b> 和 </b>，保留内容（Markdown 使用 **）
    # 但如果内容已经是 ** 包裹的，则直接移除标签
    content = re.sub(r'<b>([^<]*)</b>', r'**\1**', content, flags=re.IGNORECASE)

    # 移除 <i> 和 </i>，保留内容（Markdown 使用 *）
    content = re.sub(r'<i>([^<]*)</i>', r'*\1*', content, flags=re.IGNORECASE)

    # 移除 <strong> 和 </strong>
    content = re.sub(r'<strong>([^<]*)</strong>', r'**\1**', content, flags=re.IGNORECASE)

    # 移除 <em> 和 </em>
    content = re.sub(r'<em>([^<]*)</em>', r'*\1*', content, flags=re.IGNORECASE)

    # 保留表格单元格内的 <br/>（用于换行），表格外的转换为换行
    # 先将表格内的 <br/> 转换为临时标记
    def protect_table_br(match):
        return match.group(0).replace('<br/>', '<<<BR_IN_TABLE>>>').replace('<br />', '<<<BR_IN_TABLE>>>')

    # 匹配表格行（简化版：匹配 |...| 格式的行）
    content = re.sub(r'\|[^|\n]+\|', protect_table_br, content)

    # 将非表格内的 <br> 转换为换行
    content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)

    # 恢复表格内的 <br/>
    content = content.replace('<<<BR_IN_TABLE>>>', '<br/>')

    # 移除 <hr> 标签
    content = re.sub(r'<hr\s*/?>', '\n---\n', content, flags=re.IGNORECASE)

    return content


def clean_markdown_file(filepath: str, backup: bool = True, quiet: bool = False) -> bool:
    """
    清理单个 Markdown 文件

    Args:
        filepath: 文件路径
        backup: 是否创建备份

    Returns:
        bool: 是否成功
    """
    try:
        # 读取文件
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 按顺序执行清理步骤
        # 1. 移除锚点标签
        content = remove_anchor_tags(content)

        # 2. 转换 HTML 表格
        content = convert_html_tables(content)

        # 3. 移除术语标签
        content = remove_term_tags(content)

        # 4. 移除段落和 span 的 id 属性
        content = remove_paragraph_id_tags(content)

        # 5. 移除其他 HTML 标签
        content = remove_other_html_tags(content)

        # 6. 清理 HTML 实体
        content = clean_html_entities(content)

        # 7. 移除 HTML 属性
        content = remove_html_attributes(content)

        # 8. 清理多余空行
        content = clean_extra_newlines(content)

        # 检查是否有变化
        if content == original_content:
            if not quiet:
                print(f"  [跳过] {filepath} (无需清理)")
            return True

        # 创建备份
        if backup:
            backup_path = filepath + '.bak'
            shutil.copy2(filepath, backup_path)

        # 写入清理后的内容
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        if not quiet:
            print(f"  [完成] {filepath}")
        return True

    except Exception as e:
        print(f"  [错误] {filepath}: {e}")
        return False


def clean_markdown_directory(dirpath: str, backup: bool = True, recursive: bool = True, quiet: bool = False) -> tuple:
    """
    清理目录下的所有 Markdown 文件

    Args:
        dirpath: 目录路径
        backup: 是否创建备份
        recursive: 是否递归处理子目录

    Returns:
        tuple: (成功数, 失败数, 跳过数)
    """
    success_count = 0
    fail_count = 0
    skip_count = 0

    if recursive:
        pattern = os.path.join(dirpath, '**', '*.md')
    else:
        pattern = os.path.join(dirpath, '*.md')

    # 查找所有 Markdown 文件
    from glob import glob
    files = glob(pattern, recursive=recursive)

    if not quiet:
        print(f"\n找到 {len(files)} 个 Markdown 文件")

    for filepath in sorted(files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original = f.read()

            result = clean_markdown_file(filepath, backup=backup, quiet=quiet)

            with open(filepath, 'r', encoding='utf-8') as f:
                cleaned = f.read()

            if original == cleaned:
                skip_count += 1
            elif result:
                success_count += 1
            else:
                fail_count += 1

        except Exception:
            fail_count += 1

    return success_count, fail_count, skip_count


def main():
    parser = argparse.ArgumentParser(
        description='清理 Markdown 文件中的冗余 HTML 内容',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 清理指定目录下的所有 Markdown 文件
  python clean_markdown.py --dir "AscendC算子开发指南/"

  # 清理单个文件
  python clean_markdown.py --file "test.md"

  # 不创建备份直接清理
  python clean_markdown.py --dir "docs/" --no-backup

  # 只处理当前目录（不递归）
  python clean_markdown.py --dir "docs/" --no-recursive
        '''
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dir', '-d', help='要清理的目录路径')
    group.add_argument('--file', '-f', help='要清理的单个文件路径')

    parser.add_argument('--no-backup', action='store_true',
                        help='不创建备份文件（默认创建 .bak 备份）')
    parser.add_argument('--no-recursive', action='store_true',
                        help='不递归处理子目录（默认递归）')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式，只输出最终统计')

    args = parser.parse_args()

    backup = not args.no_backup
    quiet = args.quiet

    if not quiet:
        print("=" * 60)
        print("Markdown 文件清理工具")
        print("=" * 60)

    if args.file:
        # 清理单个文件
        if not quiet:
            print(f"\n处理文件: {args.file}")
        success = clean_markdown_file(args.file, backup=backup, quiet=quiet)
        if success:
            print("\n清理完成!")
        else:
            print("\n清理失败!")
            return 1
    else:
        # 清理目录
        if not quiet:
            print(f"\n处理目录: {args.dir}")
            print(f"递归: {'是' if not args.no_recursive else '否'}")
            print(f"备份: {'是' if backup else '否'}")
            print("-" * 60)

        success, fail, skip = clean_markdown_directory(
            args.dir,
            backup=backup,
            recursive=not args.no_recursive,
            quiet=quiet
        )

        if not quiet:
            print("-" * 60)
        print(f"\n处理完成! 成功: {success}, 失败: {fail}, 跳过: {skip}")

    return 0


if __name__ == '__main__':
    exit(main())
