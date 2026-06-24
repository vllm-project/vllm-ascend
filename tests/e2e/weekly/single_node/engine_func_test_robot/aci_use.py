import json
import os
import sys
import re
from datetime import datetime


def sanitize_filename(filename):
    """清理文件名，替换不合法字符"""
    # 替换 # 和其他不合法字符为 _
    return re.sub(r'[#<>:"/\\|?*]', '_', filename)


def remove_ansi_codes(text):
    """去除ANSI转义序列（如颜色高亮控制字符）"""
    if not text:
        return text
    # 匹配ANSI转义序列，如 \x1b[31m, \u001b[0m 等
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m|\u001b\[[0-9;]*m')
    return ansi_pattern.sub('', text)


def timestamp_to_str(timestamp_ms):
    """将毫秒级时间戳转换为标准时间格式"""
    if timestamp_ms is None:
        return ''
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def format_log_content(result_json, result_dir, is_fail=True):
    """格式化用例日志内容
    
    Args:
        result_json: 用例结果JSON
        result_dir: 结果文件目录
        is_fail: 是否为失败用例，True为失败日志，False为成功日志
    
    Returns:
        格式化后的日志内容
    """
    lines = []
    
    # 1. 方法名
    lines.append(f"### 方法名\n{result_json.get('name', '')}\n")
    
    # 2. 方法描述
    lines.append(f"### 方法描述\n{result_json.get('description', '')}\n")
    
    # 方法时间
    start_time = result_json.get('start')
    stop_time = result_json.get('stop')
    if start_time and stop_time:
        lines.append(f"### 执行时间\n开始: {timestamp_to_str(start_time)}\n结束: {timestamp_to_str(stop_time)}\n耗时: {(stop_time - start_time) / 1000:.3f}s\n")
    
    # 3. 失败描述（仅失败用例）
    if is_fail:
        status_details = result_json.get('statusDetails', {})
        message = remove_ansi_codes(status_details.get('message', ''))
        trace = remove_ansi_codes(status_details.get('trace', ''))
        lines.append(f"### 失败描述\n**message:**\n```\n{message}\n```\n\n**trace:**\n```\n{trace}\n```\n")
    
    # 4. 每一个step的name & 对应attachments
    steps = result_json.get('steps', [])
    if steps:
        lines.append("### 步骤详情\n")
        for step in steps:
            step_name = step.get('name', '')
            step_start = step.get('start')
            step_stop = step.get('stop')
            time_info = ''
            if step_start and step_stop:
                time_info = f" ({timestamp_to_str(step_start)} ~ {timestamp_to_str(step_stop)})"
            lines.append(f"#### {step_name}{time_info}")
            attachments = step.get('attachments', [])
            if attachments:
                for attachment in attachments:
                    att_name = attachment.get('name', '')
                    att_source = attachment.get('source', '')
                    # 读取attachment文件内容
                    att_filepath = os.path.join(result_dir, att_source)
                    att_content = ''
                    if os.path.exists(att_filepath):
                        try:
                            with open(att_filepath, 'r', encoding='utf-8') as f:
                                att_content = f.read()
                        except Exception as e:
                            att_content = f"读取失败: {e}"
                    lines.append(f"- **{att_name}** (source: {att_source})")
                    lines.append(f"```\n{att_content}\n```\n")
            lines.append("")
    
    return '\n'.join(lines)


def format_fail_log(result_json, result_dir):
    """格式化失败用例日志"""
    return format_log_content(result_json, result_dir, is_fail=True)


def format_success_log(result_json, result_dir):
    """格式化成功用例日志"""
    return format_log_content(result_json, result_dir, is_fail=False)


def load_result_files(directory):
    account = 0
    fail_count = 0
    skip_count = 0
    success_count = 0
    fail_fullnames = []

    if not os.path.isdir(directory):
        print(f"路径不存在: {directory}")
        return {"用例总数": account, "失败用例数": fail_count, "忽略用例数": skip_count, "成功用例数": success_count, "失败用例列表": fail_fullnames, "readme": "如有失败，详情请见html_log & fail_markdown_directory & success_markdown_directory"}

    # 创建日志目录（与aci_use.py同级）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fail_log_dir = os.path.join(base_dir, 'fail_logs')
    success_log_dir = os.path.join(base_dir, 'success_logs')
    
    if not os.path.exists(fail_log_dir):
        os.makedirs(fail_log_dir)
    if not os.path.exists(success_log_dir):
        os.makedirs(success_log_dir)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("-result.json"):
                account += 1
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        status = data.get('status')
                        full_name = data.get('fullName', '')
                        
                        if status == 'skipped':
                            skip_count += 1
                        elif status == 'passed':
                            success_count += 1
                            # 生成格式化成功日志
                            log_content = format_success_log(data, root)
                            log_filename = sanitize_filename(full_name) + '.md'
                            log_filepath = os.path.join(success_log_dir, log_filename)
                            with open(log_filepath, 'a', encoding='utf-8') as log_f:
                                log_f.write(log_content)
                                log_f.write("\n---\n\n")
                        else:
                            fail_count += 1
                            fail_fullnames.append(full_name)
                            # 生成格式化失败日志
                            log_content = format_fail_log(data, root)
                            log_filename = sanitize_filename(full_name) + '.md'
                            log_filepath = os.path.join(fail_log_dir, log_filename)
                            with open(log_filepath, 'a', encoding='utf-8') as log_f:
                                log_f.write(log_content)
                                log_f.write("\n---\n\n")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"加载文件失败 {filepath}: {e}")

    return {"用例总数": account, "失败用例数": fail_count, "忽略用例数": skip_count, "成功用例数": success_count, "失败用例列表": fail_fullnames, "readme": "如有失败，详情请见html_log & fail_markdown_directory & success_markdown_directory"}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "load_result_files":
        allure_results = sys.argv[2]
        json_data = load_result_files(allure_results)
        json_data["html_log"] = sys.argv[3] if len(sys.argv) > 3 else "未配置"
        json_data["fail_markdown_directory"] = sys.argv[4] if len(sys.argv) > 4 else "未配置"
        json_data["success_markdown_directory"] = sys.argv[5] if len(sys.argv) > 5 else "未配置"
        print(json.dumps(json_data))
