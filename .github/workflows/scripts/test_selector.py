"""
Test Selector - 基于已有覆盖率数据的精准测试选择器（支持行、函数、文件粒度）

工作流程：
1. 从现有 coverage SQLite 数据构建"用例 → 覆盖代码行"映射
2. 解析代码变更（支持 GitHub PR 或本地文件哈希比对）
3. 选择受影响的测试用例（按行、函数、文件粒度匹配）
"""

import sqlite3
import os
import json
import hashlib
import re
import ast
import ssl
import subprocess
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional
import argparse

# ==================== 配置参数 ====================
# 仓库名称：用于过滤和路径标准化
REPO_NAME = 'vllm_ascend'

# 覆盖率密度阈值：变更行被覆盖的比例达到此值才推荐该测试
# 范围：0.0 ~ 1.0，值越大筛选越严格
# 例如：0.05 表示变更行中至少 5% 被该测试覆盖才推荐
# 建议：从 0.05 开始调，如果推荐太多则调高到 0.10、0.15、0.20...
COVERAGE_DENSITY_THRESHOLD = 0.0

# 最少受影响的行数阈值
MIN_AFFECTED_LINES = 1
# ==================== 配置参数 ====================


class CoverageSelector:
    """基于覆盖率数据的测试选择器"""
    
    def __init__(self, coverage_data_dir: str = None, source_dir: str = None):
        """
        Args:
            coverage_data_dir: 覆盖率数据目录（仅构建 map 时需要）
            source_dir: 源码目录（仅函数级匹配时需要）
        """
        self.coverage_data_dir = Path(coverage_data_dir) if coverage_data_dir else None
        self.source_dir = Path(source_dir) if source_dir else None
        self.test_case_map = {}  # test_case_name -> {files: {filepath: {lines}}}
        
    def scan_test_cases(self) -> List[str]:
        """扫描所有测试用例目录"""
        test_cases = []
        for item in self.coverage_data_dir.iterdir():
            if item.is_dir() and item.name.startswith('tests__') or item.name == 'cpu-ut':
                covdata = item / 'covdata'
                if covdata.exists():
                    test_cases.append(item.name)
        return sorted(test_cases)
    
    @staticmethod
    def normalize_test_name(test_name: str) -> str:
        """
        将测试用例目录名转换为标准脚本名称格式：
        - tests__e2e__... -> tests/e2e/...
        - tests__e2e__...--test_foo -> tests/e2e/...::test_foo
        - cpu-ut -> cpu-ut (保持不变)
        """
        if test_name == 'cpu-ut':
            return test_name
        # 先处理 -- 转为 ::
        result = test_name.replace('--', '::')
        # 再处理 __ 转为 /
        result = result.replace('__', '/')
        return result
    
    def get_covered_lines_from_file(self, cov_file: str, filename: str) -> Set[int]:
        """
        从单个 coverage SQLite 文件获取某文件被覆盖的行号
        """
        lines = set()
        try:
            conn = sqlite3.connect(cov_file)
            cursor = conn.cursor()
            
            # 查找文件ID（模糊匹配路径）
            cursor.execute(
                "SELECT id FROM file WHERE path LIKE ?", 
                (f'%{filename}',)
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return lines
            file_id = row[0]
            
            # 获取所有 arc，计算覆盖的行号
            cursor.execute(
                "SELECT DISTINCT fromno, tono FROM arc WHERE file_id = ?", 
                (file_id,)
            )
            for fromno, tono in cursor.fetchall():
                if fromno > 0:
                    lines.add(fromno)
                if tono > 0:
                    lines.add(tono)
            
            conn.close()
        except Exception as e:
            print(f"  Warning: Error reading {cov_file}: {e}")
        return lines
    
    def get_covered_files_from_file(self, cov_file: str) -> Set[str]:
        """从单个 coverage 文件获取所有被覆盖的文件"""
        files = set()
        try:
            conn = sqlite3.connect(cov_file)
            cursor = conn.cursor()
            cursor.execute("SELECT path FROM file")
            for (path,) in cursor.fetchall():
                if REPO_NAME in path:
                    rel_path = path.split(f'{REPO_NAME}/')[-1] if f'{REPO_NAME}/' in path else path
                    files.add(rel_path)
            conn.close()
        except Exception as e:
            print(f"  Warning: Error reading {cov_file}: {e}")
        return files
    
    def build_test_case_map(self) -> Dict:
        """构建测试用例 → 覆盖文件的映射（包含行号）"""
        print("正在扫描测试用例...")
        test_cases = self.scan_test_cases()
        print(f"  找到 {len(test_cases)} 个测试用例")
        
        for i, test_case in enumerate(test_cases):
            print(f"  [{i+1}/{len(test_cases)}] 处理 {test_case}...")
            covdata_dir = self.coverage_data_dir / test_case / 'covdata'
            
            file_lines_map = defaultdict(set)  # filepath -> set of lines
            
            for cov_file in covdata_dir.glob('coverage.*'):
                covered_files = self.get_covered_files_from_file(str(cov_file))
                
                for filename in covered_files:
                    lines = self.get_covered_lines_from_file(str(cov_file), filename)
                    if lines:
                        file_lines_map[filename].update(lines)
            
            normalized_name = self.normalize_test_name(test_case)
            self.test_case_map[normalized_name] = {
                "files": dict(file_lines_map),
                "file_count": len(file_lines_map),
                "line_count": sum(len(v) for v in file_lines_map.values())
            }
            
            print(f"    -> {len(file_lines_map)} 个文件, {sum(len(v) for v in file_lines_map.values())} 行")
        
        return self.test_case_map
    
    def save_map(self, output_path: str = "test_case_map.json"):
        """保存测试用例映射到文件"""
        serializable_map = {}
        for test_case, data in self.test_case_map.items():
            serializable_map[test_case] = {
                "files": {k: list(v) for k, v in data["files"].items()},
                "file_count": data["file_count"],
                "line_count": data["line_count"]
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_map, f, indent=2, ensure_ascii=False)
        print(f"\n测试用例映射已保存到: {output_path}")
    
    def load_map(self, input_path: str = "test_case_map.json"):
        """从文件加载测试用例映射"""
        with open(input_path, 'r', encoding='utf-8') as f:
            serializable_map = json.load(f)
        
        self.test_case_map = {}
        for test_case, data in serializable_map.items():
            self.test_case_map[test_case] = {
                "files": {k: set(v) for k, v in data["files"].items()},
                "file_count": data["file_count"],
                "line_count": data["line_count"]
            }
        print(f"已从 {input_path} 加载 {len(self.test_case_map)} 个测试用例映射")
        return self.test_case_map


class CodeChangeDetector:
    """代码变更检测器"""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.file_hashes = {}
    
    def compute_file_hash(self, filepath: str) -> str:
        """计算文件的 MD5 哈希"""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except:
            return ""
    
    def scan_source_files(self) -> Dict[str, str]:
        """扫描源码文件，计算哈希"""
        self.file_hashes = {}
        for py_file in self.source_dir.rglob('*.py'):
            rel_path = py_file.relative_to(self.source_dir).as_posix()
            self.file_hashes[rel_path] = self.compute_file_hash(str(py_file))
        return self.file_hashes
    
    def detect_changes_by_comparison(self) -> Dict[str, Set[int]]:
        """通过文件哈希比对检测变更（返回所有变更文件的全部行）"""
        changed_files = {}
        current_hashes = {}
        
        for py_file in self.source_dir.rglob('*.py'):
            rel_path = py_file.relative_to(self.source_dir).as_posix()
            current_hashes[rel_path] = self.compute_file_hash(str(py_file))
        
        baseline_path = self.source_dir / '.file_hashes.json'
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                old_hashes = json.load(f)
            
            for rel_path, current_hash in current_hashes.items():
                old_hash = old_hashes.get(rel_path, '')
                if current_hash != old_hash:
                    # 文件有变更，返回所有行号（保守估计）
                    changed_files[rel_path] = set(range(1, 10000))  # 保守：假设全部行都可能变更
        else:
            changed_files = {rel_path: set(range(1, 10000)) for rel_path in current_hashes}
            with open(baseline_path, 'w') as f:
                json.dump(current_hashes, f)
        
        return changed_files
    
    def parse_git_diff(self, diff_output: str, filter_prefix: Optional[str] = None) -> Dict[str, Set[int]]:
        """
        解析 git diff 输出，提取变更的行号
        
        支持两种 diff 格式：
        1. unified diff: @@ -10,3 +10,4 @@ context
        2. 来自 PR 的 diff
        
        Args:
            diff_output: diff 内容
            filter_prefix: 只保留以此前缀开头的文件路径（如 '{REPO_NAME}/' 过滤出产品代码，默认使用 REPO_NAME）
        
        Returns:
            {filepath: {lineno, ...}} - 新文件中的变更行号集合
        """
        changed_files = {}
        current_file = None
        
        # 默认使用 REPO_NAME 作为过滤前缀
        if filter_prefix is None:
            filter_prefix = f'{REPO_NAME}/'
        
        # 解析模式：逐行解析，精确计算每个变更行在新文件中的行号
        in_hunk = False
        hunk_context_lines = []  # [(old_line, new_line, is_added, is_deleted), ...]
        pending_old_start = 0  # 当前 hunk 的 old_start，用于提交时计算

        for raw_line in diff_output.split('\n'):
            line = raw_line.rstrip('\r')
            # 新文件开始
            if line.startswith('diff --git'):
                in_hunk = False
                continue
            
            # 文件路径
            elif line.startswith('+++ b/') or line.startswith('--- a/'):
                path = line[6:].strip()
                # 去掉 a/ 或 b/ 前缀
                if path.startswith('a/') or path.startswith('b/'):
                    path = path[2:]
                # 过滤：只保留指定前缀的路径（排除测试文件等）
                if filter_prefix and not path.startswith(filter_prefix):
                    current_file = None
                    continue
                # 标准化路径：去掉 filter_prefix 前缀
                if filter_prefix and path.startswith(filter_prefix):
                    path = path[len(filter_prefix):]
                if not path.endswith('.py'):
                    continue
                current_file = path
                if current_file not in changed_files:
                    changed_files[current_file] = set()
            
            # hunk 头：@@ -old_start,old_count +new_start,new_count @@
            elif line.startswith('@@') and current_file:
                # 解析: @@ -100,10 +100,12 @@
                match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    # 规则：起始行 = old_start + 2，结束行 = old_start + old_count - 3
                    start_line = old_start + 2
                    end_line = old_start + old_count - 3
                    if end_line <= start_line:
                        end_line = old_start + old_count
                    # 收集 hunk 内的所有行，检查是否有新增行（+ 开头）
                    hunk_lines = []
                    for hunk_line in diff_output.split('\n')[diff_output.split('\n').index(line) + 1:]:
                        if hunk_line.startswith('@@') or hunk_line.startswith('diff --git') or hunk_line.startswith('--- a/') or hunk_line.startswith('+++ b/'):
                            break
                        hunk_lines.append(hunk_line)
                    # 如果没有 + 开头的新增行，说明变更只有删除，区间向内缩一行
                    has_addition = any(l.lstrip().startswith('+') for l in hunk_lines)
                    if not has_addition:
                        start_line += 1
                        end_line -= 1
                    for line_no in range(start_line, end_line + 1):
                        changed_files[current_file].add(line_no)
        
        return changed_files

    def parse_pr_diff_file(self, diff_file_path: str) -> Dict[str, Set[int]]:
        """
        从 PR diff 文件解析变更行号
        
        Args:
            diff_file_path: diff 文件路径
        """
        try:
            with open(diff_file_path, 'r', encoding='utf-8') as f:
                diff_content = f.read()
            return self.parse_git_diff(diff_content)
        except Exception as e:
            print(f"Warning: 读取 diff 文件失败: {e}")
            return {}


class FunctionParser:
    """Python函数解析器 - 用于获取函数和分支的行号范围"""
    
    @staticmethod
    def get_function_ranges(filepath: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        解析Python文件，返回函数名 -> [(起始行, 结束行), ...] 的映射
        支持同一函数名出现多次的情况（返回所有匹配的区间）
        """
        function_ranges = defaultdict(list)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=filepath)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_ranges[node.name].append((node.lineno, node.end_lineno or node.lineno))
        except Exception as e:
            print(f"  Warning: 解析函数定义失败 {filepath}: {e}")
        
        return function_ranges
    
    @staticmethod
    def _get_import_lines(filepath: str) -> Set[int]:
        """
        获取文件中所有 import 语句的行号
        """
        import_lines = set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_lines.add(node.lineno)
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        import_lines.update(range(node.lineno, node.end_lineno + 1))
        except Exception:
            pass
        return import_lines
    
    @staticmethod
    def get_lines_functions(filepath: str, lines: Set[int],
                            skip_imports: bool = False) -> Dict[int, str]:
        """
        获取每行所属的函数名
        
        Args:
            filepath: 源文件路径
            lines: 待查询的行号集合
            skip_imports: 是否跳过 import 语句行
        """
        line_to_function = {}
        function_ranges = FunctionParser.get_function_ranges(filepath)
        
        func_to_covered_lines = {}
        for func_name, ranges in function_ranges.items():
            func_to_covered_lines[func_name] = set()
            for start, end in ranges:
                func_to_covered_lines[func_name].update(range(start, end + 1))
        
        for line in lines:
            for func_name, covered_lines in func_to_covered_lines.items():
                if line in covered_lines:
                    line_to_function[line] = func_name
                    break
        
        return line_to_function


class TestSelector:
    """测试选择器 - 根据代码变更（行粒度）选择需要运行的测试用例"""
    
    def __init__(self, test_case_map: Dict):
        self.test_case_map = test_case_map
    
    def select_tests(self, changed_files_with_lines: Dict[str, Set[int]], 
                     min_affected_lines: int = 1,
                     source_dir: Optional[str] = None,
                    enable_line_match: bool = True,
                    enable_function_match: bool = True,
                    enable_file_match: bool = True,
                    enable_skip_imports: bool = False,
                    enable_dedup: bool = False) -> Tuple[List[Tuple[str, Dict[str, Set[int]], int]], str]:
        """
        根据变更文件选择受影响的测试用例，支持3种独立匹配粒度：
        - 行级匹配：精确的变更行与覆盖行交集
        - 函数级匹配：整个函数体范围匹配
        - 文件级匹配：文件任意覆盖行匹配
        
        各粒度级联触发：只有当前粒度未匹配到测试时，才尝试下一粒度。
        
        Args:
            changed_files_with_lines: 变更文件及其行号 {filepath: {lineno, ...}}
            min_affected_lines: 最少受影响的行数，低于此值不选择
            source_dir: 源码目录，用于函数/文件级扩展
            enable_line_match: 是否启用行级匹配
            enable_function_match: 是否启用函数级匹配
            enable_file_match: 是否启用文件级匹配
            enable_skip_imports: 是否跳过 import 语句行（仅在函数级匹配时生效）
            enable_dedup: 是否启用去重
            
        Returns:
            (selected_tests, expand_reason)
            - selected_tests: [(test_case_name, {filepath: {covered_lines}}, total_affected_lines), ...]
            - expand_reason: 扩展原因说明 ('' 表示无扩展，'line'/'function'/'file' 表示使用的粒度)
        """
        selected = []
        expand_reason = ''
        
        # 标准化变更文件路径：去掉 REPO_NAME/ 前缀
        normalized_changed = {}
        for f, lines in changed_files_with_lines.items():
            if f.startswith(f'{REPO_NAME}/'):
                normalized_changed[f[len(f'{REPO_NAME}/'):]] = lines
            else:
                normalized_changed[f] = lines
        
        total_changed_lines = sum(len(lines) for lines in normalized_changed.values())
        
        # ===== 行级匹配 + 函数级匹配（并行执行，合并去重） =====
        line_results = []  # [(test_case, affected_detail, total_lines)]
        func_results = []  # [(test_case, affected_detail, total_lines)]
        
        # ----- 第一阶段：行级匹配 -----
        if enable_line_match:
            for test_case, data in self.test_case_map.items():
                covered_files = data["files"]  # {filepath: {lineno, ...}}
                
                # 行级匹配：计算哪些变更行被此测试覆盖
                affected_detail = {}  # {filepath: set of covered changed lines}
                all_intersected_lines = set()  # 所有文件交集的并集
                
                for changed_file, changed_lines in normalized_changed.items():
                    if changed_file in covered_files:
                        covered_lines = covered_files[changed_file]
                        # 计算变更行与覆盖行的交集
                        intersected_lines = changed_lines & covered_lines
                        if intersected_lines:
                            affected_detail[changed_file] = intersected_lines
                            all_intersected_lines.update(intersected_lines)
                
                # 计算总体覆盖率密度：交集行数 / 变更行总数
                overall_density = len(all_intersected_lines) / total_changed_lines if total_changed_lines else 0
                
                # 按覆盖率密度和最少受影响行数过滤
                if all_intersected_lines and overall_density >= COVERAGE_DENSITY_THRESHOLD and len(all_intersected_lines) >= min_affected_lines:
                    line_results.append((test_case, affected_detail, len(all_intersected_lines)))
            
            # 按受影响行数排序（多的优先）
            line_results.sort(key=lambda x: x[2], reverse=True)
            
            # 行级匹配去重：相同覆盖行只选一个测试
            if line_results and enable_dedup:
                claimed_lines = set()
                deduplicated = []
                for test_case, affected_detail, total_lines in line_results:
                    # 收集这个测试覆盖的所有行
                    test_lines = set()
                    for lines in affected_detail.values():
                        test_lines.update(lines)
                    # 只保留有新行的测试
                    unclaimed = test_lines - claimed_lines
                    if unclaimed:
                        deduplicated.append((test_case, affected_detail, len(unclaimed)))
                        claimed_lines.update(test_lines)
                line_results = deduplicated
        
        # ----- 第二阶段：函数级匹配 -----
        if enable_function_match and source_dir:
            # 收集所有变更行所属的函数
            changed_functions = {}  # {filepath: {func_name: Set[linenos]}}
            
            for changed_file, changed_lines in normalized_changed.items():
                possible_paths = [
                    Path(source_dir) / changed_file,
                    Path(source_dir) / REPO_NAME / changed_file,
                    Path(source_dir) / 'covstub' / REPO_NAME / changed_file,
                    Path(source_dir) / changed_file.replace('/', os.sep),
                    Path(source_dir) / REPO_NAME / changed_file.replace('/', os.sep),
                    Path(source_dir) / 'covstub' / REPO_NAME / changed_file.replace('/', os.sep),
                ]
                
                source_file = None
                for p in possible_paths:
                    if p.exists():
                        source_file = str(p)
                        break
                
                if not source_file:
                    continue
                
                # 获取变更行的函数映射
                line_to_function = FunctionParser.get_lines_functions(source_file, changed_lines,
                                                                       skip_imports=enable_skip_imports)
                
                # 按函数名分组
                func_to_lines = defaultdict(set)
                for line, func_name in line_to_function.items():
                    func_to_lines[func_name].add(line)
                
                if func_to_lines:
                    changed_functions[changed_file] = func_to_lines
            
            if changed_functions:
                # 构建函数 -> 覆盖该函数的测试映射
                func_to_tests = defaultdict(list)
                
                for test_case, data in self.test_case_map.items():
                    covered_files = data["files"]
                    
                    for changed_file, func_to_lines in changed_functions.items():
                        if changed_file not in covered_files:
                            continue
                        
                        covered_lines = covered_files[changed_file]
                        
                        for func_name in func_to_lines.keys():
                            # 获取该函数的完整行范围
                            possible_paths = [
                                Path(source_dir) / changed_file,
                                Path(source_dir) / REPO_NAME / changed_file,
                                Path(source_dir) / 'covstub' / REPO_NAME / changed_file,
                                Path(source_dir) / changed_file.replace('/', os.sep),
                                Path(source_dir) / REPO_NAME / changed_file.replace('/', os.sep),
                                Path(source_dir) / 'covstub' / REPO_NAME / changed_file.replace('/', os.sep),
                            ]
                            
                            source_file = None
                            for p in possible_paths:
                                if p.exists():
                                    source_file = str(p)
                                    break
                            
                            if not source_file:
                                continue
                            
                            # 过滤掉 import 语句行（用于显示）
                            if enable_skip_imports:
                                import_lines = FunctionParser._get_import_lines(source_file)
                                display_changed_lines = normalized_changed.get(changed_file, set()) - import_lines
                            else:
                                display_changed_lines = normalized_changed.get(changed_file, set())
                            
                            func_ranges = FunctionParser.get_function_ranges(source_file)
                            
                            if func_name not in func_ranges:
                                continue
                            
                            # 合并所有匹配到的函数范围
                            func_all_lines = set()
                            for func_start, func_end in func_ranges[func_name]:
                                func_all_lines.update(range(func_start, func_end + 1))
                            
                            if not func_all_lines:
                                continue
                            
                            # 看看这个测试是否覆盖了该函数的任何行
                            covered_in_func = covered_lines & func_all_lines
                            if covered_in_func:
                                # 取测试覆盖行与实际变更行的交集（用于显示）
                                covered_changed_lines = covered_lines & display_changed_lines
                                func_to_tests[func_name].append((test_case, covered_changed_lines))
                
                # 选择覆盖了变更函数其他行的测试（去重）
                for changed_file, func_to_lines in changed_functions.items():
                    for func_name in func_to_lines.keys():
                        if func_name in func_to_tests:
                            for test_case, covered_changed_lines in func_to_tests[func_name]:
                                existing = [s[0] for s in func_results]
                                if test_case not in existing and covered_changed_lines:
                                    # 显示实际的变更行覆盖，而非函数全量覆盖
                                    display_lines = covered_changed_lines if covered_changed_lines else set()
                                    func_results.append((test_case, {changed_file: display_lines}, len(display_lines)))
                
                func_results.sort(key=lambda x: x[2], reverse=True)
        
        # ===== 合并行级和函数级结果，去重 =====
        if line_results or func_results:
            # 按test_case去重，保留行级结果（更精确）
            seen = set()
            for test_case, affected_detail, total_lines in line_results:
                if test_case not in seen:
                    seen.add(test_case)
                    selected.append((test_case, affected_detail, total_lines))
            
            # 添加函数级独有的结果
            for test_case, affected_detail, total_lines in func_results:
                if test_case not in seen:
                    seen.add(test_case)
                    selected.append((test_case, affected_detail, total_lines))
            
            # 按受影响行数排序
            selected.sort(key=lambda x: x[2], reverse=True)
            
            if selected:
                return selected, 'line+function'
        
        # ===== 第三阶段：文件级匹配（仅当前两级都为空时） =====
            print("  行级匹配为空，尝试函数级匹配...")
            expand_reason = 'function'
            
            # 收集所有变更行所属的函数
            changed_functions = {}  # {filepath: {func_name: Set[linenos]}}
            
            for changed_file, changed_lines in normalized_changed.items():
                possible_paths = [
                    Path(source_dir) / changed_file,
                    Path(source_dir) / REPO_NAME / changed_file,
                    Path(source_dir) / 'covstub' / REPO_NAME / changed_file,
                    Path(source_dir) / changed_file.replace('/', os.sep),
                    Path(source_dir) / REPO_NAME / changed_file.replace('/', os.sep),
                    Path(source_dir) / 'covstub' / REPO_NAME / changed_file.replace('/', os.sep),
                ]
                
                source_file = None
                for p in possible_paths:
                    if p.exists():
                        source_file = str(p)
                        break
                
                if not source_file:
                    continue
                
                # 获取变更行的函数映射
                line_to_function = FunctionParser.get_lines_functions(source_file, changed_lines,
                                                                       skip_imports=enable_skip_imports)
                
                # 按函数名分组
                func_to_lines = defaultdict(set)
                for line, func_name in line_to_function.items():
                    func_to_lines[func_name].add(line)
                
                if func_to_lines:
                    changed_functions[changed_file] = func_to_lines
            
            if not changed_functions:
                return selected, expand_reason
            
            # 构建函数 -> 覆盖该函数的测试映射
            func_to_tests = defaultdict(list)
            
            for test_case, data in self.test_case_map.items():
                covered_files = data["files"]
                
                for changed_file, func_to_lines in changed_functions.items():
                    if changed_file not in covered_files:
                        continue
                    
                    covered_lines = covered_files[changed_file]
                    
                    for func_name in func_to_lines.keys():
                        # 获取该函数的完整行范围
                        possible_paths = [
                            Path(source_dir) / changed_file,
                            Path(source_dir) / REPO_NAME / changed_file,
                            Path(source_dir) / 'covstub' / REPO_NAME / changed_file,
                            Path(source_dir) / changed_file.replace('/', os.sep),
                            Path(source_dir) / REPO_NAME / changed_file.replace('/', os.sep),
                            Path(source_dir) / 'covstub' / REPO_NAME / changed_file.replace('/', os.sep),
                        ]
                        
                        source_file = None
                        for p in possible_paths:
                            if p.exists():
                                source_file = str(p)
                                break
                        
                        if not source_file:
                            continue
                        
                        # 过滤掉 import 语句行（用于显示）
                        if enable_skip_imports:
                            import_lines = FunctionParser._get_import_lines(source_file)
                            display_changed_lines = normalized_changed.get(changed_file, set()) - import_lines
                        else:
                            display_changed_lines = normalized_changed.get(changed_file, set())
                        
                        func_ranges = FunctionParser.get_function_ranges(source_file)
                        
                        if func_name not in func_ranges:
                            continue
                        
                        # 合并所有匹配到的函数范围
                        func_all_lines = set()
                        for func_start, func_end in func_ranges[func_name]:
                            func_all_lines.update(range(func_start, func_end + 1))
                        
                        if not func_all_lines:
                            continue
                        
                        # 看看这个测试是否覆盖了该函数的任何行
                        covered_in_func = covered_lines & func_all_lines
                        if covered_in_func:
                            # 取测试覆盖行与实际变更行的交集（用于显示）
                            covered_changed_lines = covered_lines & display_changed_lines
                            func_to_tests[func_name].append((test_case, covered_changed_lines))
            
            # 选择覆盖了变更函数其他行的测试（去重）
            for changed_file, func_to_lines in changed_functions.items():
                for func_name in func_to_lines.keys():
                    if func_name in func_to_tests:
                        for test_case, covered_changed_lines in func_to_tests[func_name]:
                            existing = [s[0] for s in selected]
                            if test_case not in existing and covered_changed_lines:
                                # 显示实际的变更行覆盖，而非函数全量覆盖
                                display_lines = covered_changed_lines if covered_changed_lines else set()
                                selected.append((test_case, {changed_file: display_lines}, len(display_lines)))
            
            selected.sort(key=lambda x: x[2], reverse=True)
            
            if selected:
                return selected, expand_reason
        
        # ===== 第四阶段：文件级匹配 =====
        if not selected and enable_file_match:
            print("  函数级匹配为空，尝试文件级匹配...")
            expand_reason = 'file'
            
            # 文件级匹配：任何覆盖了变更文件的测试都被选中
            for test_case, data in self.test_case_map.items():
                covered_files = data["files"]
                
                for changed_file in normalized_changed.keys():
                    if changed_file in covered_files:
                        covered_lines = covered_files[changed_file]
                        if covered_lines:
                            selected.append((test_case, {changed_file: covered_lines}, len(covered_lines)))
                            break  # 一个文件匹配就够了，不重复计数其他文件
            
            # 去重：同一用例只选一次
            if selected:
                seen = set()
                deduplicated = []
                for s in selected:
                    if s[0] not in seen:
                        seen.add(s[0])
                        deduplicated.append(s)
                selected = deduplicated
            
            selected.sort(key=lambda x: x[2], reverse=True)
        
        return selected, expand_reason
    
    def print_selection(self, selected: List[Tuple[str, Dict[str, Set[int]], int]], 
                        changed_files: Dict[str, Set[int]],
                        min_affected_lines: int = 1,
                        expand_reason: str = ''):
        """打印选择结果"""
        total_changed_lines = sum(len(v) for v in changed_files.values())
        
        print("\n" + "=" * 70)
        print(f"代码变更: {len(changed_files)} 个文件, {total_changed_lines} 行")
        
        # 显示扩展原因
        gran_names = {'line': '行级匹配', 'function': '函数级匹配', 'file': '文件级匹配', 'line+function': '行级+函数级匹配'}
        gran_detail_titles = {'line': '详细影响（行级匹配）', 'function': '详细影响（函数级匹配）', 'file': '详细影响（文件级匹配）', 'line+function': '详细影响（行级+函数级匹配）'}
        if expand_reason and expand_reason in gran_names:
            print(f"选择测试: {len(selected)} 个用例 ({gran_names[expand_reason]})")
        else:
            print(f"选择测试: {len(selected)} 个用例 (最少影响 {min_affected_lines} 行)")
        print("=" * 70)
        
        if not selected:
            print("\n没有测试用例覆盖变更的代码行！")
            print(f"变更详情: {self._format_changed_files(changed_files)}")
            return
        
        print(f"\n{'排名':<4} {'测试用例':<50} {'影响行数'}")
        print("-" * 70)
        
        for i, (test_case, affected_detail, total_lines) in enumerate(selected, 1):
            # 构建覆盖行号显示
            line_parts = []
            for filepath, lines in sorted(affected_detail.items()):
                line_parts.append(self._format_line_range(sorted(lines)))
            line_display = f" ({', '.join(line_parts)})" if line_parts else ""
            print(f"{i:<4} {test_case:<50} {total_lines}{line_display}")
        
        print(f"\n{gran_detail_titles.get(expand_reason, '详细影响')}:")
        for test_case, affected_detail, total_lines in selected[:10]:
            print(f"\n  {test_case} ({total_lines} 行):")
            for filepath, lines in sorted(changed_files.items()):
                line_str = self._format_line_range(sorted(lines))
                print(f"    - {filepath}: {line_str}")
    
    def _format_line_range(self, lines: List[int]) -> str:
        """将行号列表压缩为范围表示"""
        if not lines:
            return ""
        
        lines = sorted(set(lines))
        ranges = []
        start = lines[0]
        end = lines[0]
        
        for line in lines[1:]:
            if line == end + 1:
                end = line
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = line
        
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ", ".join(ranges)
    
    def _format_changed_files(self, changed_files: Dict[str, Set[int]]) -> str:
        """格式化变更文件"""
        result = []
        for f, lines in sorted(changed_files.items()):
            if len(lines) > 10:
                result.append(f"{f}: {len(lines)} 行")
            else:
                result.append(f"{f}: {sorted(lines)}")
        return ", ".join(result[:5]) + ("..." if len(changed_files) > 5 else "")


def main():
    parser = argparse.ArgumentParser(description='基于覆盖率数据的精准测试选择器（行、函数、文件粒度）')
    parser.add_argument('--github-pr', '-pr',
                        help='GitHub PR，格式: owner/repo#pr_number')
    parser.add_argument('--source-dir', '-s',
                        default='covstub',
                        help='源码目录 (默认: covstub)')
    parser.add_argument('--map-file', '-m',
                        default='test_case_map.json',
                        help='测试用例映射文件 (默认: test_case_map.json)')
    parser.add_argument('--coverage-dir', '-c',
                        default='coverage',
                        help='覆盖率数据目录 (默认: ./coverage)')
    parser.add_argument('--build-map', '-b',
                        action='store_true',
                        help='重新构建测试用例映射')
    parser.add_argument('--min-affected', '-a',
                        type=int, default=1,
                        help='最少受影响的行数阈值 (默认: 1)')
    parser.add_argument('--dedup',
                        action='store_true',
                        help='启用去重（相同覆盖行的测试只保留一个，默认关闭）')
    parser.add_argument('--enable-line-match',
                        action='store_true', default=False,
                        help='启用行级匹配（默认关闭）')
    parser.add_argument('--disable-line-match',
                        action='store_true',
                        help='禁用行级匹配')
    parser.add_argument('--enable-function-match',
                        action='store_true', default=True,
                        help='启用函数级匹配（默认开启）')
    parser.add_argument('--disable-function-match',
                        action='store_true',
                        help='禁用函数级匹配')
    parser.add_argument('--enable-file-match',
                        action='store_true', default=True,
                        help='启用文件级匹配（默认开启）')
    parser.add_argument('--disable-file-match',
                        action='store_true',
                        help='禁用文件级匹配')
    parser.add_argument('--skip-imports',
                        action='store_true',
                        help='跳过 import 语句行（仅在函数级匹配时生效，默认关闭）')
    
    args = parser.parse_args()
    
    # 处理粒度开关：disable 优先于 enable
    args.enable_line_match = not args.disable_line_match
    args.enable_function_match = not args.disable_function_match
    args.enable_file_match = not args.disable_file_match
    
    # 1. 构建或加载测试用例映射
    selector = CoverageSelector(args.coverage_dir, args.source_dir)
    
    if args.build_map or not Path(args.map_file).exists():
        print("\n=== 构建测试用例映射 ===")
        selector.build_test_case_map()
        selector.save_map(args.map_file)
    else:
        print("\n=== 加载测试用例映射 ===")
        selector.load_map(args.map_file)
    
    # 如果只需要生成 map 文件，则直接退出
    if args.build_map and not args.github_pr:
        print("\n=== 仅生成 map 文件，已完成 ===")
        return
    
    # 2. 解析代码变更
    print("\n=== 解析代码变更 ===")
    change_detector = CodeChangeDetector(args.source_dir)
    
    if args.github_pr:
        # 从 GitHub PR 获取变更
        pr_spec = args.github_pr
        repo = None
        pr_num = None
        
        # 解析 owner/repo#pr_number 格式
        if '#' in pr_spec:
            parts = pr_spec.split('#')
            repo = parts[0]
            pr_num = parts[1]
        else:
            pr_num = pr_spec
            # 尝试获取当前仓库
            try:
                result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    url = result.stdout.strip()
                    if 'github.com' in url:
                        match = re.search(r'github\.com[/:]([^/]+/[^/]+?)(?:\.git)?$', url)
                        if match:
                            repo = match.group(1)
            except:
                pass
        
        if not repo or not pr_num:
            print("Error: 无法解析 PR 信息，请使用 owner/repo#pr_number 格式")
            exit(1)
        
        print(f"从 GitHub PR 获取变更: {repo}#{pr_num}")
        
        # 创建不验证 SSL 证书的 context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # 使用跨平台临时目录
        diff_file = os.path.join(tempfile.gettempdir(), 'pr.diff')
        try:
            result = subprocess.run(['gh', 'pr', 'diff', str(pr_num), '-R', repo], 
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise FileNotFoundError()
            with open(diff_file, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            print("  使用 gh cli 获取 diff")
        except FileNotFoundError:
            # gh 不可用，尝试使用 urllib 通过 GitHub API
            print("  gh cli 不可用，尝试通过 GitHub API 获取...")
            try:
                pr_url = f"https://api.github.com/repos/{repo}/pulls/{pr_num}"
                req = urllib.request.Request(pr_url, headers={'Accept': 'application/vnd.github.v3+json'})
                with urllib.request.urlopen(req, timeout=30, context=ssl_context) as response:
                    pr_data = json.loads(response.read().decode())
                    diff_url = pr_data.get('diff_url')
                
                if not diff_url:
                    print("Error: 无法获取 diff URL")
                    exit(1)
                
                # 下载 diff (使用二进制模式避免行尾转换)
                req = urllib.request.Request(diff_url)
                with urllib.request.urlopen(req, timeout=60, context=ssl_context) as response:
                    diff_bytes = response.read()
                    with open(diff_file, 'wb') as f:
                        f.write(diff_bytes)
                print("  使用 GitHub API 获取 diff")
            except Exception as e:
                print(f"Error: 获取 PR diff 失败: {e}")
                exit(1)
        
        print(f"  PR diff 已保存到: {diff_file}")
        changed_files_with_lines = change_detector.parse_pr_diff_file(diff_file)
        print(f"解析到 {len(changed_files_with_lines)} 个变更文件")
    else:
        # 从文件比对获取（默认）
        change_detector.scan_source_files()
        changed_files_with_lines = change_detector.detect_changes_by_comparison()
        print(f"检测到 {len(changed_files_with_lines)} 个变更文件")
    
    # 3. 选择测试用例
    print("\n=== 选择受影响的测试用例 ===")
    test_selector = TestSelector(selector.test_case_map)
    selected, expand_reason = test_selector.select_tests(changed_files_with_lines, 
                                          min_affected_lines=args.min_affected,
                                          source_dir=args.source_dir,
                                          enable_line_match=args.enable_line_match,
                                          enable_function_match=args.enable_function_match,
                                          enable_file_match=args.enable_file_match,
                                          enable_skip_imports=args.skip_imports,
                                          enable_dedup=args.dedup)
    test_selector.print_selection(selected, changed_files_with_lines, 
                                  min_affected_lines=args.min_affected,
                                  expand_reason=expand_reason)
    
    # 4. 输出可执行的 pytest 命令
    if selected:
        print("\n=== 建议执行的测试用例 ===")
        test_names = [s[0] for s in selected]
        print(test_names)
        
        # 输出详细的 JSON 格式
        output = {
            "changed_files": {f: list(lines) for f, lines in changed_files_with_lines.items()},
            "selected_tests": [
                {
                    "name": s[0],
                    "affected_files": {f: list(lines) for f, lines in s[1].items()},
                    "total_affected_lines": s[2]
                }
                for s in selected
            ],
            "total_selected": len(test_names),
            "expand_reason": expand_reason
        }
        with open('recommended_pytest_paths.txt', 'w', encoding='utf-8') as f:
            for test_name in test_names:
                f.write(test_name + '\n')
        print(f"\n详细结果已保存到: recommended_pytest_paths.txt")


if __name__ == '__main__':
    main()
