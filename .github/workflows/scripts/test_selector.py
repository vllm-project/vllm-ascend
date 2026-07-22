"""
Test Selector - Precision test selector based on coverage data (line, function, file granularity)

Workflow:
1. Build 'test case -> covered lines' mapping from coverage SQLite data
2. Parse code changes (supports GitHub PR or local file hash comparison)
3. Select affected test cases (by line, function, file granularity)
"""

import argparse
import ast
import hashlib
import json
import os
import regex as re
import sqlite3
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional

# ==================== Configuration ====================
# Repository name: used for filtering and path normalization
REPO_NAME = 'vllm_ascend'

# Coverage density threshold: proportion of changed lines covered
# Range: 0.0 ~ 1.0, higher value = stricter filtering
# Example: 0.05 means at least 5% of changed lines must be covered
# Recommendation: start at 0.05, increase to 0.10/0.15/0.20 if too many results
COVERAGE_DENSITY_THRESHOLD = 0.0

# Minimum affected lines threshold
MIN_AFFECTED_LINES = 1


# ==================== Configuration ====================


class CoverageSelector:
    """Coverage-based test selector"""

    def __init__(self, coverage_data_dir: str = None, source_dir: str = None):
        """
        Args:
            coverage_data_dir: Coverage data directory (only needed for building map)
            source_dir: Source code directory (only needed for function-level matching)
        """
        self.coverage_data_dir = Path(coverage_data_dir) if coverage_data_dir else None
        self.source_dir = Path(source_dir) if source_dir else None
        self.test_case_map = {}  # test_case_name -> {files: {filepath: {lines}}}

    def scan_test_cases(self) -> List[str]:
        """Scan all test case directories"""
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
        Convert test case directory name to standard script name format:
        - tests__e2e__... -> tests/e2e/...
        - tests__e2e__...--test_foo -> tests/e2e/...::test_foo
        - cpu-ut -> cpu-ut (unchanged)
        """
        if test_name == 'cpu-ut':
            return test_name
        # First convert -- to ::
        result = test_name.replace('--', '::')
        # Then convert __ to /
        result = result.replace('__', '/')
        return result

    def get_covered_lines_from_file(self, cov_file: str, filename: str) -> Set[int]:
        """
        Get covered line numbers for a file from a single coverage SQLite file
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
        """Get all covered files from a single coverage file"""
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
        """Build test case -> covered files mapping (with line numbers)"""
        print("Scanning test cases...")
        test_cases = self.scan_test_cases()
        print(f"  Found {len(test_cases)} test cases")

        for i, test_case in enumerate(test_cases):
            print(f"  [{i + 1}/{len(test_cases)}] Processing {test_case}...")
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

            print(f"    -> {len(file_lines_map)} files, {sum(len(v) for v in file_lines_map.values())} lines")

        return self.test_case_map

    def save_map(self, output_path: str = "test_case_map.json"):
        """Save test case mapping to file"""
        serializable_map = {}
        for test_case, data in self.test_case_map.items():
            serializable_map[test_case] = {
                "files": {k: list(v) for k, v in data["files"].items()},
                "file_count": data["file_count"],
                "line_count": data["line_count"]
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_map, f, indent=2, ensure_ascii=False)
        print(f"\nTest case mapping saved to: {output_path}")

    def load_map(self, input_path: str = "test_case_map.json"):
        """Load test case mapping from file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            serializable_map = json.load(f)

        self.test_case_map = {}
        for test_case, data in serializable_map.items():
            self.test_case_map[test_case] = {
                "files": {k: set(v) for k, v in data["files"].items()},
                "file_count": data["file_count"],
                "line_count": data["line_count"]
            }
        print(f"Loaded {len(self.test_case_map)} test case mappings from {input_path}")
        return self.test_case_map


class CodeChangeDetector:
    """Code change detector"""

    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.file_hashes = {}

    def compute_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file"""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except Exception as e:
            print(f"  Warning: Error computing file hash: {filepath}: {e}")
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
        for raw_line in diff_output.split('\n'):
            line = raw_line.rstrip('\r')
            # 新文件开始
            if line.startswith('diff --git'):
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
                        if hunk_line.startswith('@@') or hunk_line.startswith('diff --git') or hunk_line.startswith(
                                '--- a/') or hunk_line.startswith('+++ b/'):
                            break
                        hunk_lines.append(hunk_line)
                    # 如果没有 + 开头的新增行，说明变更只有删除，区间向内缩一行
                    has_addition = any(hline.lstrip().startswith('+') for hline in hunk_lines)
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
            print(f"Warning: Failed to read diff file: {e}")
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
            print(f"  Warning: Failed to parse function definition {filepath}: {e}")

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
                if all_intersected_lines and overall_density >= COVERAGE_DENSITY_THRESHOLD and len(
                        all_intersected_lines) >= min_affected_lines:
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
                        
                        for func_name in func_to_lines:
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
                    for func_name in func_to_lines:
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
            print("  Line-level matching empty, trying function-level matching...")
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

                    for func_name in func_to_lines:
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
                for func_name in func_to_lines:
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
            print("  Function-level matching empty, trying file-level matching...")
            expand_reason = 'file'

            # 文件级匹配：任何覆盖了变更文件的测试都被选中
            for test_case, data in self.test_case_map.items():
                covered_files = data["files"]

                for changed_file in normalized_changed:
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
        print(f"Code changes: {len(changed_files)} files, {total_changed_lines} lines")

        # 显示扩展原因
        gran_names = {'line': 'Line match', 'function': 'Function match', 'file': 'File match',
                      'line+function': 'Line+Function match'}
        gran_detail_titles = {'line': 'Details (Line match)', 'function': 'Details (Function match)',
                              'file': 'Details (File match)', 'line+function': 'Details (Line+Function match)'}
        if expand_reason and expand_reason in gran_names:
            print(f"Selected: {len(selected)} test cases ({gran_names[expand_reason]})")
        else:
            print(f"Selected: {len(selected)} test cases (min affected: {min_affected_lines} lines)")
        print("=" * 70)

        if not selected:
            print("\nNo test cases cover the changed code lines!")
            print(f"Change details: {self._format_changed_files(changed_files)}")
            return

        print(f"\n{'#':<4} {'Test Case':<50} {'Affected Lines'}")
        print("-" * 70)

        for i, (test_case, affected_detail, total_lines) in enumerate(selected, 1):
            # Build coverage line display
            line_parts = []
            for filepath, lines in sorted(affected_detail.items()):
                line_parts.append(self._format_line_range(sorted(lines)))
            line_display = f" ({', '.join(line_parts)})" if line_parts else ""
            print(f"{i:<4} {test_case:<50} {total_lines}{line_display}")

        print(f"\n{gran_detail_titles.get(expand_reason, 'Details')}:")
        for test_case, affected_detail, total_lines in selected[:10]:
            print(f"\n  {test_case} ({total_lines} lines):")
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
    parser = argparse.ArgumentParser(
        description='Coverage-based precision test selector (line, function, file granularity)')
    parser.add_argument('--github-pr', '-pr',
                        help='GitHub PR, format: owner/repo#pr_number')
    parser.add_argument('--source-dir', '-s',
                        default='covstub',
                        help='Source code directory (default: covstub)')
    parser.add_argument('--map-file', '-m',
                        default='test_case_map.json',
                        help='Test case map file (default: test_case_map.json)')
    parser.add_argument('--coverage-dir', '-c',
                        default='coverage',
                        help='Coverage data directory (default: ./coverage)')
    parser.add_argument('--build-map', '-b',
                        action='store_true',
                        help='Rebuild test case mapping')
    parser.add_argument('--min-affected', '-a',
                        type=int, default=1,
                        help='Minimum affected lines threshold (default: 1)')
    parser.add_argument('--dedup',
                        action='store_true',
                        help='Enable deduplication (keep only one test for same covered lines, default off)')
    parser.add_argument('--enable-line-match',
                        action='store_true', default=False,
                        help='Enable line-level matching (default off)')
    parser.add_argument('--disable-line-match',
                        action='store_true',
                        help='Disable line-level matching')
    parser.add_argument('--enable-function-match',
                        action='store_true', default=True,
                        help='Enable function-level matching (default on)')
    parser.add_argument('--disable-function-match',
                        action='store_true',
                        help='Disable function-level matching')
    parser.add_argument('--enable-file-match',
                        action='store_true', default=True,
                        help='Enable file-level matching (default on)')
    parser.add_argument('--disable-file-match',
                        action='store_true',
                        help='Disable file-level matching')
    parser.add_argument('--skip-imports',
                        action='store_true',
                        help='Skip import statement lines (only effective for function-level matching, default off)')

    args = parser.parse_args()

    # 处理粒度开关：disable 优先于 enable
    args.enable_line_match = not args.disable_line_match
    args.enable_function_match = not args.disable_function_match
    args.enable_file_match = not args.disable_file_match

    # 1. Build or load test case mapping
    selector = CoverageSelector(args.coverage_dir, args.source_dir)

    if args.build_map or not Path(args.map_file).exists():
        print("\n=== Building Test Case Mapping ===")
        selector.build_test_case_map()
        selector.save_map(args.map_file)
    else:
        print("\n=== Loading Test Case Mapping ===")
        selector.load_map(args.map_file)

    # If only need to generate map file, exit directly
    if args.build_map and not args.github_pr:
        print("\n=== Map file generated, done ===")
        return

    # 2. Parse code changes
    print("\n=== Parsing Code Changes ===")
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
            except Exception as e:
                print(e)
                pass

        if not repo or not pr_num:
            print("Error: Cannot parse PR info, please use owner/repo#pr_number format")
            exit(1)

        print(f"Fetching changes from GitHub PR: {repo}#{pr_num}")

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
            print("  Using gh cli to get diff")
        except FileNotFoundError:
            # gh not available, try urllib via GitHub API
            print("  gh cli not available, trying via GitHub API...")
            try:
                pr_url = f"https://api.github.com/repos/{repo}/pulls/{pr_num}"
                req = urllib.request.Request(pr_url, headers={'Accept': 'application/vnd.github.v3+json'})
                with urllib.request.urlopen(req, timeout=30, context=ssl_context) as response:
                    pr_data = json.loads(response.read().decode())
                    diff_url = pr_data.get('diff_url')

                if not diff_url:
                    print("Error: Cannot get diff URL")
                    exit(1)

                # Download diff (use binary mode to avoid line ending conversion)
                req = urllib.request.Request(diff_url)
                with urllib.request.urlopen(req, timeout=60, context=ssl_context) as response:
                    diff_bytes = response.read()
                    with open(diff_file, 'wb') as f:
                        f.write(diff_bytes)
                print("  Using GitHub API to get diff")
            except Exception as e:
                print(f"Error: Failed to get PR diff: {e}")
                exit(1)

        print(f"  PR diff saved to: {diff_file}")
        changed_files_with_lines = change_detector.parse_pr_diff_file(diff_file)
        print(f"Parsed {len(changed_files_with_lines)} changed files")
    else:
        # Get from file comparison (default)
        change_detector.scan_source_files()
        changed_files_with_lines = change_detector.detect_changes_by_comparison()
        print(f"Detected {len(changed_files_with_lines)} changed files")

    # 3. Select test cases
    print("\n=== Selecting Affected Test Cases ===")
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

    # 4. Output executable pytest command
    test_names = [s[0] for s in selected]
    if test_names:
        print("\n=== Recommended Test Cases ===")
        print(test_names)
    else:
        print("\n=== No Test Cases Recommended ===")

    # Always write output file (even if empty)
    with open('recommended_pytest_paths.txt', 'w', encoding='utf-8') as f:
        for test_name in test_names:
            f.write(test_name + '\n')
    print(f"\nResults saved to: recommended_pytest_paths.txt")


if __name__ == '__main__':
    main()
