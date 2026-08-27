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
数值边界静态分析工具 — 区间传播法

传入表达式和操作数的类型+值域，用区间运算验证安全性。
每条反例都在边界值上，不会出现无代表性的中间值。

用法:
  python3 clause.check_bounds.py \\
    --expr "totalOutputSize - aivIdx * singleCoreSize" \\
    --vars "aivIdx=uint32_t:0:47" "singleCoreSize=uint32_t:3:3" "totalOutputSize=int64_t:10:1000000" \\
    --check wraparound

  python3 clause.check_bounds.py \\
    --expr "batchSize * blockLength * sizeof(T)" \\
    --vars "batchSize=int32_t:1:128" "blockLength=int32_t:32:1024" \\
    --check overflow

var 格式: name=type:min:max
  name: 表达式中的变量名 (支持 func() / a->b / a.b 等 C++ 写法)
  type: uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, size_t
  min/max: 整数（含两端），表示该变量的值域范围
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# ─── Type table ──────────────────────────────────────────────────

TYPE_TABLE = {
    "int8_t": (8, True),
    "uint8_t": (8, False),
    "int16_t": (16, True),
    "uint16_t": (16, False),
    "int32_t": (32, True),
    "uint32_t": (32, False),
    "int64_t": (64, True),
    "uint64_t": (64, False),
    "size_t": (64, False),
}

# ─── Tokenizer ───────────────────────────────────────────────────


class TokenKind(Enum):
    NUM = 1
    ID = 2
    LPAREN = 3
    RPAREN = 4
    PLUS = 5
    MINUS = 6
    STAR = 7
    SLASH = 8
    PERCENT = 9
    END = 99


OP_PREC = {TokenKind.STAR: 3, TokenKind.SLASH: 3, TokenKind.PERCENT: 3,
           TokenKind.PLUS: 2, TokenKind.MINUS: 2}


class Token:
    def __init__(self, kind: TokenKind, value=None):
        self.kind = kind
        self.value = value


_SINGLE_CHAR_TOKENS = {
    '(': TokenKind.LPAREN, ')': TokenKind.RPAREN,
    '+': TokenKind.PLUS, '-': TokenKind.MINUS,
    '*': TokenKind.STAR, '/': TokenKind.SLASH, '%': TokenKind.PERCENT,
}


def _raw_scan(expr: str) -> List[Token]:
    """Scan expression string into raw tokens."""
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(Token(TokenKind.NUM, int(expr[i:j])))
            i = j
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            tokens.append(Token(TokenKind.ID, expr[i:j]))
            i = j
            continue
        if c == '-' and i + 1 < len(expr) and expr[i + 1] == '>':
            tokens.append(Token(TokenKind.ID, '->'))
            i += 2
            continue
        if c in _SINGLE_CHAR_TOKENS:
            tokens.append(Token(_SINGLE_CHAR_TOKENS[c]))
            i += 1
            continue
        if c == '.':
            tokens.append(Token(TokenKind.ID, '.'))
            i += 1
            continue
        raise ValueError(f"非预期的字符 {repr(c)} (位置 {i})")
    tokens.append(Token(TokenKind.END))
    return tokens


def _is_member_access(t: Token, merged: List[Token], tokens: List[Token], i: int) -> bool:
    """Check if t is '.' or '->' connecting two IDs."""
    return (t.kind == TokenKind.ID and t.value in ('.', '->') and merged and
            merged[-1].kind == TokenKind.ID and i + 1 < len(tokens) and
            tokens[i + 1].kind == TokenKind.ID)


def _merge_function_call(tokens: List[Token], i: int) -> Tuple[Token, int]:
    """Merge fn(...) into a single ID token. Returns (merged_token, next_i)."""
    depth = 1
    end = i + 2
    while end < len(tokens) and depth > 0:
        if tokens[end].kind == TokenKind.LPAREN:
            depth += 1
        elif tokens[end].kind == TokenKind.RPAREN:
            depth -= 1
        end += 1
    full = ''.join(tok.value if tok.value else
                   ('(' if tok.kind == TokenKind.LPAREN else ')')
                   for tok in tokens[i:end])
    return Token(TokenKind.ID, full), end


def _merge_compounds(tokens: List[Token]) -> List[Token]:
    """Merge adjacent tokens into C++ compound identifiers (func(), a->b, a.b)."""
    merged = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        # fn(...) → single ID
        if t.kind == TokenKind.ID and i + 1 < len(tokens) and tokens[i + 1].kind == TokenKind.LPAREN:
            token, end = _merge_function_call(tokens, i)
            merged.append(token)
            i = end
            continue
        # a.b or a->b → single ID
        if _is_member_access(t, merged, tokens, i):
            left = merged.pop()
            member = tokens[i + 1]
            merged.append(Token(TokenKind.ID, left.value + t.value + member.value))
            i += 2
            continue
        merged.append(t)
        i += 1
    return merged


def tokenize(expr: str) -> List[Token]:
    return _merge_compounds(_raw_scan(expr))


# ─── AST ─────────────────────────────────────────────────────────

class ExprNode:
    def __init__(self, kind: str, value=None, left=None, right=None):
        self.kind = kind        # 'num' | 'var' | 'binop'
        self.value = value      # int | str
        self.left = left
        self.right = right
        self.op: 'Optional[TokenKind]' = None


@dataclass
class SafetyContext:
    """Safety analysis context shared across _safe_at / _find_threshold."""
    ast: ExprNode
    var_info: Dict
    target: str
    fixed_vals: Dict[str, int]
    used_vars: Set[str]
    check_type: str
    want_min_expr: bool


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def eat(self, expected: Optional[TokenKind] = None) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        if expected is not None and t.kind != expected:
            raise ValueError(f"期望 {expected}, 遇到 {t.kind}")
        return t

    def parse(self, min_prec: int = 0) -> ExprNode:
        t = self.peek()
        if t.kind == TokenKind.LPAREN:
            self.eat(TokenKind.LPAREN)
            node = self.parse(0)
            self.eat(TokenKind.RPAREN)
        elif t.kind == TokenKind.NUM:
            self.eat()
            node = ExprNode('num', t.value)
        elif t.kind == TokenKind.ID:
            self.eat()
            node = ExprNode('var', t.value)
        elif t.kind == TokenKind.MINUS:
            self.eat()
            node = ExprNode('binop', None, ExprNode('num', 0),
                            self.parse(OP_PREC[TokenKind.STAR]))
            node.op = TokenKind.MINUS
        else:
            raise ValueError(f"非预期的 token: {t.kind}")

        while True:
            t = self.peek()
            if t.kind not in OP_PREC:
                break
            prec = OP_PREC[t.kind]
            if prec < min_prec:
                break
            op = self.eat().kind
            rhs = self.parse(prec + 1)
            node = ExprNode('binop', None, node, rhs)
            node.op = op
        return node


# ─── Variable parsing ────────────────────────────────────────────

VarInfo = Tuple[int, bool, int, int]  # (bit_width, signed, lo, hi)


def parse_variables(specs: List[str]) -> Dict[str, VarInfo]:
    result = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f"变量格式错误: {spec!r}, 期望 name=type:min:max")
        name, rest = spec.split('=', 1)
        parts = rest.split(':')
        if len(parts) < 3:
            raise ValueError(f"变量格式错误: {spec!r}, 期望 name=type:min:max")
        typ, lo_str, hi_str = parts[0], parts[1], parts[2]
        lo, hi = int(lo_str), int(hi_str)
        if typ not in TYPE_TABLE:
            raise ValueError(f"未知类型: {typ!r}, 支持: {list(TYPE_TABLE)}")
        bw, signed = TYPE_TABLE[typ]
        result[name] = (bw, signed, lo, hi)
    return result


def collect_vars(node: ExprNode) -> Set[str]:
    if node.kind == 'var':
        return {node.value}
    if node.kind == 'binop':
        return collect_vars(node.left) | collect_vars(node.right)
    return set()


# ─── Arithmetic helpers ──────────────────────────────────────────

def _apply_op(op: TokenKind, l: int, r: int) -> Optional[int]:
    if op == TokenKind.PLUS:
        return l + r
    if op == TokenKind.MINUS:
        return l - r
    if op == TokenKind.STAR:
        return l * r
    if op == TokenKind.SLASH:
        return l // r if r != 0 else None
    if op == TokenKind.PERCENT:
        return l % r if r != 0 else None
    return None


def _div_mod_vals(l_lo, l_hi, r_lo, r_hi, is_mod):
    """All possible results of a/b or a%b over the interval corners to capture extremes."""
    vals = set()
    for l in (l_lo, l_hi):
        for r in (r_lo, r_hi):
            if r != 0:
                vals.add(l % r if is_mod else l // r)
    for r in (1, -1):
        if r_lo <= r <= r_hi:
            for l in (l_lo, l_hi):
                vals.add(l % r if is_mod else l // r)
    if not vals:
        return (0, 0)
    return (min(vals), max(vals))


# ─── Interval propagation ────────────────────────────────────────

def compute_interval(node: ExprNode, var_info: Dict[str, VarInfo]
                     ) -> Tuple[int, int]:
    """Compute [min, max] of the expression using interval arithmetic.

    All arithmetic is done in Python's arbitrary-precision integers,
    giving the true mathematical result before C truncation.
    """
    if node.kind == 'num':
        return (node.value, node.value)
    if node.kind == 'var':
        _, _, lo, hi = var_info[node.value]
        return (lo, hi)

    l_lo, l_hi = compute_interval(node.left, var_info)
    r_lo, r_hi = compute_interval(node.right, var_info)
    op = node.op

    if op == TokenKind.PLUS:
        return (l_lo + r_lo, l_hi + r_hi)
    if op == TokenKind.MINUS:
        return (l_lo - r_hi, l_hi - r_lo)
    if op == TokenKind.STAR:
        p = [l_lo * r_lo, l_lo * r_hi, l_hi * r_lo, l_hi * r_hi]
        return (min(p), max(p))
    if op == TokenKind.SLASH:
        return _div_mod_vals(l_lo, l_hi, r_lo, r_hi, False)
    if op == TokenKind.PERCENT:
        return _div_mod_vals(l_lo, l_hi, r_lo, r_hi, True)
    return (0, 0)


# ─── Counter-example generation ──────────────────────────────────

def _div_extra_candidates(op: TokenKind, l_lo: int, l_hi: int,
                          r_lo: int, r_hi: int) -> List[Tuple[int, int, bool, bool]]:
    """Build extra corner candidates for division/modulo (r=±1)."""
    if op not in (TokenKind.SLASH, TokenKind.PERCENT):
        return []
    extra = []
    for r_val in (1, -1):
        if r_lo <= r_val <= r_hi:
            for l_min in (True, False):
                l_val = l_lo if l_min else l_hi
                extra.append((l_val, r_val, l_min, l_min))
    return extra


def _build_corner_candidates(op: TokenKind, l_lo: int, l_hi: int,
                             r_lo: int, r_hi: int) -> List[Tuple[int, int, bool, bool]]:
    """Build corner (endpoint) combinations for a binary op."""
    candidates = []
    for l_min in (True, False):
        l_val = l_lo if l_min else l_hi
        for r_min in (True, False):
            r_val = r_lo if r_min else r_hi
            candidates.append((l_val, r_val, l_min, r_min))
    candidates.extend(_div_extra_candidates(op, l_lo, l_hi, r_lo, r_hi))
    return candidates


def _pick_binop_corners(node: ExprNode, var_info: Dict[str, VarInfo],
                        want_min: bool) -> Dict[str, int]:
    """Enumerate corner (endpoint) combinations for a binary op to find
    variable assignments that achieve the target extreme value.

    For division/modulo, also checks r=±1 when the divisor range crosses
    them, since the extreme may occur at those boundaries.
    """
    op = node.op
    l_lo, l_hi = compute_interval(node.left, var_info)
    r_lo, r_hi = compute_interval(node.right, var_info)

    candidates = _build_corner_candidates(op, l_lo, l_hi, r_lo, r_hi)

    best_val = None
    best_choices = None
    for l_val, r_val, l_min, r_min in candidates:
        result = _apply_op(op, l_val, r_val)
        if result is None:
            continue
        better = (best_val is None or
                  (want_min and result < best_val) or
                  (not want_min and result > best_val))
        if better:
            best_val = result
            best_choices = {
                **pick_values(node.left, var_info, l_min),
                **pick_values(node.right, var_info, r_min),
            }
    return best_choices or {}


def pick_values(node: ExprNode, var_info: Dict[str, VarInfo],
                want_min: bool) -> Dict[str, int]:
    """Pick variable values at interval endpoints to achieve min or max value."""
    if node.kind == 'num':
        return {}
    if node.kind == 'var':
        _, _, lo, hi = var_info[node.value]
        return {node.value: lo if want_min else hi}
    return _pick_binop_corners(node, var_info, want_min)


# ─── Type bounds ─────────────────────────────────────────────────

def _type_bounds(bw: int, signed: bool) -> Tuple[int, int]:
    """Return (min, max) for a C integer type of given bit width."""
    if signed:
        return (-(1 << (bw - 1)), (1 << (bw - 1)) - 1)
    return (0, (1 << bw) - 1)


# ─── Violation detection ─────────────────────────────────────────

def _detect_violation(result_lo: int, result_hi: int,
                      var_info: Dict[str, VarInfo],
                      used_vars: Set[str], check_type: str
                      ) -> Optional[Tuple[str, int, bool]]:
    """Check if the result interval violates the safety condition.

    Returns (description, violating_value, want_min) or None if safe.
    want_min=True means the minimum value violates; False means maximum violates.
    """
    result_bw = max(var_info[v][0] for v in used_vars)

    if check_type == 'divzero':
        return None  # handled separately by scanning divisor intervals

    if check_type == 'wraparound':
        # Always check against unsigned range — wraparound is about
        # unsigned modulo semantics regardless of operand signedness.
        umax = (1 << result_bw) - 1
        if result_hi > umax:
            return (f"结果上界 {result_hi} 超出最大位宽 {result_bw}-bit 无符号上限 {umax}",
                    result_hi, False)
        if result_lo < 0:
            return (f"结果下界 {result_lo} < 0，按无符号运算将发生回绕",
                    result_lo, True)

    if check_type == 'overflow':
        # Always check against signed range — overflow is about
        # two's complement signed limits regardless of operand unsignedness.
        smax = (1 << (result_bw - 1)) - 1
        smin = -(1 << (result_bw - 1))
        if result_hi > smax:
            return (f"结果上界 {result_hi} 超出最大位宽 {result_bw}-bit 有符号上限 {smax}",
                    result_hi, False)
        if result_lo < smin:
            return (f"结果下界 {result_lo} 低于最大位宽 {result_bw}-bit 有符号下限 {smin}",
                    result_lo, True)

    return None


def _check_divzero(node: ExprNode, var_info: Dict[str, VarInfo]
                   ) -> Optional[Dict[str, int]]:
    """Check if any divisor can be zero. Returns counter-example or None."""
    if node.kind in ('num', 'var'):
        return None
    if node.kind == 'binop':
        if node.op in (TokenKind.SLASH, TokenKind.PERCENT):
            r_lo, r_hi = compute_interval(node.right, var_info)
            if r_lo <= 0 <= r_hi:
                # Pick values that make divisor zero
                return pick_values(node.right, var_info, True)
        # Recurse into sub-expressions
        left_result = _check_divzero(node.left, var_info)
        if left_result:
            return left_result
        return _check_divzero(node.right, var_info)
    return None


# ─── Sensitivity analysis ────────────────────────────────────────

def _eval_fixed(ast: ExprNode, var_info: Dict[str, VarInfo],
                target_var: str, fixed_vals: Dict[str, int],
                target_val: int) -> Optional[int]:
    """Evaluate expression with one variable free, others fixed."""
    def walk(node):
        if node.kind == 'num':
            return node.value
        if node.kind == 'var':
            return target_val if node.value == target_var else fixed_vals[node.value]
        l_val = walk(node.left)
        r_val = walk(node.right)
        if l_val is None or r_val is None:
            return None
        return _apply_op(node.op, l_val, r_val)
    return walk(ast)


def _var_relation(ast: ExprNode, var_info: Dict[str, VarInfo],
                  target: str, fixed_vals: Dict[str, int]) -> bool:
    """Return True if increasing *target* value increases the expression result."""
    _, _, lo, hi = var_info[target]
    v_lo = _eval_fixed(ast, var_info, target, fixed_vals, lo)
    v_hi = _eval_fixed(ast, var_info, target, fixed_vals, hi)
    if v_lo is None or v_hi is None:
        return True
    return v_hi >= v_lo


def _safe_at(val: int, ctx: SafetyContext) -> bool:
    """Check if expression is safe when target=val and others fixed."""
    result = _eval_fixed(ctx.ast, ctx.var_info, ctx.target, ctx.fixed_vals, val)
    if result is None:
        return False
    if ctx.check_type == 'wraparound':
        return result >= 0
    result_bw = max(ctx.var_info[v][0] for v in ctx.used_vars)
    hi_bound = (1 << (result_bw - 1)) - 1
    lo_bound = -(1 << (result_bw - 1))
    return result >= lo_bound if ctx.want_min_expr else result <= hi_bound


def _find_threshold(ctx: SafetyContext) -> Tuple[int, bool]:
    """Binary search for target's safety threshold.
    Returns (threshold, is_lower_bound)."""
    _, _, lo, hi = ctx.var_info[ctx.target]
    direct = _var_relation(ctx.ast, ctx.var_info, ctx.target, ctx.fixed_vals)
    want_min_for_var = ctx.want_min_expr if direct else not ctx.want_min_expr

    def safe(v):
        return _safe_at(v, ctx)

    if want_min_for_var:
        s_lo, s_hi = 0, hi
        if safe(s_lo):
            return s_lo, True
        if not safe(s_hi):
            return s_hi, True
    else:
        result_bw = max(ctx.var_info[v][0] for v in ctx.used_vars)
        s_lo, s_hi = lo, max(hi * 10, (1 << min(result_bw, 63)) - 1)
        if not safe(s_lo):
            return s_lo, False
        if safe(s_hi):
            return s_hi, False

    left, right = s_lo, s_hi
    while left < right:
        if want_min_for_var:
            mid = (left + right) // 2
            if safe(mid):
                right = mid
            else:
                left = mid + 1
        else:
            mid = (left + right + 1) // 2
            if safe(mid):
                left = mid
            else:
                right = mid - 1
    return left, want_min_for_var


def _compute_sensitivity(ast: ExprNode, var_info: Dict[str, VarInfo],
                         used_vars: Set[str], check_type: str,
                         want_min: bool) -> Dict[str, Tuple[int, int, int, bool]]:
    """Compute safety threshold for each variable.

    Returns {var_name: (lo, hi, threshold, is_lower_bound)}.
    """
    worst = pick_values(ast, var_info, want_min)
    result = {}
    for var in used_vars:
        fixed = {k: v for k, v in worst.items() if k != var}
        ctx = SafetyContext(ast, var_info, var, fixed, used_vars, check_type, want_min)
        threshold, is_lb = _find_threshold(ctx)
        _, _, lo, hi = var_info[var]
        result[var] = (lo, hi, threshold, is_lb)
    return result


def _format_lb_sensitivity(name: str, lo: int, thresh: int, is_safe: bool) -> str:
    """Format sensitivity line for lower-bound threshold."""
    if thresh <= lo:
        margin = lo - thresh
        label = f"余量={margin}" if is_safe else f"安全需 ≥ {thresh}"
        return f"  {name} 下限={lo}: 临界值={thresh}, {label}"
    if is_safe:
        return f"  {name} 下限={lo}: 临界值={thresh} (在范围内)"
    return f"  {name} 下限={lo}: 安全需 ≥ {thresh}, 缺口={thresh - lo}"


def _format_ub_sensitivity(name: str, hi: int, thresh: int, is_safe: bool) -> str:
    """Format sensitivity line for upper-bound threshold."""
    if thresh >= hi:
        margin = thresh - hi
        label = f"余量={margin}" if is_safe else f"安全需 ≤ {thresh}"
        return f"  {name} 上限={hi}: 临界值={thresh}, {label}"
    if is_safe:
        return f"  {name} 上限={hi}: 临界值={thresh} (在范围内)"
    return f"  {name} 上限={hi}: 安全需 ≤ {thresh}, 超出={hi - thresh}"


def _print_sensitivity(sens, var_info: Dict[str, VarInfo], is_safe: bool):
    """Print sensitivity analysis — which boundaries matter most."""
    if is_safe:
        logger.info("边界敏感性分析 (放宽哪些边界会打破安全):")
    else:
        logger.info("边界敏感性分析 (调整哪些边界可恢复安全):")

    for name, (lo, hi, thresh, is_lb) in sens.items():
        if is_lb:
            logger.info(_format_lb_sensitivity(name, lo, thresh, is_safe))
        else:
            logger.info(_format_ub_sensitivity(name, hi, thresh, is_safe))
    logger.info("")


def _print_guidance(is_safe: bool, sens, var_info: Dict[str, VarInfo]):
    """Print step-by-step action guidance after sensitivity analysis."""
    logger.info("💡 行动指引:")
    if is_safe:
        # Find the variable with smallest safety margin
        closest_var, closest_margin = None, float('inf')
        for name, (lo, hi, thresh, is_lb) in sens.items():
            margin = lo - thresh if is_lb else thresh - hi
            if margin < closest_margin:
                closest_var, closest_margin = name, margin
        if closest_var:
            logger.info(f"  最敏感变量: {closest_var}, 安全余量={closest_margin}")
        logger.info("  → 步骤1: 每个边界值能否追溯到代码行 (守卫/constexpr/赋值)？")
        logger.info("  → 步骤2: 无证据 → 向不利方向放宽该边界重跑")
        logger.info("  → 步骤3: 仍 SAFE → PASS。变 FAIL → 该边界是关键，须找代码证据")
    else:
        logger.info("  → 步骤1: 反例中「触及上限/下限」的变量，其边界来自代码证据还是推测？")
        logger.info("  → 步骤2: 来自 constexpr/守卫 → 边界可靠，确认 FAIL")
        logger.info("  → 步骤3: 来自推测 → Grep 找该变量的真实限定值，修正边界重跑")
        logger.info("  → 步骤4: 重跑仍 FAIL → 确认风险。变 PASS → 边界需代码证据支撑")
        logger.info("  → 步骤5: 找不到真实限定 → 输出 SUSPICIOUS + 标注边界不确定")
    logger.info("")


# ─── Output ──────────────────────────────────────────────────────

def _check_type_label(check_type: str) -> str:
    return {'overflow': '溢出', 'wraparound': '回绕', 'divzero': '除零'}[check_type]


def print_summary(expr_str: str, check_type: str, var_info: Dict[str, VarInfo]):
    logger.info(f"表达式: {expr_str}")
    logger.info(f"检查类型: {check_type}")
    logger.info(f"方法: 区间传播\n")
    logger.info("变量:")
    for name, (bw, signed, lo, hi) in var_info.items():
        r = f"[{lo}, {hi}]" if lo != hi else f"{lo} (固定)"
        logger.info(f"  {name}: {'int' if signed else 'uint'}{bw}_t {r}")
    logger.info("")
    logger.info("⚠️ 能力边界: 本工具仅分析算术表达式(+ - * / %)的溢出/回绕/除零。")
    logger.info("   不覆盖: 类型转换截断(如INT32→half)、有符号/无符号比较语义、")
    logger.info("   返回类型宽度不匹配。")
    logger.info("   若本工具输出 SAFE 或报错，但代码涉及上述场景 → 必须手动推演。\n")


def print_safe(check_type: str):
    label = _check_type_label(check_type)
    logger.info(f"结果: 安全 ✅  给定值域内所有边界组合均不触发{label}。")
    logger.info(f"   ⚠️ 注意: 此结论仅覆盖算术{label}。若代码还涉及类型转换截断、")
    logger.info(f"   有符号/无符号比较、返回类型宽度不匹配 → 需额外手工推演。\n")


def print_violation(check_type: str, counter: Dict[str, int],
                    var_info: Dict[str, VarInfo], description: str,
                    expr_range: Tuple[int, int]):
    label = _check_type_label(check_type)
    logger.info(f"结果: 存在{label}风险 ⚠️")
    logger.info(f"  {description}\n")
    logger.info("反例 (边界值组合):")
    for name, val in counter.items():
        _, _, lo, hi = var_info[name]
        tag = ""
        if val == hi:
            tag = " ← 触及上限"
        elif val == lo:
            tag = " ← 触及下限"
        logger.info(f"  {name} = {val}{tag}")
    logger.info(f"\n  表达式区间: [{expr_range[0]}, {expr_range[1]}]\n")


# ─── Main check ──────────────────────────────────────────────────

def check(expr_str: str, var_specs: List[str], check_type: str) -> int:
    var_info = parse_variables(var_specs)
    tokens = tokenize(expr_str)
    ast = Parser(tokens).parse()
    used_vars = collect_vars(ast)

    for v in used_vars:
        if v not in var_info:
            raise ValueError(f"表达式中使用了未定义的变量: {v!r}")

    print_summary(expr_str, check_type, var_info)

    # Check divzero by scanning AST
    if check_type == 'divzero':
        counter = _check_divzero(ast, var_info)
        if counter:
            print_violation(check_type, counter, var_info,
                            "除数可以为零", (0, 0))
            return 1
        print_safe(check_type)
        return 0

    # Compute interval for overflow/wraparound
    result_lo, result_hi = compute_interval(ast, var_info)
    violation = _detect_violation(result_lo, result_hi, var_info,
                                  used_vars, check_type)

    # Determine direction for sensitivity analysis
    want_min = (check_type == 'wraparound')
    if violation:
        _, _, want_min = violation

    # Compute and print sensitivity for all variables
    sens = _compute_sensitivity(ast, var_info, used_vars, check_type, want_min)

    if violation is None:
        print_safe(check_type)
        _print_sensitivity(sens, var_info, True)
        _print_guidance(True, sens, var_info)
        return 0

    # Generate counter-example at boundary values
    description, _, _ = violation
    counter = pick_values(ast, var_info, want_min)
    print_violation(check_type, counter, var_info, description,
                    (result_lo, result_hi))
    _print_sensitivity(sens, var_info, False)
    _print_guidance(False, sens, var_info)
    return 1


HELP_EPILOG = """
示例:
  # 检查无符号回绕
  %(prog)s --expr "totalOutputSize - aivIdx * singleCoreSize" \\
    --vars "aivIdx=uint32_t:0:47" "singleCoreSize=uint32_t:3:3" "totalOutputSize=int64_t:10:1000000" \\
    --check wraparound

  # 检查有符号溢出
  %(prog)s --expr "blockLength * coreNum * 4" \\
    --vars "blockLength=int32_t:1:1024" "coreNum=int32_t:1:48" \\
    --check overflow

  # 检查除零
  %(prog)s --expr "totalSize / coreNum" \\
    --vars "totalSize=uint32_t:0:1000000" "coreNum=uint32_t:0:48" \\
    --check divzero

支持的类型:
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, size_t

表达式支持 C++ 写法:
  GetBlockNum() * GetTaskRation()          函数调用
  tilingData->bSize * antiqSeqSize         成员访问
"""


def main():
    p = argparse.ArgumentParser(
        description="数值边界静态分析工具 — 区间传播法，反例永远在边界值上",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expr", required=True, metavar="EXPR",
                   help="C++ 算术表达式 (支持 + - * / %% 和括号)")
    p.add_argument("--vars", required=True, nargs="+", metavar="VAR",
                   help="操作数定义: name=type:min:max (可多次指定)")
    p.add_argument("--check", required=True,
                   choices=["overflow", "wraparound", "divzero"],
                   help="overflow=有符号溢出 | wraparound=无符号回绕 | divzero=除零")
    args = p.parse_args()
    try:
        return check(args.expr, args.vars, args.check)
    except ValueError as e:
        logger.info(f"错误: {e}")
        logger.info("")
        logger.info("💡 工具无法处理此表达式，切换到手动推演模式:")
        logger.info("  Step 1 — 提取操作数的 C++ 类型和值域（Grep 声明位置 + constexpr + 赋值链）")
        logger.info("  Step 2 — 按 SEC-2.1/2.2/2.3 检视策略手工判定:")
        logger.info("    类型转换截断: 源值域上限 > 目标类型上限 → 溢出")
        logger.info("    有符号/无符号比较: 检查值域是否跨越有符号最大值")
        logger.info("    返回类型宽度: 内部计算值域上限 > 返回类型最大值 → 截断")
        logger.info("    INT64_MIN 取反: 检查是否有 INT64_MIN 防护（std::abs 或显式判断）")
        logger.info("  Step 3 — 不确定 → SUSPICIOUS + 标注边界不确定的关键变量")
        sys.exit(2)


if __name__ == "__main__":
    sys.exit(main())
