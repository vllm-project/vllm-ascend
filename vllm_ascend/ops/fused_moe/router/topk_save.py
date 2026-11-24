# -*- coding: utf-8 -*-
"""TOPK_SAVE_PATH 热点专家采集(moe-small-verify-big skill 机制移植,图模式安全)。

移植自 vllm-ascend ba1d0ca 补丁的 TOPK_SAVE_PATH 语义,适配 v0.30.0 V2 MoE 栈:
  - 原: select_experts 内 np.savetxt 每层追加 topk_ids 文本(python 落盘,图回放不执行)
  - 本: 模型构建期预分配每层 int64[num_experts] 设备计数器;路由后
        counts += torch.bincount(topk_ids) —— 纯设备算子,ACL graph 捕获后
        每次 prefill(eager)与 decode(回放)都会累加,与图模式/FULL_DECODE_ONLY 兼容。
  - 数据量恒定(层×num_experts 计数),长时间运行不增长。
  - 后台线程每 TOPK_SAVE_FLUSH_SEC(默认 30s)落盘一次;目录下出现 ARM 文件后,
    以当时计数为基线扣除(warmup/dummy/启动流量),输出净计数。
  - 输出: {TOPK_SAVE_PATH}/top{K}_experts.json —— 与仓库 stat_moe.py /
    expert_pruned.py 的 {"<layer>": {"topk_ids": [...], "topk_counts": [...]}} 同构;
    另存 topk_counts.npz(全量逐层计数)与 topk_save_meta.json。

使用: 服务环境变量 export TOPK_SAVE_PATH=<dir>(非空开启;仅 TP rank0 累加)。
"""
import atexit
import json
import os
import threading
import time

import numpy as np
import torch

_STATE = {
    'enabled': False,
    'tried_init': False,
    'rank': None,
    'out_dir': None,
    'flush_sec': 30.0,
    'topk_k': 64,
    'layers': {},      # name -> int64[num_experts] device tensor(预分配)
    'num_experts': {}, # name -> int
    'updates': {},     # name -> int (python 侧执行次数: eager 每步+捕获一次性)
    'base': {},        # ARM 时刻快照
    'armed': False,
    'lock': threading.Lock(),
    'warned': set(),
}


def _log_once(msg):
    key = msg[:60]
    if key not in _STATE['warned']:
        _STATE['warned'].add(key)
        try:
            print(f'[topk_save] {msg}', flush=True)
        except Exception:
            pass


def _init() -> bool:
    if _STATE['tried_init']:
        return _STATE['enabled']
    _STATE['tried_init'] = True
    path = os.environ.get('TOPK_SAVE_PATH', '')
    if not path:
        return False
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        _log_once(f'cannot create {path}: {e}')
        return False
    try:
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    except ValueError:
        rank = 0
    _STATE['enabled'] = True
    _STATE['rank'] = rank
    _STATE['out_dir'] = path
    try:
        _STATE['flush_sec'] = float(os.environ.get('TOPK_SAVE_FLUSH_SEC', '30'))
    except ValueError:
        pass
    try:
        _STATE['topk_k'] = int(os.environ.get('TOPK_SAVE_TOPK', '64'))
    except ValueError:
        pass
    if rank == 0:
        t = threading.Thread(target=_flush_loop, daemon=True, name='topk-save-flush')
        t.start()
        atexit.register(dump_final)
        _log_once(f'enabled: out_dir={path} flush={_STATE["flush_sec"]}s rank0 only')
    return True


def name_router(experts_obj, prefix: str) -> None:
    """在 AscendKimiMoE.__init__ 尾部调用:为该层 router 标注层名并预分配计数器。"""
    try:
        if not _init():
            return
        if _STATE['rank'] != 0:
            return
        r = getattr(experts_obj, 'router', None)
        if r is None:
            for v in vars(experts_obj).values():
                if 'TopKRouter' in type(v).__name__:
                    r = v
                    break
        if r is None:
            _log_once(f'no router found under {prefix}')
            return
        name = str(prefix)
        i = name.find('layers.')
        if i >= 0:
            name = 'model.' + name[i:]
        n = getattr(r, 'num_logical_experts', None)
        if n is None:
            n = getattr(r, 'global_num_experts', None)
        if not isinstance(n, int) or n <= 0:
            _log_once(f'unknown num_experts for {name}, skip')
            return
        r._wb_layer_name = name
        with _STATE['lock']:
            if name not in _STATE['layers']:
                _STATE['layers'][name] = torch.zeros(n, dtype=torch.int64, device='npu')
                _STATE['num_experts'][name] = n
    except Exception as e:
        _log_once(f'name_router failed: {e}')


def accumulate(router, topk_ids) -> None:
    """在 AscendFusedTopKRouter._compute_routing 返回前调用(仅设备算子,图安全)。"""
    try:
        if not _STATE['enabled'] and not _init():
            return
        if _STATE['rank'] != 0:
            return
        name = getattr(router, '_wb_layer_name', None)
        if name is None:
            return
        counts = _STATE['layers'].get(name)
        if counts is None:
            return
        n = _STATE['num_experts'][name]
        flat = topk_ids.reshape(-1).to(torch.int64)
        bc = torch.bincount(flat, minlength=n).to(torch.int64)
        if bc.shape[0] != n:
            _log_once(f'{name}: bincount size {bc.shape[0]} != {n}, skip once')
            return
        counts += bc
        # python 侧计数仅用于诊断: eager 模式每步 +1;图模式仅捕获时 +1
        _STATE['updates'][name] = _STATE['updates'].get(name, 0) + 1
    except Exception as e:
        _log_once(f'accumulate failed: {e}')


def _arm_check() -> None:
    arm = os.path.join(_STATE['out_dir'], 'ARM')
    if not _STATE['armed'] and os.path.exists(arm):
        with _STATE['lock']:
            if not _STATE['armed']:
                _STATE['base'] = {k: v.clone() for k, v in _STATE['layers'].items()}
                _STATE['armed'] = True
                _log_once('armed: baseline snapshot taken (warmup excluded)')


def _effective_locked():
    out = {}
    for k, v in _STATE['layers'].items():
        b = _STATE['base'].get(k) if _STATE['armed'] else None
        out[k] = (v - b) if b is not None else v.clone()
    return out


def _write_locked() -> None:
    eff = _effective_locked()
    if not eff:
        return
    k = _STATE['topk_k']
    result = {}
    npz = {}
    totals = {}
    for name in sorted(eff):
        c = eff[name].cpu().to(torch.int64)
        order = torch.argsort(c, descending=True, stable=True)
        top = [int(x) for x in order[:k].tolist()]
        cnt = [int(x) for x in c[top].tolist()]
        result[name] = {'topk_ids': top, 'topk_counts': cnt}
        npz[name] = c.numpy()
        totals[name] = int(c.sum().item())
    d = _STATE['out_dir']
    with open(os.path.join(d, f'top{k}_experts.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(d, 'topk_counts.npz'), **npz)
    meta = {
        'armed': _STATE['armed'],
        'rank': _STATE['rank'],
        'totals': totals,
        'python_side_updates': dict(_STATE['updates']),
        'flush_sec': _STATE['flush_sec'],
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(d, 'topk_save_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _flush_loop() -> None:
    while True:
        time.sleep(_STATE['flush_sec'])
        try:
            _arm_check()
            with _STATE['lock']:
                _write_locked()
        except Exception as e:
            _log_once(f'flush failed: {e}')


def dump_final() -> None:
    try:
        if not _STATE['enabled'] or _STATE['rank'] != 0:
            return
        torch.npu.synchronize()
        with _STATE['lock']:
            _write_locked()
        _log_once('final dump done')
    except Exception as e:
        _log_once(f'final dump failed: {e}')
