import torch
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
import xops
import torch_npu

DEVICE_ID = 0
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

def get_chunk(
    input,  # tensor of shape (S, ...)
    C,      # chunk size
    start   # chunk start position
):
    S = input.shape[0]
    end = start + C
    if end <= S:
        return input[start:end]
    else:
        pad_size = end - S
        if len(input.shape) > 1:
            # for q, k, v
            return F.pad(input[start:], (0, 0, 0, pad_size))
        else:
            return F.pad(input[start:], (0, pad_size))


def stage1_chunk(
    query,  # (C, Dk)
    key,    # (C, Dk)
    value,  # (C, Dv)
    g,      # (C,)
    beta,   # (C,)
    scale
):
    """Stage1: chunk 内预处理，计算逆矩阵并准备 stage2/stage3 所需的中间量。
    
    Gated Delta Rule 的 chunk 内递推:
        S_i = exp(g_i)*S_{i-1} + beta_i*(v_i - k_i @ exp(g_i)*S_{i-1}) ⊗ k_i
    等价于求解线性系统 (I + L) @ v_new = v_beta, 其中:
        L[i,j] = beta_i * (k_i @ k_j^T) * exp(g_cum_i - g_cum_j),  j < i
    本函数通过逐行迭代计算 (I + L)^{-1}，因 L 是严格下三角（幂零）矩阵，有限步精确。
    """
    device = query.device
    C = query.shape[0]

    # kkt[i,j] = k_i @ k_j^T, 再按行乘 beta_i -> L 矩阵的 key 内积部分
    kkt = key.float() @ key.transpose(-1, -2).float()
    kkt = kkt * beta.float().unsqueeze(-1)    # (C, C)

    # 原始 Q@K^T 注意力分数，供 stage3 使用
    qkt = query.float() @ key.transpose(-1, -2).float()    # (C, C)

    # 累积 gating: g_cum[i] = g_0 + g_1 + ... + g_i
    g_cum = g.cumsum(dim=-1)  # (C,)
    g_cum_exp = g_cum.exp()   # (C,)

    # 衰减矩阵: attn[i,j] = exp(g_cum_i - g_cum_j), 仅 j < i（严格下三角）
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1)
    attn = ((g_cum[:, None] - g_cum[None, :]) * lower).exp() * lower   # (C, C)

    # 构造 -L 并通过逐行迭代求 (I + L)^{-1} 的下三角部分
    # 原理: 对严格下三角 L, (I+L)^{-1} 可逐行递推:
    #   A[i,j] = -sum_{k=0}^{i-1} L[i,k]*A[k,j], 其中 A[k,k]=1
    attn_1 = kkt * attn   # (C, C)
    attn_1 *= -1.0
    for i in range(1, C):
        row = attn_1[i, :i].clone()
        sub = attn_1[:i, :i].clone()
        attn_1[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn_1 = attn_1 + torch.eye(C, dtype=attn_1.dtype, device=attn_1.device)

    # kg[i] = key_i * exp(g_cum_last - g_cum_i): key 衰减到 chunk 末尾，用于 stage2 state 更新
    kg = key.float() * (g_cum[-1, None] - g_cum).exp()[..., None]  # (C, Dk)
    # k_cumdecay[i] = -beta_i * exp(g_cum_i) * key_i: key 与 state 交互的系数
    k_cumdecay = (beta.unsqueeze(-1).float() * g_cum_exp[:, None]) * (-1) * key.float()      # (C, Dk)

    # v_beta = beta * value: delta rule 中的新值权重
    v_beta = value.float() * beta.unsqueeze(-1).float()  # (C, Dv)
    # q_prime = scale * query * exp(g_cum): 吸收 scale 和衰减后的 query，用于 stage2 inter-chunk attention
    q_prime = query.float() * scale * g_cum_exp[:, None]       # (C, Dk)

    # 应用逆矩阵: v_inner = (I+L)^{-1} @ v_beta, k_cumdecay = (I+L)^{-1} @ k_cumdecay
    v_inner = attn_1 @ v_beta    # (C, Dv)
    k_cumdecay = attn_1 @ k_cumdecay        # (C, Dk)

    return g_cum, k_cumdecay.to(torch.bfloat16), v_inner, q_prime.to(torch.bfloat16), kg.to(torch.bfloat16), qkt


def stage1(
    query,  # (S, Nk, Dk)
    key,    # (S, Nk, Dk)
    value,  # (S, Nv, Dv)
    g,      # (S, Nv)
    beta,   # (S, Nv)
    scale,
    C       # chunk size
):
    """Stage1 外层循环: 按 head 和 chunk 拆分序列，并行调用 stage1_chunk。"""
    S, Nk, Dk = query.shape
    _, Nv, Dv = value.shape
    device=query.device

    # GQA: 当 value head 数 > key head 数时，复制 q/k 以对齐
    if Nv // Nk > 1:
        query = query.repeat_interleave(Nv // Nk, dim=1)
        key = key.repeat_interleave(Nv // Nk, dim=1)
    
    # 序列长度向上对齐到 chunk size 的整数倍
    padded_seq_len = (S + C - 1) // C * C

    g_cum = torch.zeros((Nv, padded_seq_len), dtype=torch.float32, device=device)
    k_cumdecay = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    v_inner = torch.zeros((Nv, padded_seq_len, Dv), dtype=torch.float32, device=device)
    q_prime = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    kg = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    qkt = torch.zeros((Nv, padded_seq_len, C), dtype=torch.float32, device=device)
    
    loop_range = range(0, padded_seq_len, C)
    for nid in range(Nv):
        for idx in reversed(loop_range):     # use reverse loop to simulate parallel
            q_chunk = get_chunk(query[:, nid, :], C, idx)
            k_chunk = get_chunk(key[:, nid, :], C, idx)
            v_chunk = get_chunk(value[:, nid, :], C, idx)
            g_chunk = get_chunk(g[:, nid], C, idx)
            beta_chunk = get_chunk(beta[:, nid], C, idx)

            g_cum_chunk, k_cumdecay_chunk, v_inner_chunk, qg_chunk, kg_chunk, qkt_chunk = stage1_chunk(
                q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, scale)
                    
            g_cum[nid, idx:idx + C] = g_cum_chunk
            k_cumdecay[nid, idx:idx + C, :] = k_cumdecay_chunk
            v_inner[nid, idx:idx + C, :] = v_inner_chunk
            q_prime[nid, idx:idx + C, :] = qg_chunk
            kg[nid, idx:idx + C, :] = kg_chunk
            qkt[nid, idx:idx + C, :] = qkt_chunk

    return g_cum, k_cumdecay, v_inner, q_prime, kg, qkt

def stage2_chunk(
    q_prime,    # (C, Dk)
    v_inner,    # (C, Dv)
    g_cum,  # (C)
    k_cumdecay, # (C, Dk)
    state,      # (Dv, Dk)
    kg,         # (C, Dk)
):
    """Stage2: 串行处理 chunk 间的 state 传递。
    
    对每个 chunk:
    1. 计算 inter-chunk attention: q_prime @ state^T (query 对历史 state 的注意力)
    2. 修正 value: v_new = v_inner + k_cumdecay @ state^T (加入 state 贡献)
    3. 更新 state: state_new = exp(chunk_total_g) * state_old + v_new^T @ kg
    """
    bf16_state = state.to(torch.bfloat16)

    # inter-chunk attention: q_prime_i @ S^T = scale*q_i*exp(g_cum_i) 对 chunk 起始 state 的注意力
    attn_inter = q_prime.float() @ bf16_state.float().transpose(0, 1)         # (C, Dv)

    # state 对 value 的修正: k_cumdecay_i = -beta_i*exp(g_cum_i)*k_i 经逆矩阵处理后与 state 交互
    v_prime = k_cumdecay.float() @ bf16_state.float().transpose(0, 1)           # (C, Dv)
    # 最终 corrected value = chunk 内 (I+L)^{-1} 处理的值 + state 修正
    v_new = v_inner + v_prime                                # (C, Dv)

    # state 贡献: sum_i v_new_i ⊗ (key_i * exp(g_cum_last - g_cum_i))
    state_out = v_new.transpose(0, 1).to(torch.bfloat16).float() @ kg.float()  # (Dv, Dk)

    # state 更新: 旧 state 按 chunk 总衰减衰减 + 本 chunk 贡献
    state_old = state.float() * g_cum.exp()[-1]
    state_old = state_old + state_out                        # (Dv, Dk)

    return state_old, attn_inter, v_new.to(torch.bfloat16)


def stage2(
    q_prime,    # (Nv, Sp, Dk)
    v_inner,    # (Nv, Sp, Dv)
    g_cum,  # (Nv, Sp)
    k_cumdecay, # (Nv, Sp, Dk)
    state,      # (Nv, Dv, Dk)
    kg,         # (Nv, Sp, Dk)
    C,           # chunk size
):
    """Stage2 外层循环: 按 head 串行遍历 chunk，逐步更新 state（不可并行）。"""
    Nv, Sp, Dv = v_inner.shape
    _, _, Dk = q_prime.shape
    attn_inter = torch.zeros((Nv, Sp, Dv), dtype=torch.float32, device=q_prime.device)
    v_new = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=q_prime.device)
    final_state = torch.empty_like(state).to(torch.float32)

    for nid in range(Nv):
        cur_state = state[nid]
        for idx in range(0, Sp, C):
            qg_chunk = q_prime[nid, idx:idx + C, :]
            v_inner_chunk = v_inner[nid, idx:idx + C, :]
            g_cum_chunk = g_cum[nid, idx:idx + C]
            k_cumdecay_chunk = k_cumdecay[nid, idx:idx + C, :]
            kg_chunk = kg[nid, idx:idx + C, :]
            cur_state, attn_inter_chunk, v_new_chunk = stage2_chunk(
                qg_chunk, v_inner_chunk, g_cum_chunk, k_cumdecay_chunk, cur_state, kg_chunk)
                
            attn_inter[nid, idx:idx + C, :] = attn_inter_chunk
            v_new[nid, idx:idx + C, :] = v_new_chunk
        final_state[nid, ...] = cur_state
    return final_state, attn_inter, v_new

def stage3_chunk(
    qkt,       # (C, C)
    value,       # (C, Dv)
    scale,       # float
    g_cum,   # (C,)
    attn_inter,  # (C, Dv)
    v_new        # (C, Dv)
):
    """Stage3: 合并 inter-chunk 和 intra-chunk 注意力，得到最终输出。
    
    o_i = attn_inter_i + sum_{j<=i} scale*(q_i@k_j^T)*exp(g_cum_i - g_cum_j) * v_new_j
    """
    device = value.device
    C = value.shape[0]
    core_attn_out = torch.zeros_like(value).to(torch.bfloat16)  # (C, Dv)
    # 构造下三角 causal mask（含对角线），用于 intra-chunk 因果注意力
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0)
    # masked_qkt[i,j] = (q_i @ k_j^T) * scale * exp(g_cum_i - g_cum_j), 仅 j <= i
    masked_qkt = qkt.float() * scale * ((g_cum[:, None] - g_cum[None, :]) * lower).exp() * lower.float()
    # 模拟 NPU bf16 输入 fp32 累加：先截断到 bf16 再转 float 做矩阵乘
    attn_inner = masked_qkt.to(torch.bfloat16).float() @ v_new.to(torch.bfloat16).float()           # (C, Dv)
    # 最终输出 = inter-chunk（来自 state）+ intra-chunk（来自 chunk 内部注意力）
    core_attn_out = (attn_inter + attn_inner).to(torch.bfloat16)   # (C, Dv)

    return core_attn_out


def stage3(
    qkt,         # (Nv, Sp, C)
    value,       # (S, Nv, Dv)
    scale,       # float
    g_cum,   # (Nv, Sp)
    attn_inter,  # (Nv, Sp, Dv)
    v_new,       # (Nv, Sp, Dv)
    C,            # chunk size
):
    Nv, Sp, Dv = attn_inter.shape
    S, _, _ = value.shape
    assert Sp == (S + C - 1) // C * C
    
    attn_out = torch.empty((Sp, Nv, Dv), dtype=torch.bfloat16, device=value.device)   # (Sp, Nv, Dv)

    for nid in range(Nv):
        for idx in range(0, Sp, C):
            v_chunk = get_chunk(value[:, nid, :], C, idx)
            g_cum_chunk = g_cum[nid, idx:idx + C]
            attn_inter_chunk = attn_inter[nid, idx:idx + C, :]
            v_new_chunk = v_new[nid, idx:idx + C, :]
            qkt_chunk = qkt[nid, idx:idx + C, :]
            attn_out_chunk = stage3_chunk(qkt_chunk, v_chunk, scale, g_cum_chunk, attn_inter_chunk, v_new_chunk)
            
            attn_out[idx:idx + C, nid, ...] = attn_out_chunk
    

    return attn_out

def chunk_gdn_benchmark(
    query,              # (T, Nk, Dk)
    key,                # (T, Nk, Dk)
    value,              # (T, Nv, Dv)
    beta,               # (T, Nv)
    scale,              # float
    initial_state,      # (B, Nv, Dv, Dk)
    actual_seq_lengths, # (B,)
    g = None            # (T, Nv)
):
    """Chunked Gated Delta Rule 完整 golden 实现。
    
    整体流程 (per batch):
      Stage1: 各 chunk 并行 —— 计算逆矩阵 (I+L)^{-1}，产出 q_prime, v_inner, k_cumdecay, kg, qkt
      Stage2: chunk 间串行 —— 用 state 修正 value，更新 state，计算 inter-chunk attention
      Stage3: 各 chunk 并行 —— 合并 inter-chunk + intra-chunk attention 得到最终输出
    """
    T, Nk, Dk = query.shape
    B, Nv, Dv, _ = initial_state.shape
    device=query.device

    if g is None:
        g = torch.zeros((T, Nv), dtype=torch.float32, device=device)
    attn_out = torch.empty((T, Nv, Dv), dtype=query.dtype, device=device)
    attn_out = (attn_out).to(torch.bfloat16)
    final_state = torch.empty_like(initial_state).to(torch.float32)

    start = 0
    C = 64
    for bid in range(B):
        cur_state = initial_state[bid].clone()
        S = actual_seq_lengths[bid]
        end = start + S

        # Stage1: chunk 内预处理（可并行），计算逆矩阵相关中间量
        g_cum, k_cum_decay, v_inner, q_prime, kg, qkt = stage1(
            query[start:end], key[start:end], value[start:end], g[start:end], beta[start:end], scale, C)

        # Stage2: chunk 间串行，state 传递与 inter-chunk attention
        cur_state, attn_inter, v_new = stage2(
            q_prime, v_inner, g_cum, k_cum_decay, cur_state, kg, C)
        final_state[bid] = cur_state

        # Stage3: chunk 内合并 attention（可并行），得到最终输出
        attn_out_paddend = stage3(
            qkt, value[start:end], scale, g_cum, attn_inter, v_new, C)
        
        attn_out[start:end, ...] = attn_out_paddend[:S]
        start = end

    return attn_out, final_state


def print_compare(name, npu, golden, topk=5):
    npu = npu.detach().float().cpu()
    golden = golden.detach().float().cpu()
    diff = (npu - golden).abs()
    flat_diff = diff.flatten()
    flat_npu = npu.flatten()
    flat_golden = golden.flatten()

    max_idx = int(torch.argmax(flat_diff))
    print(f"\n========== compare {name} ==========")
    print(f"shape: npu={tuple(npu.shape)}, golden={tuple(golden.shape)}")
    print(f"abs_err: max={flat_diff[max_idx].item():.6e}, "
          f"mean={flat_diff.mean().item():.6e}, min={flat_diff.min().item():.6e}")
    print(f"max_err_idx={max_idx}")
    print(f"  npu[{max_idx}]    = {flat_npu[max_idx].item():.6e}")
    print(f"  golden[{max_idx}] = {flat_golden[max_idx].item():.6e}")
    print(f"  abs_err          = {flat_diff[max_idx].item():.6e}")

    topk = min(topk, flat_diff.numel())
    topk_vals, topk_idx = torch.topk(flat_diff, k=topk)
    print(f"top-{topk} abs_err:")
    for rank, (idx, err) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), start=1):
        print(f"  #{rank} idx={idx}, err={err:.6e}, "
              f"npu={flat_npu[idx].item():.6e}, golden={flat_golden[idx].item():.6e}")


def generate_test_cases(num_cases=200):
    """生成覆盖 (1, 1536, 8, 24, 128, 128) 的测试参数组合，共 num_cases 个。"""
    import itertools

    batch_sizes = [1, 2, 3, 4]
    seqlens = [64, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536]
    nk_values = [1, 2, 4, 8]
    nv_values = [8, 16, 24, 32]
    dk_values = [128]
    dv_values = [128]

    all_cases = []
    for batch, seq, nk, nv, dk, dv in itertools.product(
        batch_sizes, seqlens, nk_values, nv_values, dk_values, dv_values
    ):
        if nv % nk == 0:
            all_cases.append((batch, seq, nk, nv, dk, dv))

    base_case = (1, 1536, 8, 24, 128, 128)

    step = max(1, len(all_cases) // num_cases)
    selected = all_cases[::step][:num_cases]

    if base_case not in selected:
        selected[-1] = base_case

    assert len(selected) == num_cases, f"Expected {num_cases} cases, got {len(selected)}"
    return selected


class TestChunkGatedDeltaRuleBenchmark(TestCase):

    @staticmethod
    def benchmark_golden(q, k, v, g, beta, scale, initial_state, actual_seq_lengths, dtype):
        if g is None:
            g = torch.zeros((v.shape[0], v.shape[1]), dtype=torch.float32, device=v.device)
        o_bench, state_bench = chunk_gdn_benchmark(
            q.to(dtype),
            k.to(dtype),
            v.to(dtype),
            beta.to(dtype),
            scale,
            initial_state.to(torch.float32),
            actual_seq_lengths,
            g,
        )
        return o_bench.to(torch.float32), state_bench.to(torch.float32)

    def execute_chunk_gated_delta_rule_case(self, batch_size, seqlen, nk, nv, dk, dv, dtype):
        torch_npu.npu.set_device(int(DEVICE_ID))
        device = "npu:%s" % DEVICE_ID
        # ======================== gen input data start =============================
        T = batch_size * seqlen
        q = torch.rand((T, nk, dk), dtype=dtype, device=device)
        k = torch.rand((T, nk, dk), dtype=dtype, device=device)
        v = torch.rand((T, nv, dv), dtype=dtype, device=device)
        g = torch.rand((T, nv), dtype=torch.float32, device=device) * -1.0
        beta = torch.rand((T, nv), dtype=dtype, device=device)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        scale = 1 / (dk ** 0.5)
        initial_state = torch.rand((batch_size, nv, dv, dk), dtype=torch.float32, device=device)
        actual_seq_lengths = torch.tensor([seqlen] * batch_size, dtype=torch.int32, device=device)
        # ======================== gen input data finish =============================

        # calculate on npu
        o_npu, state_npu = torch_npu.npu_x_chunk_gated_delta_rule(
            q, k, v, beta, initial_state, actual_seq_lengths, g, scale)
        o_npu = o_npu.cpu().to(torch.float32)
        state_npu = state_npu.cpu().to(torch.float32)

        # calculate golden on cpu
        o_bench, state_bench = self.benchmark_golden(
            q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), scale,
            initial_state.cpu(), actual_seq_lengths.cpu(), dtype)

        self.assertRtolEqual(o_npu, o_bench, 0.001)
        self.assertRtolEqual(state_npu, state_bench, 0.001)

    def test_single_batch(self):
        self.execute_chunk_gated_delta_rule_case(1, 128, 16, 16, 128, 128, torch.bfloat16)

    def test_multi_batch(self):
        self.execute_chunk_gated_delta_rule_case(4, 256, 16, 16, 128, 128, torch.bfloat16)

    def test_head_num_differ(self):
        self.execute_chunk_gated_delta_rule_case(2, 128, 16, 32, 128, 128, torch.bfloat16)

    def test_base_case(self):
        self.execute_chunk_gated_delta_rule_case(1, 1536, 8, 24, 128, 128, torch.bfloat16)

    def test_generated_cases(self):
        cases = generate_test_cases(num_cases=200)
        total = len(cases)
        for i, (batch_size, seqlen, nk, nv, dk, dv) in enumerate(cases):
            print(f"[{i + 1}/{total}] batch={batch_size}, seqlen={seqlen}, "
                  f"nk={nk}, nv={nv}, dk={dk}, dv={dv}")
            self.execute_chunk_gated_delta_rule_case(
                batch_size, seqlen, nk, nv, dk, dv, torch.bfloat16)


if __name__ == "__main__":
    run_tests()
