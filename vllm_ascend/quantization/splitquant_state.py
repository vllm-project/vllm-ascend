""" GDN state I/O. Keep the original arithmetic and dtype conversion order."""
import torch
import torch.nn.functional as F

COUNTERS = dict(conv=0, recurrent=0, gather=0, scatter=0)


def gather(cache, indices):
    from .w4a4_splitquant import load_splitquant_operator
    load_splitquant_operator()
    COUNTERS['gather'] += 1
    return torch.ops.npu.splitquant_state_gather(cache, indices)


def scatter(cache, source, indices):
    from .w4a4_splitquant import load_splitquant_operator
    load_splitquant_operator()
    COUNTERS['scatter'] += 1
    return torch.ops.npu.splitquant_state_scatter(cache, source.contiguous(), indices)


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None,
                         conv_state_indices=None, **kwargs):
    if x.ndim != 2:
        raise NotImplementedError(' supports single-token causal decode')
    COUNTERS['conv'] += 1
    indices = (conv_state_indices.long() if conv_state_indices is not None
               else torch.arange(x.shape[0], device=x.device))
    # vLLM cache is [pages,width,channels] with padded page stride.
    base = conv_state.transpose(-1, -2)
    history = gather(base, indices).transpose(-1, -2)
    window = torch.cat((history[..., -(weight.shape[1]-1):], x.unsqueeze(-1)), -1)
    out = (window.float() * weight.float()).sum(-1)
    if bias is not None:
        out = out + bias.float()
    if activation in (True, 'silu', 'swish'):
        out = F.silu(out)
    source = window[..., -conv_state.shape[-1]:].to(conv_state.dtype)
    scatter(base, source.transpose(-1, -2), indices)
    return out.to(x.dtype)


def recurrent_gated_delta_rule(q, k, v, g, beta, initial_state,
                               inplace_final_state=True, cu_seqlens=None,
                               ssm_state_indices=None,
                               use_qk_l2norm_in_kernel=False, **kwargs):
    if kwargs.get('num_accepted_tokens') is not None:
        raise NotImplementedError('Speculative GDN decode is not supported')
    COUNTERS['recurrent'] += 1
    nseq = len(cu_seqlens) - 1
    indices = (ssm_state_indices[:nseq] if ssm_state_indices is not None
               else torch.arange(nseq, device=initial_state.device))
    selected = gather(initial_state, indices)
    outs, states = [], []
    for i in range(nseq):
        start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        state = selected[i:i+1].float()
        query, key = q[:, start:end].float(), k[:, start:end].float()
        if use_qk_l2norm_in_kernel:
            query = F.normalize(query, p=2, dim=-1)
            key = F.normalize(key, p=2, dim=-1)
        ratio = v.shape[2] // k.shape[2]
        query = query.repeat_interleave(ratio, dim=2) * q.shape[-1] ** -0.5
        key = key.repeat_interleave(ratio, dim=2)
        seq_out = []
        for t in range(end - start):
            kt = key[:, t]
            state = state * g[:, start + t].float().exp()[..., None, None]
            residual = v[:, start + t].float() - (state * kt[..., None]).sum(-2)
            state = state + kt[..., None] * (residual * beta[:, start + t].float()[..., None])[..., None, :]
            seq_out.append((state * query[:, t, :, :, None]).sum(-2))
        outs.append(torch.stack(seq_out, dim=1).to(v.dtype))
        states.append(state.to(initial_state.dtype))
    if inplace_final_state:
        scatter(initial_state, torch.cat(states, dim=0), indices)
    return torch.cat(outs, dim=1), initial_state


def install(model):
    if model.quant_config is None:
        raise ValueError(' state optimization requires the quantized model')
    count = 0
    for layer in model.modules():
        if hasattr(layer, 'fix_query_key_value_ordering'):
            layer._splitquant_state_ops = True
            count += 1
    num_layers = model.config.num_hidden_layers
    if count != num_layers:
        raise RuntimeError(f'Expected {num_layers} GDN layers, found {count}')
    model.splitquant_state_layers = count
