# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Activation / checkpoint hidden-state CPU offload: pin vs pageable with async on.

Holds use_streams=True. Compares use_pin_memory True vs False. Do not treat
this as DeepSpeed #8282's async-vs-blocking table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_PIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIN_ROOT not in sys.path:
    sys.path.insert(0, _PIN_ROOT)

from common import (  # noqa: E402
    dist_env,
    patch_cpp_extension_drop_cxx17,
    print_arm_result,
    print_driver_result,
    run_arm_subprocess,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default=None, help="HF model id (real weights); ZeRO-3 CPU offload. Omit for synthetic."
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="load architecture from config with random weights (no from_pretrained)",
    )
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=(2, 3),
        default=2,
        help="ZeRO stage for the --model branch (Tunji row = ZeRO-2)",
    )
    parser.add_argument(
        "--pin-offload-axis",
        action="store_true",
        help="pin axis = ZeRO offload buffers (param/optimizer), not just the module",
    )
    parser.add_argument(
        "--pin-first", action="store_true", help="run the pinned arm before the pageable arm (order-reversal check)"
    )
    parser.add_argument(
        "--no-zero",
        action="store_true",
        help="upstream default config: plain GPU AdamW, no ZeRO/Liger (real model only)",
    )
    parser.add_argument("--pin", type=int, default=None, help="internal: use_pin_memory 0/1")
    return parser.parse_args()


def run_arm(args):
    # 多卡（PIN_BENCH_MULTIRANK=1）时环境由 driver 注入（RANK/WORLD_SIZE/端口），
    # 跳过 dist_env 避免两个 rank 各拿一个随机端口无法会合。
    if os.environ.get("PIN_BENCH_MULTIRANK") != "1":
        dist_env()
    patch_cpp_extension_drop_cxx17()

    import torch
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.activation_checkpointing.offload_activations import CheckpointHiddenStatesOffload
    from torch.utils.checkpoint import checkpoint

    accelerator = get_accelerator()
    if not accelerator.is_available():
        raise RuntimeError(f"No {accelerator.device_name()} device is available")
    accelerator.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = accelerator.current_device_name()

    class Block(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.fc1 = torch.nn.Linear(hidden, 4 * hidden)
            self.fc2 = torch.nn.Linear(4 * hidden, hidden)

        def forward(self, hidden_states):
            return self.fc2(torch.nn.functional.gelu(self.fc1(hidden_states)))

    class Net(torch.nn.Module):
        def __init__(self, hidden, layers):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(hidden) for _ in range(layers)])

        def forward(self, hidden_states, offload):
            x = hidden_states
            for block in self.blocks:
                offload.mark(x)
                x = x + checkpoint(block, x, use_reentrant=False)
            return x.sum()

    # 真实权重分支（本地适配，见 汇总.md）：Tunji 行口径 ZeRO-2 CPU AdamW +
    # Liger/TiledMLP。8B 的 ZeRO-2 静态显存（参数+梯度 32.8GB）超单卡 29.5GiB，
    # 需两卡双进程（driver 注入 RANK/WORLD_SIZE）；逐层 checkpoint 压激活；
    # pin 轴只动 offload 模块的 use_pin_memory。
    if args.model:
        import types

        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import create_causal_mask, create_sliding_window_causal_mask

        if args.no_zero:
            # 上游默认口径：普通 GPU AdamW、无 ZeRO/Liger——与合成分支（1.23× 出处）
            # 同配置，只换真实模型；不传 torch_dtype，加载 checkpoint 原生 dtype
            # （Qwen3 为 bf16——910B 上 bf16 计算 + FSA 才是快路径）。
            if args.random_init:
                model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model))
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model)
            model = model.to(device)
            base_model = model.model
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        else:
            import deepspeed

            if args.random_init:
                model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model))
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model)
            # Liger/TiledMLP 同款：融合 swiglu/RMSNorm/rope/LCE + 分片 MLP；
            # rms_norm 强制 fp32 accumulate（ascend 后端 auto casting 数值偏大）。
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3

            apply_liger_kernel_to_qwen3(model)
            for module in model.modules():
                if getattr(module, "casting_mode", None) == "llama":
                    module.casting_mode = 1

            class ZeRO2TiledMLP(torch.autograd.Function):
                # TiledMLP 的 ZeRO-2 兼容版：逐 shard 重算的 backward 里，除最后一片外
                # 置 ds_grad_is_ready=False，让 ZeRO-2 只在全部梯度累加完后 reduce 一次
                # （Liger 移植自 DeepSpeed 原版时丢了这个协作逻辑，直接嵌套 backward
                # 会触发 "already been reduced" 断言）。

                @staticmethod
                def forward(ctx, fn, mlp_module, x, shards, params):
                    ctx.fn = fn
                    ctx.mlp_module = mlp_module
                    ctx.shards = shards
                    ctx.params = params
                    ctx.save_for_backward(x)
                    x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
                    with torch.no_grad():
                        output_shards = [fn(mlp_module, x_shard) for x_shard in x_shards]
                    return torch.cat(output_shards, dim=-2)

                @staticmethod
                def backward(ctx, *grads):
                    fn = ctx.fn
                    (x,) = ctx.saved_tensors
                    mlp_module = ctx.mlp_module
                    shards = ctx.shards
                    params = ctx.params
                    x_requires_grad = x.requires_grad
                    x = x.detach()
                    x.requires_grad_(x_requires_grad)
                    hidden_size = x.shape[-1]
                    x_shape_orig = x.shape
                    x = x.view(-1, hidden_size)
                    incoming_grad = grads[0].view(-1, hidden_size)
                    x_grad = torch.zeros_like(x)
                    x_shards = list(torch.chunk(x, chunks=shards, dim=0))
                    for i, x_shard in enumerate(x_shards):
                        x_shard.requires_grad_(x_requires_grad)
                        shard_step = x_shards[i].shape[0]
                        shard_offset = i * x_shards[0].shape[0]
                        x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
                        incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
                        if params is not None:
                            for param in params:
                                param.ds_grad_is_ready = i + 1 == shards
                        with torch.enable_grad():
                            output = fn(mlp_module, x_shard)
                        torch.autograd.backward(output, incoming_grad_shard)
                    x_grad = x_grad.view(x_shape_orig)
                    return (None, None, x_grad, None, None)

            for layer in model.model.layers:
                orig_mlp_forward = layer.mlp.forward
                mlp_params = [p for p in layer.mlp.parameters() if p.requires_grad]

                def _tiled_forward(self, x, _orig=orig_mlp_forward, _params=mlp_params):
                    return ZeRO2TiledMLP.apply(lambda _m, t: _orig(t), self, x, 2, _params)

                layer.mlp.forward = types.MethodType(_tiled_forward, layer.mlp)
            zero_cfg = {"stage": args.zero_stage}
            # 默认口径：offload 的 pin 双关（两臂一致），pin 轴只动激活 offload 模块。
            # --pin-offload-axis：pin 轴改为 ZeRO offload buffer（Tunji 行"CPU AdamW"
            # 的 pin 语义候选），两臂随 --pin 翻转。
            offload_pin = bool(args.pin) and args.pin_offload_axis
            if args.zero_stage == 3:
                zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": offload_pin}
            zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": offload_pin}
            ds_config = {
                "train_micro_batch_size_per_gpu": args.batch,
                "gradient_accumulation_steps": 1,
                "bf16": {"enabled": True},
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
                "zero_optimization": zero_cfg,
            }
            model, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
            base_model = model.module.model
            opt = None
        vocab = model.get_input_embeddings().weight.shape[0]
        dev = model.device
        ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=dev)
        cache_position = torch.arange(0, args.seq, device=dev)
        position_ids = cache_position.unsqueeze(0)
        # 掩码只依赖 embeddings 的形状/设备，用 detach 建一次（常量，无梯度连接）。
        inputs_embeds = base_model.embed_tokens(ids).detach()
        mask_kwargs = {
            "config": base_model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if base_model.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        def forward_loss():
            # 每步重建 embeddings 与位置编码：循环外复用会让第 2 步 backward 撞上
            # 已释放的图（"backward a second time"）。
            inputs_embeds = base_model.embed_tokens(ids)
            position_embeddings = base_model.rotary_emb(inputs_embeds, position_ids)
            h = inputs_embeds
            for decoder_layer in base_model.layers:
                offload.mark(h)
                h = checkpoint(
                    decoder_layer,
                    h,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    use_reentrant=False,
                )
            h = base_model.norm(h)
            return model.lm_head(h).sum()

    else:
        model = Net(args.hidden, args.layers).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(args.batch, args.seq, args.hidden, device=device, requires_grad=True)

        def forward_loss():
            return model(x, offload)

    # Async side stream stays on; pin vs pageable is the only axis.
    offload = CheckpointHiddenStatesOffload(
        use_pin_memory=bool(args.pin), use_streams=True, min_offload_bytes=0, keep_last_count=1
    )

    step_times = []
    for step in range(args.warmup + args.steps):
        accelerator.synchronize()
        t0 = time.perf_counter()
        if opt is None:
            model.zero_grad()
        else:
            opt.zero_grad(set_to_none=True)
        with offload:
            loss = forward_loss()
            if opt is None:
                model.backward(loss)
            else:
                loss.backward()
        if opt is None:
            model.step()
        else:
            opt.step()
        accelerator.synchronize()
        if step >= args.warmup:
            step_times.append(time.perf_counter() - t0)
        offload.reset()

    def _peak():
        try:
            return round(torch.get_device_module(device).max_memory_allocated() / 1e9, 2)
        except Exception:
            return None

    print_arm_result(
        {
            "experiment": "activation_offload",
            "use_pin_memory": bool(args.pin),
            "use_streams": True,
            "device": accelerator.device_name(),
            "model": args.model or f"synthetic-h{args.hidden}-l{args.layers}",
            "hidden": args.hidden,
            "layers": args.layers,
            "batch": args.batch,
            "seq": args.seq,
            "steps": len(step_times),
            "step_avg_s": sum(step_times) / len(step_times),
            "step_min_s": min(step_times),
            "gpu_peak_gb": _peak(),
        }
    )


def run_driver(args):
    script = os.path.abspath(__file__)
    base = [
        "--hidden",
        str(args.hidden),
        "--layers",
        str(args.layers),
        "--batch",
        str(args.batch),
        "--seq",
        str(args.seq),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
    ]
    if args.model:
        base += ["--model", args.model]
        base += ["--zero-stage", str(args.zero_stage)]
        if args.random_init:
            base += ["--random-init"]
        if args.pin_offload_axis:
            base += ["--pin-offload-axis"]
        if args.no_zero:
            base += ["--no-zero"]
    results = {}
    arms = [("pageable", 0), ("pinned", 1)]
    if args.pin_first:
        arms = list(reversed(arms))
    for name, pin in arms:
        results[name] = run_arm_subprocess(script, base + ["--pin", str(pin)])

    pageable = results["pageable"]
    pinned = results["pinned"]
    print("\n================ Activation offload (async on) ================")
    print(
        f"device: {pinned['device']}  hidden: {pinned['hidden']}  layers: {pinned['layers']}  "
        f"batch: {pinned['batch']} seq: {pinned['seq']}"
    )
    print(f"{'arm':<22}{'avg step (s)':>14}{'min step (s)':>14}{'GPU peak (GB)':>16}")
    for name in ("pageable", "pinned"):
        row = results[name]
        print(f"{name:<22}{row['step_avg_s']:>14.3f}{row['step_min_s']:>14.3f}{(row['gpu_peak_gb'] or 0):>16.2f}")
    speedup = pageable["step_avg_s"] / pinned["step_avg_s"]
    print()
    print(f"pin vs pageable step-time ratio: {speedup:.2f}x (async held on)")
    print_driver_result({"pageable": pageable, "pinned": pinned, "speedup": speedup})


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.pin is None:
        run_driver(parsed)
    else:
        run_arm(parsed)
