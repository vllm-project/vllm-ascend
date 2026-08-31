"""DeepSeek-V4 W8A8 quantization script.

Converts native FP8 weights to W8A8 quantized weights in ModelSlim format,
loadable by vllm-ascend without --quantization flag (auto-detected on NPU).

Supports both DeepSeek-V4-Flash and DeepSeek-V4-DSpark (Flash + Pro) variants.
DSpark is auto-detected via config.json's dspark_block_size field.

Quantization logic references msmodelslim:
- FP8 dequant: msmodelslim/model/deepseek_v4/convert_fp8_to_bf16.py
- W8A8 weight quant: msmodelslim/pytorch/llm_ptq/llm_ptq_tools/flat_quant/components/quantizer.py
- Output format: msmodelslim/pytorch/llm_ptq/llm_ptq_tools/save/

Script form (argparse, shard reading, memory management) references:
- cann-recipes-infer/models/deepseek_v4/utils/convert_model.py
"""

import argparse
import fnmatch
import json
import os
import shutil
from glob import glob

import torch
import yaml
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# torch_npu is optional: only needed when --device npu (or auto with NPU present).
# Importing it registers the NPU backend so tensor.to("npu") works.
try:
    import torch_npu  # noqa: F401  required for tensor.to("npu")

    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False

# W8A8 symmetric per-channel quantization constants
NUM_BITS = 8
Q_MAX = 2 ** (NUM_BITS - 1) - 1  # 127
Q_MIN = -(2 ** (NUM_BITS - 1))  # -128
# Epsilon to prevent division by zero when the input is all zeros
QUANT_EPSILON = 1e-5

# FP8 block size for dequantization (DeepSeek-V4 uses 128x128 blocks)
FP8_BLOCK_SIZE = 128
# MXFP4 block size: 32 FP4 elements (packed into 16 uint8) share 1 block scale
MXFP4_BLOCK_SIZE = 32

# Output shard size (GB), matches YAML save.ascendv1_saver.part_file_size
OUTPUT_SHARD_GB = 4
ONE_GB_BYTES = 1073741824

# Max number of input safetensors shards to keep in memory simultaneously.
# FP8 weight and its scale may land in adjacent shards, so a window of 2 is
# sufficient for the common case; get_tensor() may pull in extra scale shards
# which are evicted by the while-loop after each iteration.
MAX_CACHED_SHARDS = 2

# Quant type strings for quant_model_description.json
QUANT_TYPE_W8A8_DYNAMIC = "W8A8_DYNAMIC"
QUANT_TYPE_W4A8_DYNAMIC = "W4A8_DYNAMIC"
QUANT_TYPE_FLOAT = "FLOAT"

# W4A8 symmetric per-channel quantization constants (4-bit signed: [-8, 7])
W4_NUM_BITS = 4
W4_Q_MAX = 2 ** (W4_NUM_BITS - 1) - 1  # 7
W4_Q_MIN = -(2 ** (W4_NUM_BITS - 1))  # -8
# Bit mask for extracting low 4 bits: (1 << W4_NUM_BITS) - 1
W4_BIT_MASK = (1 << W4_NUM_BITS) - 1  # 0x0F
# scale_bias factor: 2 ** (W4_NUM_BITS - 1), from msmodelslim process_scale formula
W4_SCALE_BIAS_FACTOR = 2 ** (W4_NUM_BITS - 1)  # 8

# DSpark has 3 MTP draft layers (vs Flash's 1); used for post-quantization assertions
DSPARK_MTP_LAYERS = 3

# Checkpoint filename for crash recovery (stored in output directory)
CHECKPOINT_FILENAME = ".quantize_checkpoint.json"


def decode_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 e4m3 weight to bfloat16.

    References: msmodelslim/model/deepseek_v4/convert_fp8_to_bf16.py:decode_fp8
    """
    weight = weight.unflatten(0, (-1, FP8_BLOCK_SIZE)).unflatten(-1, (-1, FP8_BLOCK_SIZE)).float()
    weight = weight * scale[:, None, :, None].float()
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def decode_fp4(packed_fp4_data: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 packed weight to bfloat16.

    References: msmodelslim/model/deepseek_v4/convert_fp8_to_bf16.py:decode_fp4
    """
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed_fp4_data.device,
        dtype=torch.float32,
    )

    uint8 = packed_fp4_data.view(torch.uint8)
    low = uint8 & 0x0F
    high = (uint8 >> 4) & 0x0F
    indices = torch.stack([low, high], dim=-1).flatten(-2)

    sign = 1.0 - 2.0 * ((indices >> 3) & 1).float()
    abs_idx = indices & 0x07
    values = sign * lut[abs_idx.long()]

    # MXFP4: 32 FP4 elements share 1 block scale, so repeat_interleave(32)
    # expands block_scales to match the unpacked weight dimension
    scales_expanded = block_scales.to(torch.float32).repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    return (values * scales_expanded).to(torch.bfloat16)


def weight_quant_sym_perchannel(
    tensor: torch.Tensor, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel symmetric W8A8 weight quantization.

    Returns (quant_weight[int8], weight_scale[fp32], weight_offset[fp32 zeros]).

    References: msmodelslim WeightQuantizer.find_params(sym=True) + sym_quant()
    """
    x = tensor.to(device)
    if x.dim() > 2:
        x = x.flatten(1)

    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    # symmetric: take max of abs(xmin) and xmax, clamp to avoid div-by-zero
    xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=QUANT_EPSILON)
    scale = (xmax / Q_MAX).unsqueeze(-1)  # (M, 1) for broadcasting with x (M, N)
    zero = torch.zeros_like(scale)

    quant_weight = torch.clamp(torch.round(x / scale), Q_MIN, Q_MAX).to(torch.int8)
    return quant_weight.cpu(), scale.cpu().to(torch.float32), zero.cpu().to(torch.float32)


def w4a8_pack_int4(weight: torch.Tensor) -> torch.Tensor:
    """Pack int4 weight to int8 (two 4-bit values per int8 element).

    Aligns with msmodelslim ``format/common/pack.py:w4a8_pack_int4``:
    transpose → pack along output dim → transpose back.
    Input shape [output, input] → output shape [output // 2, input].
    """
    assert weight.dim() == 2, f"Expected 2D tensor, got {weight.dim()}D"
    output_size, input_size = weight.shape
    assert output_size % 2 == 0, f"Output dimension must be even for int4 packing, got {output_size}"

    weight = weight.transpose(0, 1).contiguous()  # [input, output]
    weight = weight.reshape(-1, 2)  # [input * output//2, 2]
    high = torch.bitwise_left_shift(weight[:, 1:], W4_NUM_BITS)
    low = weight[:, :1] & W4_BIT_MASK
    packed = torch.bitwise_or(high, low)
    packed = packed.reshape(input_size, output_size // 2)  # [input, output//2]
    packed = packed.transpose(0, 1).contiguous()  # [output//2, input]
    return packed.to(torch.int8)


def weight_quant_w4a8_perchannel(
    tensor: torch.Tensor, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel symmetric W4A8 weight quantization with int4 packing.

    Returns (packed_weight[int8], weight_scale[fp32], weight_offset[fp32 zeros],
    scale_bias[fp32]).

    Aligns with msmodelslim WeightQuantizer.find_params(bits=4, sym=True) +
    ascendv1.py:on_w4a8_dynamic save format.
    """
    x = tensor.to(device)
    if x.dim() > 2:
        x = x.flatten(1)

    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=QUANT_EPSILON)
    scale = (xmax / W4_Q_MAX).unsqueeze(-1)  # (M, 1)
    zero = torch.zeros_like(scale)

    quant_weight = torch.clamp(torch.round(x / scale), W4_Q_MIN, W4_Q_MAX).to(torch.int8)
    packed_weight = w4a8_pack_int4(quant_weight)

    # scale_bias: (weight * scale).sum(dim=1) * W4_SCALE_BIAS_FACTOR
    # Aligns with msmodelslim process_scale (column-parallel formula).
    scale_bias = (quant_weight.to(torch.float32) * scale).sum(dim=1, keepdim=True) * W4_SCALE_BIAS_FACTOR

    return (
        packed_weight.cpu(),
        scale.cpu().to(torch.float32),
        zero.cpu().to(torch.float32),
        scale_bias.cpu().to(torch.float32),
    )


def match_yaml_rules(weight_name: str, yaml_rules: list) -> dict | None:
    """Check if weight_name matches any linear_quant rule in YAML config.

    A rule matches if weight_name matches any include pattern AND does not
    match any exclude pattern. Only `linear_quant` type rules are considered.

    Returns the matched rule dict (including qconfig) or None.
    """
    for rule in yaml_rules:
        if rule.get("type") != "linear_quant":
            continue
        includes = rule.get("include", [])
        excludes = rule.get("exclude", [])
        if any(fnmatch.fnmatch(weight_name, pat) for pat in includes):
            if not any(fnmatch.fnmatch(weight_name, pat) for pat in excludes):
                return rule
    return None


class BufferedSafetensorsWriter:
    """Buffered safetensors writer with 4GB shard flushing.

    Simplified from msmodelslim BufferedSafetensorsWriter.
    Accumulates tensors in memory; flushes to a new shard when exceeding
    OUTPUT_SHARD_GB. On close, renames shards to {N:05d}-of-{M:05d} format
    and writes the index.json.
    """

    def __init__(
        self,
        save_directory: str,
        save_prefix: str = "quant_model_weights",
        resume_state: dict | None = None,
    ):
        self.save_directory = save_directory
        self.save_prefix = save_prefix
        self.max_size = OUTPUT_SHARD_GB * ONE_GB_BYTES
        self.wait_save_keys: dict[str, torch.Tensor] = {}
        self.saved_keys_map: dict[str, str] = {}
        self.quant_description: dict[str, str] = {
            "version": "1.0.0",
            "model_quant_type": QUANT_TYPE_W8A8_DYNAMIC,
            "group_size": 0,
            "metadata": {},
        }
        self.total_size = 0
        self._wait_save_size = 0
        self._save_count = 0

        # Resume from checkpoint: restore previously flushed state so that
        # close() can rename the correct number of shards and write a complete
        # index.json/quant_model_description.json.
        if resume_state is not None:
            self._save_count = resume_state.get("save_count", 0)
            self.saved_keys_map = resume_state.get("saved_keys_map", {})
            self.quant_description.update(resume_state.get("quant_description", {}))
            self.total_size = resume_state.get("total_size", 0)

    def write(self, name: str, tensor: torch.Tensor, quant_type: str) -> None:
        """Write a tensor and record its quant type in description."""
        tensor = tensor.detach().cpu().contiguous()
        tensor_size = tensor.numel() * tensor.element_size()

        if self._wait_save_size + tensor_size >= self.max_size:
            self._flush()

        self.wait_save_keys[name] = tensor
        self.quant_description[name] = quant_type
        self.total_size += tensor_size
        self._wait_save_size += tensor_size

    def _flush(self) -> None:
        if not self.wait_save_keys:
            return
        self._save_count += 1
        file_name = f"{self.save_prefix}-{self._save_count:05d}-of-00000.safetensors"
        file_path = os.path.join(self.save_directory, file_name)
        save_file(self.wait_save_keys, file_path, metadata={"format": "pt"})
        self.saved_keys_map.update({k: file_name for k in self.wait_save_keys})
        self.wait_save_keys.clear()
        self._wait_save_size = 0

    def close(self) -> None:
        """Flush remaining, rename shards, write index.json and description."""
        self._flush()

        # Rename shards: -of-00000 -> -of-{M:05d}
        # Idempotent: if a previous close() already renamed some/all shards
        # (crash between close() and checkpoint deletion), skip the rename
        # for those shards instead of raising FileNotFoundError.
        for i in range(self._save_count):
            shard_idx = i + 1
            src_name = f"{self.save_prefix}-{shard_idx:05d}-of-00000.safetensors"
            src = os.path.join(self.save_directory, src_name)
            dst_name = f"{self.save_prefix}-{shard_idx:05d}-of-{self._save_count:05d}.safetensors"
            dst = os.path.join(self.save_directory, dst_name)
            if os.path.exists(dst):
                # Already renamed by a previous close() call
                pass
            elif os.path.exists(src):
                shutil.move(src, dst)
            else:
                raise FileNotFoundError(
                    f"Shard {shard_idx} not found as either {src_name} or {dst_name} "
                    f"in {self.save_directory}. Output may be corrupted."
                )
            # Update weight_map: replace src_name with dst_name for keys in this shard
            for key in self.saved_keys_map:
                if self.saved_keys_map[key] == src_name:
                    self.saved_keys_map[key] = dst_name

        # Write index.json
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.saved_keys_map,
        }
        index_path = os.path.join(self.save_directory, f"{self.save_prefix}.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        # Write quant_model_description.json
        desc_path = os.path.join(self.save_directory, "quant_model_description.json")
        with open(desc_path, "w") as f:
            json.dump(self.quant_description, f, indent=2)

    def get_state(self) -> dict:
        """Return a serializable snapshot of flushed state for checkpointing.

        Only includes already-flushed state (saved_keys_map, quant_description,
        total_size, _save_count). The in-memory buffer (wait_save_keys) is
        excluded — caller must _flush() before calling get_state().
        """
        return {
            "save_count": self._save_count,
            "saved_keys_map": dict(self.saved_keys_map),
            "quant_description": dict(self.quant_description),
            "total_size": self.total_size,
        }


def load_yaml_rules(config_path: str) -> list:
    """Load linear_quant rules from YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("spec", {}).get("process", [])


def main(input_fp8_hf_path: str, output_hf_path: str, config_path: str, device: str = "cpu") -> None:
    """Convert FP8 weights to W8A8 quantized weights in ModelSlim format.

    Args:
        input_fp8_hf_path: Path to HF directory containing FP8 safetensors.
        output_hf_path: Path to output directory for quantized weights.
        config_path: Path to YAML quantization config.
        device: "cpu" or "npu" — where to run W8A8 quantization compute.
    """
    if device == "npu":
        assert HAS_TORCH_NPU, "torch_npu not installed; cannot use NPU. Use --device cpu."
        assert torch.npu.is_available(), "NPU not available. Use --device cpu."

    os.makedirs(output_hf_path, exist_ok=True)

    # Read model index and config
    with open(os.path.join(input_fp8_hf_path, "model.safetensors.index.json")) as f:
        model_index = json.load(f)
    with open(os.path.join(input_fp8_hf_path, "config.json")) as f:
        config = json.load(f)

    # Detect DSpark variant: dspark_block_size field is present in DSpark config.json
    is_dspark = "dspark_block_size" in config

    weight_map = model_index["weight_map"]
    yaml_rules = load_yaml_rules(config_path)

    # Remove quantization_config so vllm-ascend auto-detects via override_quantization_method
    config.pop("quantization_config", None)

    # --- Checkpoint / resume logic ---
    checkpoint_path = os.path.join(output_hf_path, CHECKPOINT_FILENAME)
    resume_state = None
    completed_shards: set[str] = set()

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        resume_state = ckpt.get("writer_state")
        completed_shards = set(ckpt.get("completed_shards", []))
        print(f"Resuming from checkpoint: {len(completed_shards)} shards already done")

        # Crash recovery: delete half-finished temporary shard files
        # (-of-00000.safetensors) left by a crash between _flush() and
        # checkpoint write. Only delete orphans NOT referenced by
        # saved_keys_map — the referenced ones are valid checkpoint state.
        saved_files = set(resume_state.get("saved_keys_map", {}).values()) if resume_state else set()
        for stale in glob(os.path.join(output_hf_path, "*-of-00000.safetensors")):
            if os.path.basename(stale) not in saved_files:
                os.remove(stale)
                print(f"  Removed stale shard: {os.path.basename(stale)}")

    writer = BufferedSafetensorsWriter(output_hf_path, resume_state=resume_state)

    # Cache for loaded safetensors files (keep last 2 to limit memory)
    loaded_files: dict[str, dict[str, torch.Tensor]] = {}

    def get_tensor(name: str) -> torch.Tensor:
        file_name = weight_map[name]
        if file_name not in loaded_files:
            loaded_files[file_name] = load_file(os.path.join(input_fp8_hf_path, file_name), device="cpu")
        return loaded_files[file_name][name]

    safetensor_files = sorted(glob(os.path.join(input_fp8_hf_path, "*.safetensors")))

    for safetensor_file in tqdm(safetensor_files, desc="Quantizing shards"):
        file_name = os.path.basename(safetensor_file)

        # Skip already-completed shards when resuming
        if file_name in completed_shards:
            continue

        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        for weight_name, weight in current_state_dict.items():
            # Skip FP8 scale files - consumed during dequantization or passed
            # through for draft wo_a (DSpark) below.
            if weight_name.endswith(".scale"):
                continue

            # DSpark draft model wo_a: keep FP8 + .scale as-is.
            # vllm-ascend dspark load_weights has a dedicated FP8->BF16
            # dequant path for draft wo_a (detects .self_attn.wo_a.scale).
            # If we dequantize/quantize it here, that path breaks. Main model
            # wo_a (layers.N.*) is NOT affected — only mtp.N.* prefix matches.
            if (
                is_dspark
                and weight_name.startswith("mtp.")
                and weight_name.endswith(".attn.wo_a.weight")
                and weight.element_size() == 1
            ):
                # Output wo_a.weight as-is (FP8) + wo_a.scale as-is
                writer.write(weight_name, weight, QUANT_TYPE_FLOAT)
                scale_name = weight_name.replace(".weight", ".scale")
                try:
                    wo_a_scale = get_tensor(scale_name)
                    writer.write(scale_name, wo_a_scale, QUANT_TYPE_FLOAT)
                except KeyError:
                    print(f"Warning: Missing scale tensor for draft wo_a {weight_name}")
                continue

            # Dequantize FP8 weights to BF16 first
            if weight.element_size() == 1:
                scale_name = weight_name.replace(".weight", ".scale")
                try:
                    scale_inv = get_tensor(scale_name)
                    if weight.dtype == torch.float8_e4m3fn:
                        weight = decode_fp8(weight, scale_inv)
                    elif weight.dtype in (torch.int8, torch.uint8):
                        # MXFP4 packed as int8/uint8
                        weight = decode_fp4(weight, scale_inv)
                    else:
                        raise ValueError(
                            f"Unexpected 1-byte dtype {weight.dtype} for {weight_name}; "
                            "expected float8_e4m3fn (FP8) or int8/uint8 (MXFP4 packed)."
                        )
                except KeyError:
                    print(f"Warning: Missing scale tensor for {weight_name}, keeping original")
                    writer.write(weight_name, weight, QUANT_TYPE_FLOAT)
                    continue

            # Determine quantization: YAML match + 2D tensor (Linear weights)
            base_name = weight_name.rsplit(".", 1)[0] if weight_name.endswith(".weight") else None
            matched_rule = (
                match_yaml_rules(base_name, yaml_rules) if base_name is not None and weight.dim() == 2 else None
            )

            if matched_rule is not None:
                weight_dtype = matched_rule.get("qconfig", {}).get("weight", {}).get("dtype", "int8")
                if weight_dtype == "int4":
                    # W4A8: int4 weight + int8 activation, packed int4 in int8
                    quant_weight, weight_scale, weight_offset, scale_bias = weight_quant_w4a8_perchannel(weight, device)
                    writer.write(weight_name, quant_weight, QUANT_TYPE_W4A8_DYNAMIC)
                    writer.write(
                        weight_name.replace(".weight", ".weight_scale"),
                        weight_scale,
                        QUANT_TYPE_W4A8_DYNAMIC,
                    )
                    writer.write(
                        weight_name.replace(".weight", ".weight_offset"),
                        weight_offset,
                        QUANT_TYPE_W4A8_DYNAMIC,
                    )
                    writer.write(
                        weight_name.replace(".weight", ".scale_bias"),
                        scale_bias,
                        QUANT_TYPE_W4A8_DYNAMIC,
                    )
                else:
                    # W8A8: int8 weight + int8 activation (default path)
                    quant_weight, weight_scale, weight_offset = weight_quant_sym_perchannel(weight, device)
                    writer.write(weight_name, quant_weight, QUANT_TYPE_W8A8_DYNAMIC)
                    writer.write(
                        weight_name.replace(".weight", ".weight_scale"),
                        weight_scale,
                        QUANT_TYPE_W8A8_DYNAMIC,
                    )
                    writer.write(
                        weight_name.replace(".weight", ".weight_offset"),
                        weight_offset,
                        QUANT_TYPE_W8A8_DYNAMIC,
                    )
            else:
                writer.write(weight_name, weight, QUANT_TYPE_FLOAT)

            # MTP shared weights: vLLM loads MTP as a separate model instance that
            # only accepts weights with "mtp.0." prefix. The embedding and head
            # are shared with the main model in training (tie_weight), but at
            # deployment each model instance needs its own copy in the safetensors
            # file. FP8 path auto-renames these in vLLM; W8A8 path does not, so
            # we duplicate them here (matches msmodelslim warp_mtp_model behavior).
            # clone() is required because safetensors rejects tensors sharing the
            # same storage in the same shard.
            #
            # DSpark: load_weights explicitly skips embed/head (line 929 of
            # deepseek_v4_dspark.py), so no copy is needed.
            if not is_dspark:
                if weight_name == "embed.weight":
                    writer.write("mtp.0.emb.tok_emb.weight", weight.clone(), QUANT_TYPE_FLOAT)
                elif weight_name == "head.weight":
                    writer.write("mtp.0.head.weight", weight.clone(), QUANT_TYPE_FLOAT)

        # Memory management: keep only the most recently used shards.
        # Use while (not if) because get_tensor() may pull in multiple scale
        # shards beyond the current one, so we evict until we're back at the limit.
        while len(loaded_files) > MAX_CACHED_SHARDS:
            oldest = next(iter(loaded_files))
            del loaded_files[oldest]

        # Checkpoint: force flush + write checkpoint after each input shard.
        # This ensures a crash only loses the current in-progress shard.
        writer._flush()
        completed_shards.add(file_name)
        ckpt = {
            "completed_shards": sorted(completed_shards),
            "writer_state": writer.get_state(),
        }
        # Atomic write: temp file + rename to avoid partial JSON on crash
        tmp_path = checkpoint_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(ckpt, f)
        os.replace(tmp_path, checkpoint_path)

    writer.close()

    # Checkpoint no longer needed — all shards processed and close() succeeded
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Post-quantization sanity checks (design.md risk 1: YAML glob matching)
    # Only count weight entries (exclude top-level metadata like model_quant_type)
    w8a8_weight_suffixes = (".weight", ".weight_scale", ".weight_offset")
    w8a8_count = sum(
        1
        for k, v in writer.quant_description.items()
        if v == QUANT_TYPE_W8A8_DYNAMIC and k.endswith(w8a8_weight_suffixes)
    )
    assert w8a8_count > 0, "No weights were quantized — check YAML config include patterns"
    assert w8a8_count % 3 == 0, (
        f"W8A8_DYNAMIC weight entry count ({w8a8_count}) must be a multiple of 3 "
        "(weight + weight_scale + weight_offset per quantized layer)"
    )

    # W4A8 count check: each W4A8 layer produces 4 tensors
    # (weight + weight_scale + weight_offset + scale_bias)
    w4a8_weight_suffixes = (".weight", ".weight_scale", ".weight_offset", ".scale_bias")
    w4a8_count = sum(
        1
        for k, v in writer.quant_description.items()
        if v == QUANT_TYPE_W4A8_DYNAMIC and k.endswith(w4a8_weight_suffixes)
    )
    if w4a8_count > 0:
        assert w4a8_count % 4 == 0, (
            f"W4A8_DYNAMIC weight entry count ({w4a8_count}) must be a multiple of 4 "
            "(weight + weight_scale + weight_offset + scale_bias per quantized layer)"
        )

    # DSpark-specific check: draft wo_a must be preserved as FP8+scale.
    # DSpark has 3 MTP layers, each with one wo_a → 3 wo_a.weight + 3 wo_a.scale.
    if is_dspark:
        draft_wo_a_weights = [
            k for k in writer.quant_description if k.startswith("mtp.") and k.endswith(".attn.wo_a.weight")
        ]
        assert len(draft_wo_a_weights) == DSPARK_MTP_LAYERS, (
            f"DSpark draft wo_a count ({len(draft_wo_a_weights)}) must be {DSPARK_MTP_LAYERS} "
            "(one per MTP layer). Check that mtp.N.attn.wo_a weights exist in input."
        )
        draft_wo_a_scales = [
            k for k in writer.quant_description if k.startswith("mtp.") and k.endswith(".attn.wo_a.scale")
        ]
        assert len(draft_wo_a_scales) == DSPARK_MTP_LAYERS, (
            f"DSpark draft wo_a scale count ({len(draft_wo_a_scales)}) must be {DSPARK_MTP_LAYERS}. "
            "Check that mtp.N.attn.wo_a.scale files exist in input."
        )

    # Write config.json (without quantization_config)
    with open(os.path.join(output_hf_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy non-safetensors files (tokenizer, modeling code, etc.)
    # Skip config.json (already written above) and model.safetensors.index.json
    # (stale — references input FP8 shard names that don't exist in the output;
    # vllm's filter_duplicate_safetensors_files would use it to filter out all
    # quant_model_weights-*.safetensors, causing "Cannot find any model weights").
    # The script writes quant_model_weights.safetensors.index.json via writer.close(),
    # and vllm falls back to globbing *.safetensors when model.safetensors.index.json
    # is absent.
    skip_files = {"config.json", "model.safetensors.index.json"}
    copy_extensions = (".py", ".json", ".jinja")
    for root, _, files in os.walk(input_fp8_hf_path):
        for file in files:
            if file in skip_files:
                continue
            # .gitattributes is a hidden file with no extension, copy by name
            if file.endswith(copy_extensions) or file == ".gitattributes":
                src = os.path.join(root, file)
                rel_dir = os.path.relpath(root, input_fp8_hf_path)
                dst_dir = os.path.join(output_hf_path, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, file))

    print(f"\nQuantization complete. Output saved to: {output_hf_path}")
    print(f"  - {writer._save_count} safetensors shards")
    print(f"  - quant_model_description.json ({len(writer.quant_description)} entries)")
    print("  - config.json (quantization_config removed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize DeepSeek-V4 FP8 weights (Flash/DSpark) to W8A8 for vllm-ascend."
    )
    parser.add_argument(
        "--input_fp8_hf_path",
        type=str,
        required=True,
        help="Path to HF directory containing DeepSeek-V4 FP8 weights (Flash, Flash-DSpark, or Pro-DSpark).",
    )
    parser.add_argument(
        "--output_hf_path",
        type=str,
        required=True,
        help="Path to output directory for W8A8 quantized weights.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML quantization config. Auto-selected based on model "
        "variant: quantize_config_dspark.yaml for DSpark, quantize_config.yaml "
        "otherwise.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "npu"],
        default="auto",
        help="Device for W8A8 quantization compute. 'auto' uses NPU if "
        "torch_npu is installed and NPU is available, otherwise CPU. "
        "CPU-only mode works everywhere and is only ~1 min slower for "
        "Flash-scale models (quantization compute is not the bottleneck; "
        "FP8 dequant + disk I/O dominate).",
    )
    args = parser.parse_args()

    # Resolve 'auto' to concrete device
    if args.device == "auto":
        if HAS_TORCH_NPU and torch.npu.is_available():
            args.device = "npu"
        else:
            args.device = "cpu"

    if args.config is None:
        # Auto-select YAML based on DSpark detection
        with open(os.path.join(args.input_fp8_hf_path, "config.json")) as f:
            hf_config = json.load(f)
        if "dspark_block_size" in hf_config:
            args.config = os.path.join(os.path.dirname(__file__), "quantize_config_dspark.yaml")
        else:
            args.config = os.path.join(os.path.dirname(__file__), "quantize_config.yaml")

    main(args.input_fp8_hf_path, args.output_hf_path, args.config, args.device)
