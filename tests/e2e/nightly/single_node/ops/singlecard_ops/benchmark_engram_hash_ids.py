# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check and benchmark Triton against the current Ascend C engram_hash_ids.

Build/install vllm-ascend with EngramHashIds and run this file from the
repository root. No model, weights or distributed initialization is needed.
Both paths share input tensors and the production request-index kernel.
Defaults: slot_cache, eager, request_and_hash, B=1/4/16/32 and S=1/4/6/8192/16384.
--quick reduces repetitions only; --reference-only needs just the stdlib.
On A5, this standalone benchmark loads the extension directly because the
framework's runtime custom-op capability is disabled. Global policy is unchanged.
Results are printed to stdout; no result files are written.
Event intervals include host dispatch gaps and use warm-cache averages.
Both paths write into preallocated outputs; timings exclude output allocation.
"""

import argparse
import copy
import importlib
import math
import random
import statistics
import sys
from pathlib import Path

# ------------------------ Edit configuration here -------------------------
DEVICE = "npu:0"
BATCH_SIZES = (1, 4, 16, 32)  # Timed cases only; untimed correctness checks remain separate.
TOKENS_PER_REQUEST = (1, 4, 6, 8192, 16384)
HISTORY_PATHS = ("slot_cache",)  # Optional: add "lookback".
SCOPE = "request_and_hash"  # Optional: "hash_only".
WARMUP_CALLS = 30
TRIALS = 30
CALLS_PER_SAMPLE = 100
SEED = 2026
NUM_LAYERS = 2
MAX_NGRAM = 4
NUM_HEADS = 8
HASH_COLUMNS = (MAX_NGRAM - 1) * NUM_HEADS
LOOKBACK_DEPTH = MAX_NGRAM - 1
CORRECTNESS_COUNTS = (1, 6, 33, 129)
VOCAB_SIZE = 129280
COMPRESSED_VOCAB_SIZE = 99092
HEAD_TABLE_ROWS = 16000000
HISTORY_TOKENS = 128
SLOT_CACHE_BLOCK_SIZE = 128
PAD_ID = 2
DEAD_ID = -1
BLOCK_T = 32  # Production Triton tile.
NUM_WARPS = 4
IMPLEMENTATIONS = ("triton", "ascendc")
# -------------------------------------------------------------------------


def positive_ints(value, name):
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value,)
    try:
        value = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer or sequence of integers") from exc
    if not value or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in value):
        raise ValueError(f"{name} must contain positive integers")
    return value


def make_hash_parameters():
    """Preserve the reference benchmark RNG order and hash parameters."""

    def is_prime(value):
        return value > 1 and all(value % divisor for divisor in range(2, math.isqrt(value) + 1))

    rng = random.Random(SEED)
    token_map = [index % COMPRESSED_VOCAB_SIZE for index in range(VOCAB_SIZE)]
    rng.shuffle(token_map)
    multipliers = [[rng.randrange(1 << 30, 1 << 40) | 1 for _ in range(MAX_NGRAM)] for _ in range(NUM_LAYERS)]
    primes, offsets = [], []
    candidate = HEAD_TABLE_ROWS + 1
    for _ in range(NUM_LAYERS):
        layer_primes, layer_offsets, offset = [], [], 0
        for _ in range(MAX_NGRAM - 1):
            order_primes, order_offsets = [], []
            for _ in range(NUM_HEADS):
                while not is_prime(candidate):
                    candidate += 1
                order_primes.append(candidate)
                order_offsets.append(offset)
                offset += candidate
                candidate += 1
            layer_primes.append(order_primes)
            layer_offsets.append(order_offsets)
        primes.append(layer_primes)
        offsets.append(layer_offsets)
    return dict(token_map=token_map, multipliers=multipliers, primes=primes, offsets=offsets)


def make_inputs(batch, count, *, boundary=False):
    rng = random.Random(SEED + batch * 17 + count)
    starts = [index % MAX_NGRAM if boundary else HISTORY_TOKENS for index in range(batch)]
    lookback = [[rng.randrange(VOCAB_SIZE) if start > col else -1 for col in range(LOOKBACK_DEPTH)] for start in starts]
    return dict(
        input_ids=[rng.randrange(VOCAB_SIZE) for _ in range(batch * count)],
        positions=[start + step for start in starts for step in range(count)],
        query_start_loc=[index * count for index in range(batch + 1)],
        req_ids=[req for req in range(batch) for _ in range(count)],
        lookback_token_ids=lookback,
        dead_mask=[boundary and index % 5 == 0 for index in range(batch * count)],
        lookback_dead_mask=[
            [boundary and req % 3 == 0 and col == 0 for col in range(LOOKBACK_DEPTH)] for req in range(batch)
        ],
    )


def make_slot_cache(inputs, parameters):
    batch = len(inputs["query_start_loc"]) - 1
    max_blocks = max(inputs["positions"]) // SLOT_CACHE_BLOCK_SIZE + 1
    physical = list(range(batch * max_blocks))
    random.Random(SEED + batch).shuffle(physical)
    table = [physical[req * max_blocks : (req + 1) * max_blocks] for req in range(batch)]
    cache = [PAD_ID] * (batch * max_blocks * SLOT_CACHE_BLOCK_SIZE)
    for req in range(batch):
        start = inputs["positions"][inputs["query_start_loc"][req]]
        for col, raw in enumerate(inputs["lookback_token_ids"][req]):
            position = start - 1 - col
            if position < 0:
                continue
            value = parameters["token_map"][raw] if raw >= 0 else PAD_ID
            if raw >= 0 and inputs["lookback_dead_mask"][req][col]:
                value = DEAD_ID
            block = table[req][position // SLOT_CACHE_BLOCK_SIZE]
            cache[block * SLOT_CACHE_BLOCK_SIZE + position % SLOT_CACHE_BLOCK_SIZE] = value
    return dict(block_table=table, cache=cache)


def history_values(inputs, parameters, slot_data=None):
    """Independent scalar addressing; also supports partial/unknown lookback."""
    history = []
    for token, req in enumerate(inputs["req_ids"]):
        start = inputs["positions"][inputs["query_start_loc"][req]]
        position = inputs["positions"][token]
        blocked, values = False, []
        for shift in range(MAX_NGRAM):
            previous = position - shift
            if previous >= start:
                index = max(token - shift, 0)
                source = parameters["token_map"][inputs["input_ids"][index]]
                if inputs["dead_mask"][index]:
                    source = DEAD_ID
            else:
                col = start - 1 - previous
                window = inputs["lookback_token_ids"][req]
                raw = window[col] if 0 <= col < len(window) else -1
                if raw >= 0:
                    source = parameters["token_map"][raw]
                    if inputs["lookback_dead_mask"][req][col]:
                        source = DEAD_ID
                elif slot_data is not None:
                    table = slot_data["block_table"]
                    clamped = min(max(previous, 0), len(table[0]) * SLOT_CACHE_BLOCK_SIZE - 1)
                    block = table[min(req, len(table) - 1)][clamped // SLOT_CACHE_BLOCK_SIZE]
                    slot = min(
                        max(block * SLOT_CACHE_BLOCK_SIZE + clamped % SLOT_CACHE_BLOCK_SIZE, 0),
                        len(slot_data["cache"]) - 1,
                    )
                    source = slot_data["cache"][slot]
                else:
                    source = PAD_ID
            blocked = blocked or previous < 0 or source == DEAD_ID
            values.append(PAD_ID if blocked else source)
        history.append(values)
    return history


def signed(value, bits):
    return ((value + (1 << (bits - 1))) & ((1 << bits) - 1)) - (1 << (bits - 1))


def scalar_oracle(history, parameters):
    """Wrapping i64 mul/XOR, truncating signed remainder, wrapping i32 output."""
    output = []
    for values in history:
        layers = []
        for layer in range(NUM_LAYERS):
            rolling, row = 0, []
            for shift, value in enumerate(values):
                rolling = signed(rolling ^ signed(value * parameters["multipliers"][layer][shift], 64), 64)
                if shift:
                    for head in range(NUM_HEADS):
                        divisor = abs(parameters["primes"][layer][shift - 1][head])
                        remainder = abs(rolling) % divisor
                        if rolling < 0:
                            remainder = -remainder
                        row.append(signed(remainder + parameters["offsets"][layer][shift - 1][head], 32))
            layers.append(row)
        output.append(layers)
    return output


def tensor_oracle(torch, history, parameters):
    values = torch.tensor(history, dtype=torch.int64)
    output = torch.empty((len(history), NUM_LAYERS, HASH_COLUMNS), dtype=torch.int32)
    for layer in range(NUM_LAYERS):
        rolling = torch.zeros(len(history), dtype=torch.int64)
        for shift in range(MAX_NGRAM):
            rolling = rolling ^ (values[:, shift] * parameters["multipliers"][layer][shift])
            if shift:
                for head in range(NUM_HEADS):
                    prime = parameters["primes"][layer][shift - 1][head]
                    # Avoid host INT64_MIN/-1 overflow in the reference.
                    remainder = torch.zeros_like(rolling) if abs(prime) == 1 else torch.fmod(rolling, prime)
                    output[:, layer, (shift - 1) * NUM_HEADS + head] = (
                        remainder + parameters["offsets"][layer][shift - 1][head]
                    ).to(torch.int32)
    return output


def make_case(torch, kernel, inputs, parameters, history_path):
    cpu = {**inputs, **parameters}
    slots = make_slot_cache(inputs, parameters)
    use_cache = history_path != "lookback"
    cpu.update(slots if use_cache else dict(block_table=[[0]], cache=[PAD_ID]))
    if history_path == "slot_cache":
        cpu["lookback_token_ids"] = [[-1] * LOOKBACK_DEPTH for _ in inputs["lookback_token_ids"]]
        cpu["lookback_dead_mask"] = [[False] * LOOKBACK_DEPTH for _ in inputs["lookback_token_ids"]]
    elif history_path == "mixed":
        # Exercise known window + unknown window fallback in the same request.
        cpu["lookback_token_ids"] = [row[:1] + [-1, -1] for row in inputs["lookback_token_ids"]]
    dtypes = dict(
        dead_mask=torch.bool,
        lookback_dead_mask=torch.bool,
        positions=torch.int64,
        multipliers=torch.int64,
        primes=torch.int64,
        offsets=torch.int64,
    )
    tensors = {
        name: torch.tensor(value, dtype=dtypes.get(name, torch.int32), device=DEVICE) for name, value in cpu.items()
    }
    total = len(inputs["input_ids"])
    outputs = {
        name: torch.empty((total, NUM_LAYERS, HASH_COLUMNS), dtype=torch.int32, device=DEVICE)
        for name in IMPLEMENTATIONS
    }
    t = tensors
    num_slots = t["cache"].numel() if use_cache else 0
    table_rows, max_blocks = t["block_table"].shape if use_cache else (0, 0)
    lookback_depth = t["lookback_token_ids"].shape[1]

    def request_index():
        kernel._engram_req_index_kernel[(math.ceil(total / BLOCK_T),)](
            t["query_start_loc"], t["req_ids"], total, len(inputs["query_start_loc"]) - 1, 1, BLOCK_T=BLOCK_T
        )

    def hash_ids(operation, output, **launch_options):
        operation(
            t["input_ids"],
            t["token_map"],
            t["dead_mask"],
            t["positions"],
            t["block_table"] if use_cache else None,
            t["query_start_loc"],
            t["req_ids"],
            t["multipliers"],
            t["primes"],
            t["offsets"],
            t["cache"] if use_cache else None,
            t["lookback_token_ids"],
            t["lookback_dead_mask"],
            output,
            total,
            num_slots,
            PAD_ID,
            input_stride=1,
            mask_stride=1,
            position_stride=1,
            table_stride=t["block_table"].stride(0) if use_cache else 0,
            table_col_stride=1,
            query_stride=1,
            num_table_rows=table_rows,
            max_blocks=max_blocks,
            cache_block_size=SLOT_CACHE_BLOCK_SIZE,
            MAX_NGRAM=MAX_NGRAM,
            num_heads=NUM_HEADS,
            dead_id=DEAD_ID,
            lookback_depth=lookback_depth,
            lookback_row_stride=t["lookback_token_ids"].stride(0),
            lookback_col_stride=1,
            lookback_mask_row_stride=t["lookback_dead_mask"].stride(0),
            lookback_mask_col_stride=1,
            **launch_options,
        )

    def triton_hash():
        hash_ids(
            kernel._hash_ids_kernel[(math.ceil(total / BLOCK_T), NUM_LAYERS)],
            outputs["triton"],
            BLOCK_T=BLOCK_T,
            BLOCK_H=NUM_HEADS,
            num_warps=NUM_WARPS,
        )

    def ascendc_hash():
        hash_ids(torch.ops._C_ascend.engram_hash_ids, outputs["ascendc"], num_layers=NUM_LAYERS)

    def run(function):
        if SCOPE == "request_and_hash":
            request_index()
        function()

    paths = {
        "triton": lambda: run(triton_hash),
        "ascendc": lambda: run(ascendc_hash),
    }
    history = history_values(cpu, parameters, slots if use_cache else None)
    return paths, tensors, outputs, history


def validate_pad_fallback(torch, kernel):
    """No-cache pad is int32; padding after a dead source retains the full ID."""
    divisor = 7
    for map_dtype in (torch.int32, torch.int64):
        data = (
            torch.tensor([0], dtype=torch.int32, device=DEVICE),
            torch.tensor([0], dtype=map_dtype, device=DEVICE),
            torch.tensor([False], dtype=torch.bool, device=DEVICE),
            torch.tensor([1], dtype=torch.int64, device=DEVICE),
            None,
            torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
            torch.tensor([0], dtype=torch.int32, device=DEVICE),
            torch.tensor([[0, 1]], dtype=torch.int64, device=DEVICE),
            torch.tensor([[[divisor]]], dtype=torch.int64, device=DEVICE),
            torch.tensor([[[0]]], dtype=torch.int64, device=DEVICE),
            None,
            torch.tensor([[-1]], dtype=torch.int32, device=DEVICE),
            torch.tensor([[False]], dtype=torch.bool, device=DEVICE),
        )
        for pad in (PAD_ID, (1 << 32) + 1, -(1 << 63), (1 << 63) - 1):
            options = dict(
                num_tokens=1,
                num_slots=0,
                pad_id=pad,
                input_stride=1,
                mask_stride=1,
                position_stride=1,
                table_stride=0,
                table_col_stride=0,
                query_stride=1,
                num_table_rows=0,
                max_blocks=0,
                cache_block_size=SLOT_CACHE_BLOCK_SIZE,
                MAX_NGRAM=2,
                num_heads=1,
                dead_id=DEAD_ID,
                lookback_depth=1,
                lookback_row_stride=1,
                lookback_col_stride=1,
                lookback_mask_row_stride=1,
                lookback_mask_col_stride=1,
            )
            fallback = signed(pad, 32)
            value = pad if fallback == DEAD_ID else fallback
            remainder = abs(value) % divisor * (-1 if value < 0 else 1)
            expected = torch.tensor([[[signed(remainder, 32)]]], dtype=torch.int32)
            triton_output = torch.empty((1, 1, 1), dtype=torch.int32, device=DEVICE)
            ascendc_output = torch.empty_like(triton_output)
            kernel._hash_ids_kernel[(1, 1)](
                *data, triton_output, **options, BLOCK_T=BLOCK_T, BLOCK_H=1, num_warps=NUM_WARPS
            )
            torch.ops._C_ascend.engram_hash_ids(*data, ascendc_output, **options, num_layers=1)
            torch.npu.synchronize()
            torch.testing.assert_close(triton_output.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(ascendc_output.cpu(), expected, rtol=0, atol=0)


def validate_prefix_buffers(torch, kernel, parameters):
    """Effective sizes must remain independent of allocation capacity."""
    inputs = make_inputs(1, CORRECTNESS_COUNTS[-1])
    paths, tensors, outputs, history = make_case(torch, kernel, inputs, parameters, "slot_cache")
    total = len(inputs["input_ids"])
    sentinel = -1234567
    for name in ("input_ids", "dead_mask", "positions", "req_ids", "cache"):
        tensors[name] = torch.cat((tensors[name], tensors[name].new_zeros((BLOCK_T,))))
    for name in ("multipliers", "primes", "offsets"):
        value = tensors[name]
        tensors[name] = torch.cat((value, torch.zeros_like(value[:1])))
    expected = torch.tensor(scalar_oracle(history, parameters), dtype=torch.int32)
    for name, run in paths.items():
        outputs[name] = torch.full(
            (total + BLOCK_T, NUM_LAYERS, HASH_COLUMNS), sentinel, dtype=torch.int32, device=DEVICE
        )
        run()
        torch.npu.synchronize()
        torch.testing.assert_close(outputs[name][:total].cpu(), expected, rtol=0, atol=0)
        assert (outputs[name][total:] == sentinel).all().item(), f"{name} wrote beyond num_tokens"


def validate(torch, paths, tensors, outputs, expected, req_ids):
    for name, run in paths.items():
        if SCOPE == "request_and_hash":
            tensors["req_ids"].fill_(-1)
        outputs[name].fill_(-1234567)
        run()
        torch.npu.synchronize()
        torch.testing.assert_close(outputs[name].cpu(), expected, rtol=0, atol=0, msg=f"{name} != CPU oracle")
        torch.testing.assert_close(tensors["req_ids"].cpu(), torch.tensor(req_ids, dtype=torch.int32))
    for name in IMPLEMENTATIONS[1:]:
        torch.testing.assert_close(outputs[name], outputs["triton"], rtol=0, atol=0)


def measure(torch, run, warmup, trials, calls):
    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(trials):
        begin.record()
        for _ in range(calls):
            run()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / calls)
    return dict(median_us=statistics.median(samples), min_us=min(samples), max_us=max(samples))


def edge_parameters(base):
    """Signed overflow, negative rolling, wide divisors, int32 truncation."""
    parameters = copy.deepcopy(base)
    parameters["token_map"][:5] = [-(1 << 31), -2, DEAD_ID, (1 << 31) - 1, 0]
    multipliers = [-(1 << 63), (1 << 63) - 1, -1, -0x123456789ABCDEF]
    for layer in range(NUM_LAYERS):
        parameters["multipliers"][layer] = multipliers[layer:] + multipliers[:layer]
        for order in range(LOOKBACK_DEPTH):
            parameters["primes"][layer][order] = [
                1,
                -1,
                257,
                16000057,
                (1 << 30) - 1,
                1 << 30,
                (1 << 63) - 1,
                -(1 << 63),
            ]
            parameters["offsets"][layer][order] = [
                0,
                -1,
                (1 << 31) - 1,
                1 << 31,
                -(1 << 63),
                (1 << 63) - 1,
                -0x123456789ABCDEF,
                0x123456789ABCDEF,
            ]
    return parameters


def avoid_local_triton_shadow():
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "triton" / "__init__.py").is_file():
        sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != script_dir]


def load_custom_ops():
    utils = importlib.import_module("vllm_ascend.utils")
    if not utils.is_950():
        if not utils.enable_custom_op():
            raise RuntimeError("Framework custom-op loading is disabled or failed; check the preceding logs")
        return

    # A5 disables runtime custom ops globally; explicitly load only for this benchmark.
    utils.bootstrap_custom_op_env()
    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    except ImportError as exc:
        if "libcust_opapi.so" not in str(exc):
            raise
        utils.bootstrap_custom_op_env(include_vendor_lib=True)
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    importlib.import_module("vllm_ascend.meta_registration")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="1 warmup, 3 trials, 1 call/sample; all input cases retained"
    )
    args = parser.parse_args()
    batches = positive_ints(BATCH_SIZES, "BATCH_SIZES")
    counts = positive_ints(TOKENS_PER_REQUEST, "TOKENS_PER_REQUEST")
    if MAX_NGRAM != 4 or NUM_HEADS != 8 or NUM_LAYERS not in (1, 2):
        raise ValueError("This DS4.1 benchmark supports N=4,H=8,L=1 or 2")
    if (
        SCOPE not in ("request_and_hash", "hash_only")
        or not HISTORY_PATHS
        or set(HISTORY_PATHS) - {"slot_cache", "lookback"}
    ):
        raise ValueError("Invalid scope/history configuration")
    if min(WARMUP_CALLS, TRIALS, CALLS_PER_SAMPLE, SLOT_CACHE_BLOCK_SIZE) <= 0:
        raise ValueError("Sampling counts and block size must be positive")
    parameters = make_hash_parameters()
    if args.reference_only:
        for boundary in (False, True):
            inputs = make_inputs(3, 6, boundary=boundary)
            a = history_values(inputs, parameters)
            slots = make_slot_cache(inputs, parameters)
            cached = {**inputs, "lookback_token_ids": [[-1] * LOOKBACK_DEPTH for _ in inputs["lookback_token_ids"]]}
            assert history_values(cached, parameters, slots) == a
            for p in (parameters, edge_parameters(parameters)):
                result = scalar_oracle(a, p)
                assert len(result) == 18 and all(len(row) == NUM_LAYERS for row in result)
        print("CPU oracle/history checks passed. AscendC build, device correctness and latency were NOT tested.")
        return
    avoid_local_triton_shadow()
    importlib.import_module("triton.language")
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("An Ascend A3/A5 NPU is required")
    torch.npu.set_device(DEVICE)
    load_custom_ops()
    if not hasattr(torch.ops._C_ascend, "engram_hash_ids"):
        raise RuntimeError("engram_hash_ids is missing: rebuild/install EngramHashIds and the torch extension")
    kernel = importlib.import_module("vllm_ascend.ops.triton.engram_hash")
    samples = (1, 3, 1) if args.quick else (WARMUP_CALLS, TRIALS, CALLS_PER_SAMPLE)
    # Untimed tests: first-token/dead-mask boundaries, ragged requests, tile
    # tails, mixed window/cache sources and signed/wrapping arithmetic.
    for count in CORRECTNESS_COUNTS:
        for boundary in (False, True):
            inputs = make_inputs(3, count, boundary=boundary)
            if count == 33:
                # Keep the total shape but make request lengths heterogeneous.
                inputs["query_start_loc"] = [0, 1, 35, 99]
                inputs["req_ids"] = [0] + [1] * 34 + [2] * 64
                starts = [0, 1, 3] if boundary else [HISTORY_TOKENS] * 3
                inputs["positions"] = [
                    start + pos for start, length in zip(starts, (1, 34, 64)) for pos in range(length)
                ]
            for p in (parameters, edge_parameters(parameters)):
                for history_path in ("lookback", "slot_cache", "mixed"):
                    checked_inputs = inputs
                    if p is not parameters:
                        checked_inputs = {**inputs, "input_ids": [i % 5 for i in range(len(inputs["input_ids"]))]}
                    paths, tensors, outputs, history = make_case(torch, kernel, checked_inputs, p, history_path)
                    expected = torch.tensor(scalar_oracle(history, p), dtype=torch.int32)
                    torch.testing.assert_close(tensor_oracle(torch, history, p), expected, rtol=0, atol=0)
                    validate(torch, paths, tensors, outputs, expected, inputs["req_ids"])
    validate_pad_fallback(torch, kernel)
    validate_prefix_buffers(torch, kernel, parameters)
    print("Exact Triton/AscendC/CPU edge and buffer-prefix checks passed.", flush=True)
    print(
        f"Device={torch.npu.get_device_name()} scope={SCOPE} "
        f"warmup={samples[0]} trials={samples[1]} calls_per_sample={samples[2]}",
        flush=True,
    )
    print("Both paths use preallocated outputs; allocation is excluded from timing.", flush=True)
    for count in counts:
        for batch in batches:
            inputs = make_inputs(batch, count)
            for history_path in HISTORY_PATHS:
                print(f"Preparing B={batch} S={count} T={batch * count} {history_path}", flush=True)
                paths, tensors, outputs, history = make_case(torch, kernel, inputs, parameters, history_path)
                expected = tensor_oracle(torch, history, parameters)
                validate(torch, paths, tensors, outputs, expected, inputs["req_ids"])
                order = list(paths)
                random.Random(SEED + batch + count).shuffle(order)
                measured = {}
                for name in order:
                    measured[name] = measure(torch, paths[name], *samples)
                    torch.testing.assert_close(outputs[name].cpu(), expected, rtol=0, atol=0)
                    timing = measured[name]
                    print(
                        f"B={batch:3d} S={count} {history_path} {name:14s} eager {SCOPE} "
                        f"median={timing['median_us']:.3f} us "
                        f"min={timing['min_us']:.3f} us max={timing['max_us']:.3f} us",
                        flush=True,
                    )
                baseline = measured["triton"]["median_us"]
                candidate = measured["ascendc"]["median_us"]
                print(
                    f"COMPARE B={batch} S={count} {history_path}: "
                    f"triton={baseline:.3f} us ascendc={candidate:.3f} us "
                    f"speedup={baseline / candidate:.3f}x "
                    f"reduction={100 * (baseline - candidate) / baseline:.2f}%",
                    flush=True,
                )
                torch.npu.synchronize()
                del paths, tensors, outputs, history, expected


if __name__ == "__main__":
    main()
