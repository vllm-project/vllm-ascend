#!/usr/bin/env python3
"""Layerwise + PP benchmark: 50 prompts, concurrency 16, input 512, output 128.

Executed on 8x Ascend 910B3 against a PP=2 x TP=4 DeepSeek-V4-Flash-w8a8-mtp
kv_both instance with layerwise KV transfer (memcache host_shm backend,
prefetch_layers=3). See PR #15507 for the full run instructions.
"""
import json, time, asyncio, aiohttp

API_URL = "http://localhost:8006/v1/completions"
MODEL = "deepseek_v4"
NUM_PROMPTS = 50
CONCURRENCY = 16
INPUT_LEN = 512
OUTPUT_LEN = 128


async def send_request(session, idx):
    payload = {"model": MODEL, "prompt": "Hello " * INPUT_LEN,
               "max_tokens": OUTPUT_LEN, "temperature": 0, "stream": False}
    start = time.time()
    try:
        async with session.post(API_URL, json=payload, timeout=600) as resp:
            data = await resp.json()
            return {"idx": idx, "latency": time.time() - start,
                    "output_tokens": data.get("usage", {}).get("completion_tokens", OUTPUT_LEN),
                    "success": True}
    except Exception as e:
        return {"idx": idx, "error": str(e), "success": False, "latency": time.time() - start}


async def main():
    print(f"=== layerwise PP2xTP4: {NUM_PROMPTS} prompts, cc={CONCURRENCY} ===")
    print("Warmup...")
    async with aiohttp.ClientSession() as s:
        await send_request(s, -1)
    print("Benchmark...")
    t0 = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(s, i):
        async with sem:
            return await send_request(s, i)

    async with aiohttp.ClientSession() as s:
        results = await asyncio.gather(*[bounded(s, i) for i in range(NUM_PROMPTS)])
    total = time.time() - t0
    ok = [r for r in results if r.get("success")]
    toks = sum(r.get("output_tokens", 0) for r in ok)
    lat = sorted(r["latency"] for r in ok)
    print(f"\n=== Results ===")
    print(f"Total: {total:.2f}s | Success: {len(ok)}/{NUM_PROMPTS}")
    print(f"Request tput: {len(ok)/total:.2f} req/s")
    print(f"Output tok tput: {toks/total:.2f} tokens/s")
    if lat:
        print(f"Avg: {sum(lat)/len(lat):.2f}s | P50: {lat[len(lat)//2]:.2f}s | P99: {lat[int(len(lat)*0.99)]:.2f}s")
    with open("lwpp_bench_result.json", "w") as f:
        json.dump({"total_time": total, "req_tput": len(ok)/total,
                   "tok_tput": toks/total, "latencies": lat}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
