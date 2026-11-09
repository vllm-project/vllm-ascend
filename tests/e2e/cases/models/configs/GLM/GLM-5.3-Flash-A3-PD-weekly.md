# GLM-5.3 Flash A3 PD weekly coverage

Recipes follow section 5.3 of [the Flash tutorial](../../../../../../docs/source/tutorials/models/GLM5.3-Flash.md)
at main commit `ca99ce8fe12fdd5aeaf9f2559517068f06efe285`.

- `/weekly GLM-5.3-Flash-W8A8-A3-PD-ReplaySSM`
- `/weekly GLM-5.3-Flash-W8A8-A3-PD-Decode-Replay`

A3 uses two separate deployments because the tutorial requires disabling
ReplaySSM for prefix caching. The ReplaySSM case preserves that serving mode
with caching disabled and measures prefill, PD and GPQA. The Decode replay
case disables ReplaySSM, enables producer caching and disables consumer caching.

Performance commissioning profiles use existing GSM8K text datasets: 16K has
64 requests at concurrency 16; 128K has 16 requests at concurrency 4.
These dataset labels are not measured Flash token lengths. Verify actual
input token counts on hardware. Prefill generates one token; PD and Decode
replay generate 1024. Request rate is unlimited. No unmeasured throughput
baseline or TPOT threshold is installed.

Cache-enabled prefill and PD measurements reset producer caches after warmup.
Decode resets producer caches, warms all measured prompts, then repeats the
same workload without an intervening reset. This measures request-level
Decode-dominated replay, not a device-only Decode microbenchmark. Check cache
hit counters and capacity before interpreting results; routing across A3's
two producer ranks and eviction can prevent full hits. AISBench also has its
own internal warmup request.

GPQA uses temperature 1, top-p 0.95 and maximum output 131072. The provisional
accuracy floor is five percentage points below the tutorial reference:
84.9 for A3 (reference 89.9, measured with an older deployment).
This is a commissioning floor, not a measured baseline for these new cases.
Calibrate performance and accuracy after successful hardware runs. This
coverage does not exercise multimodal inputs or one-million-token requests.

Serving adaptations include CI-assigned addresses and device slices, explicit
Mooncake topology metadata, and omission of the tutorial's `glm` served alias
so AISBench can address the configured model. No static-kernel switch is added.
The existing nightly and weekly recipes remain registered unchanged.

CPU checks validate rank allocation, command rendering and replay boundaries.
NPU startup, transfer correctness, cache behavior and benchmark results still
require validation before merge.
