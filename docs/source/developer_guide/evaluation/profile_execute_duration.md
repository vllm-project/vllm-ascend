# Profile Execute Duration

The execution duration of each stage (including pre/post-processing, model forward, etc.) usually needs to be captured during a complete inference process. Typically, this is done by using `torch.npu.synchronize()` and obtaining CPU timestamps, which increases the performance overhead of host/device synchronization.

**To reduce the performance overhead, we add this feature, using the NPU event timestamp mechanism to observe the device execution time asynchronously.**

## Usage
* Use the environment variable `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE` to enable this feature.
* Use the non-blocking API `ProfileExecuteDuration().capture_async` to set observation points asynchronously when you need to observe the execution duration.
* Use the blocking API `ProfileExecuteDuration().pop_captured_sync` at an appropriate time to get and print the execution durations of all observed stages.

**We have instrumented the key inference stages (including pre-processing, model forward pass, etc.) for execute duration profiling. Execute the script as follows:**
```
VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1 python3 vllm-ascend/examples/offline_inference_npu.py
```

## Example Output

```
5691:(IntegratedWorker pid=1502285) Profile execute duration [Decode]: [post process]:14.17ms [prepare input and forward]:9.57ms [forward]:4.14ms
5695:(IntegratedWorker pid=1502285) Profile execute duration [Decode]: [post process]:14.29ms [prepare input and forward]:10.19ms [forward]:4.14ms
5697:(IntegratedWorker pid=1502343) Profile execute duration [Decode]: [post process]:14.81ms [prepare input and forward]:10.29ms [forward]:3.99ms
5701:(IntegratedWorker pid=1502343) Profile execute duration [Decode]: [post process]:14.10ms [prepare input and forward]:10.62ms [forward]:4.33ms
5705:(IntegratedWorker pid=1502343) Profile execute duration [Decode]: [post process]:14.65ms [prepare input and forward]:9.58ms [forward]:4.20ms
5709:(IntegratedWorker pid=1502343) Profile execute duration [Decode]: [post process]:14.43ms [prepare input and forward]:9.88ms [forward]:4.20ms
5711:(IntegratedWorker pid=1502401) Profile execute duration [Decode]: [post process]:14.89ms [prepare input and forward]:10.49ms [forward]:4.19ms
5715:(IntegratedWorker pid=1502401) Profile execute duration [Decode]: [post process]:14.14ms [prepare input and forward]:11.21ms [forward]:4.18ms
5719:(IntegratedWorker pid=1502401) Profile execute duration [Decode]: [post process]:14.71ms [prepare input and forward]:10.15ms [forward]:4.42ms
5723:(IntegratedWorker pid=1502401) Profile execute duration [Decode]: [post process]:14.62ms [prepare input and forward]:10.31ms [forward]:4.25ms
5725:(IntegratedWorker pid=1502462) Profile execute duration [Decode]: [post process]:14.12ms [prepare input and forward]:10.33ms [forward]:4.24ms
5729:(IntegratedWorker pid=1502462) Profile execute duration [Decode]: [post process]:14.58ms [prepare input and forward]:10.85ms [forward]:4.32ms
5733:(IntegratedWorker pid=1502462) Profile execute duration [Decode]: [post process]:14.32ms [prepare input and forward]:9.79ms [forward]:4.28ms
5737:(IntegratedWorker pid=1502462) Profile execute duration [Decode]: [post process]:15.06ms [prepare input and forward]:9.89ms [forward]:4.32ms
5739:(IntegratedWorker pid=1502524) Profile execute duration [Decode]: [post process]:14.62ms [prepare input and forward]:10.48ms [forward]:4.27ms
5743:(IntegratedWorker pid=1502524) Profile execute duration [Decode]: [post process]:14.60ms [prepare input and forward]:10.71ms [forward]:4.61ms
5747:(IntegratedWorker pid=1502524) Profile execute duration [Decode]: [post process]:14.21ms [prepare input and forward]:10.10ms [forward]:4.52ms
5751:(IntegratedWorker pid=1502524) Profile execute duration [Decode]: [post process]:15.03ms [prepare input and forward]:10.00ms [forward]:4.42ms

```