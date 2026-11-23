# Pipeline activation send buffer

Set `--additional-config '{"pp_send_buffer_size":2}'` to enable bounded activation snapshots with Model Runner V2 and a fixed pipeline partition. The default is zero.

Each non-last pipeline worker reserves independent output storage for each slot before KV memory budgeting. After forward it copies intermediate tensors into a slot and posts sends on a communication stream ordered after the copy. Later forwards can reuse model output storage. A slot remains alive until its sends complete. A full ring waits for its oldest send.

This feature retains the fixed layer partition. Chunked prefill and speculative decoding retain their existing model and platform requirements. Output length and graph execution follow the existing serving configuration. Their combinations require device end-to-end validation.

Slots are sized for `max_num_batched_tokens`. Additional slots reduce available KV memory and add device copies. The underlying worker's external-launcher restrictions still apply. Ascend sequence/context parallel configurations are rejected until their communication layout is adapted.

Typed PP transport factories expose a `for_send` factory for output reservation. It includes Top-K indices required at the outgoing boundary and auxiliary states produced within the stage. Receive templates remain based on the incoming boundary; a stage's send and receive tensor sets can differ.

The buffer implementation is provided entirely by vLLM Ascend and uses the standard vLLM pipeline communication interfaces. Compare buffer sizes zero and two using identical weights, request packing, graph and speculative settings. This isolates buffering from layer-boundary movement. No device performance result is claimed by this extraction.
