# Pipeline communication buffers

Set `--additional-config '{"pp_send_buffer_size":2}'` to enable bounded activation snapshots with Model Runner V2 and a fixed pipeline partition. The default is zero.

Each non-last pipeline worker reserves independent output storage for each slot before KV memory budgeting. After forward it copies intermediate tensors into a slot and posts sends on a communication stream ordered after the copy. Later forwards can reuse model output storage. A slot remains alive until its sends complete. A full ring waits for its oldest send.

This feature retains the fixed layer partition. Chunked prefill and speculative decoding retain their existing model and platform requirements. Output length and graph execution follow the existing serving configuration. Their combinations require device end-to-end validation.

Slots are sized for `max_num_batched_tokens`. Additional slots reduce available KV memory and add device copies. The underlying worker's external-launcher restrictions still apply. Ascend sequence/context parallel configurations are rejected until their communication layout is adapted.

Typed PP transport factories expose a `for_send` factory for output reservation. It includes Top-K indices required at the outgoing boundary and auxiliary states produced within the stage. Receive templates remain based on the incoming boundary; a stage's send and receive tensor sets can differ.

The buffer implementation is provided entirely by vLLM Ascend and uses the standard vLLM pipeline communication interfaces. Compare buffer sizes zero and two using identical weights, request packing, graph and speculative settings. This isolates buffering from layer-boundary movement. No device performance result is claimed by this extraction.

## Independent receive submission

Set `--additional-config '{"pp_send_buffer_size":5,"pp_recv_buffer_size":2}'`
to submit PP receives on a separate NPU stream with Model Runner V2. The receive
size defaults to zero (disabled) and can also be used without the send buffer. It applies
only to non-first stages and does not add a thread or change RPC scheduling.

Receive tensor allocation and `irecv_tensor_dict` execute on the receive submission
stream. This prevents the HCCL backend from inheriting the preceding compute stream
tail. Before consuming the input, the compute stream waits on the receive handles;
receive storage is recorded on the consumer stream before any all-gather or input
copy. Sleep and shutdown drain outstanding transport before releasing storage.

`PPRecvBuffer` and `PPSendBuffer` share `worker/pp/buffer.py`. Receive storage is
allocated dynamically from incoming metadata, without a staging copy or startup
reservation. The receive size bounds retained batches, not bytes. After forward,
an event fences the consumer's reads; completed batches are reaped. At capacity,
the oldest event is synchronized before another receive is allocated. This
conservatively retains input through forward completion. Send and receive sizes
are independent; an unconsumed input is drained without running all-gather.

Dynamic allocation can increase transient memory usage and depends on
RPC availability, metadata arrival, and device communication resources. Enabling
the option does not guarantee overlap or a performance improvement. Validate each
model, graph mode, and parallel layout on the target device.
