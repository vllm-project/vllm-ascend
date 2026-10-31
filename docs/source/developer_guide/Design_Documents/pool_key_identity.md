# Source identity for layerwise pool keys

`PoolKeyIdentity` is a shared interface used by
`LayerwisePoolConnector`. It exports the writer's resolved key identity so a
reader can reconstruct source keys without using its own parallel or model
configuration. See [the connector design](layerwise_pool_connector.md) for
request handling, state compatibility, and implementation limitations. Key
reconstruction alone does not enable safe pool-backed PD.

## Existing backend integration

`MemcacheBackend.bind_layerwise_keys()` and
`MooncakeBackend.bind_layerwise_keys()` still return `LayerwiseKeyBuilder`.
The builder now also exposes an immutable `identity`. Existing callers continue
to use `make_full_key()` and `make_hit_check_keys()`.

The identity delegates to the existing backend key functions. It retains the
deployed MemCache, Mooncake block and Mooncake hybrid key formats. The writer
computes Mooncake's topology namespace and hybrid layout hash once. The reader
receives those values rather than recomputing them from its own configuration.

```python
# Writer: existing backend binding, including resolved source layout.
writer_keys = backend_class.bind_layerwise_keys(
    vllm_config=source_config,
    kv_cache_config=source_cache_config,
    model_name=source_model_name,
    use_hybrid=source_uses_hybrid,
    grouped_block_size=source_group_block_sizes,
)
source_identity = writer_keys.identity.to_wire()

# Reader: source_identity has passed through JSON metadata.
source_keys = PoolKeyIdentity.from_wire(source_identity).bind()
key = source_keys.make_full_key(
    source_group_id, source_block_hash, source_head_or_tp_rank, source_pp_rank
)
```

The reader's local group and rank only select the destination cache. They must
not replace the source coordinates passed to `make_full_key()`.

## Protocol boundaries

- `format` selects a known existing key format. Unknown formats and malformed
  identities are rejected.
- `group_block_sizes` and `layout_id` apply to Mooncake hybrid keys.
- `partial_key()` delegates to the ordinary MemCache partial-key function. A PD
  tail object ID is a separate protocol and uses `make_full_key()` when the
  producer has explicitly selected that protocol.
- Key reconstruction does not prove that an object is committed, retained, or
  compatible with the target layout. The transfer profile and restore planner
  must check those properties before reading.
- The legacy MemCache key format does not encode an arbitrary layout namespace.
  Shared GVA mode therefore passes a versioned content identity to the existing
  key function: `pool-full-v1:<schema>:<prefix-hash>` for complete component-major
  objects, or `pool-component-v1:<digest>` for explicit component ranges. The
  schema includes the declared byte order, head range, component identities and
  offsets. Ordinary shared pooling and pooled PD use the same planner. Legacy
  unqualified full-object keys are not inferred to have a compatible format.
- Unaccepted layer-reuse work uses request-step-private `pool-step-v1` IDs.
  The scheduler supplies the step ID to every replica; only the writer owns
  deletion. These IDs cannot be discovered by a shared prefix query. Accepted
  hash-backed pages retain shared IDs. Shorter acceptance on the next step
  reads the saved descriptors rather than synthesizing a new partial key.
- Existing backend topology checks still run when binding the writer. Exporting
  an identity does not expand the backend's supported TP/PP/CP combinations.

## Tests

`test_key_codec.py` covers JSON round trips, source/target layout differences,
partial versus PD tail keys, invalid metadata, and immutable identity snapshots.
`test_layerwise_keys.py` additionally checks that exported and rebound identities
produce the same keys as the existing writer/lookup paths across backends,
groups, ranks and PP stages. `test_backend.py` retains its wire-format regression
checks.
