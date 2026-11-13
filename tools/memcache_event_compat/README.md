# Memcache 1.3.0 event key compatibility

This optional deployment workaround preserves existing cache keys while fixing
Memcache CPU confirmation events for GVA keys whose hash precedes a rank suffix.
It also ignores request-private partial snapshots and supports hash-last and
DSpark keys. It is not loaded by the Ascend plugin.

Run `bash tools/memcache_event_compat/build.sh` in the validated Memcache image.
The build rejects a different package version or library SHA256; review a new
library before updating `native_manifest.json`. The recorded library path is
specific to the validation image. Build outputs are ignored by Git.

Set `LD_PRELOAD` to the resulting `libmemcache_event_key_fix.so` for the Memcache
MetaService process only, and enable its `kv_events_hash_as_int=True` option to
match vLLM integer KV-event hashes. Do not preload it into inference workers.
Restart only MetaService after stopping its deployment's traffic. Roll back by
removing that preload and restarting MetaService. No cache-key migration is needed.

Remove this workaround once Memcache natively parses these existing key formats.
The C++ ABI override is version specific; this is not a general Memcache extension.
