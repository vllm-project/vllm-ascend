# Dependency ownership

`runtime.txt` is the source for the wheel's runtime dependencies. The root
`requirements.txt` is a compatibility entry point that includes this file;
`setup.py` resolves that include when generating `Requires-Dist` metadata.
Keep runtime entries as PEP 508 requirements, without pip options or includes.

`pyproject.toml` declares isolated build dependencies. These are a separate
environment and must not be mechanically copied from the runtime list. Shared
native-stack versions must remain compatible. `requirements-dev.txt` includes
the runtime and lint entry points and adds development/test dependencies.

These declarations are not complete environment locks. Version range changes,
dependency reclassification and environment locks require separate compatibility
evidence. This layout migration preserves every runtime requirement and the
existing build configuration.
