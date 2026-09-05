"""KV cache business errors shared by the client and server."""

SERVICE_NOT_REGISTERED_PREFIX = "ServiceNotRegisteredError:"
STALE_SESSION_PREFIX = "StaleSessionError:"


class ServiceNotRegisteredError(RuntimeError):
    pass


class ServiceSessionExpiredError(RuntimeError):
    pass
