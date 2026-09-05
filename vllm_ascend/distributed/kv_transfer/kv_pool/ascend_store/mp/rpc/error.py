class MPError(Exception):
    pass


class MPProtocolError(MPError):
    pass


class MPRemoteError(MPError):
    """An exception raised by a remote RPC handler.

    The summary remains the first line for compatibility with business error
    translation. Diagnostic context follows without replacing the root error.
    """

    def __init__(
        self,
        message: str,
        *,
        method: str | None = None,
        request_id: bytes | None = None,
        remote_traceback: str | None = None,
    ):
        self.remote_method = method
        self.remote_request_id = request_id
        self.remote_traceback = remote_traceback
        details = [message]
        if method is not None:
            details.append(f"Remote RPC method: {method}")
        if request_id is not None:
            details.append(f"Remote request ID: {request_id!r}")
        if remote_traceback:
            details.extend(("Remote traceback:", remote_traceback.rstrip()))
        super().__init__("\n".join(details))


class MPClientClosedError(MPError):
    pass


class MPServerUnavailableError(MPError, ConnectionError):
    pass


class MPServerAbortedError(MPServerUnavailableError):
    pass


class MPRequestTimeoutError(MPError, TimeoutError):
    pass


class MPServerBusyError(MPError):
    pass
