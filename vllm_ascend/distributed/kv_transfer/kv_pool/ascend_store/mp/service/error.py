class RegistrationConflictError(RuntimeError):
    pass


class ServiceBusyError(RuntimeError):
    pass


class StaleSessionError(RuntimeError):
    pass
