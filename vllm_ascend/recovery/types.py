import msgspec
from typing import Any, Optional

class ExceptionInfo(msgspec.Struct):
    exception_type: str
    message: str

class RecoveryAction(msgspec.Struct):
    """
    Represents a single recovery action with its name and parameters.
    
    Attributes:
        name: Name of the recovery action
        config: Dictionary containing the parameters for this action
    """
    name: str
    config: dict[str, Any] = msgspec.field(default_factory=dict)

class RecoveryStep(msgspec.Struct):
    """
    Represents a single step in the recovery plan.
    
    A recovery step contains a series of actions that need to be executed
    by a specific executor, with an optional timeout.
    
    Attributes:
        name: Name of the recovery step
        executor: Name of the executor responsible for this step
        actions: List of RecoveryAction objects to be executed
        timeout: Timeout in seconds for this step (None means no timeout)
    """
    name: str
    target: str
    actions: list[RecoveryAction] = msgspec.field(default_factory=list)
    timeout: Optional[float] = None
    
class RecoveryPlan(msgspec.Struct):
    """
    Represents a complete recovery plan containing multiple steps.
    
    Attributes:
        steps: List of RecoveryStep objects defining the recovery workflow
    """
    name: str
    steps: list[RecoveryStep] = msgspec.field(default_factory=list)
    def add_step(self, step: RecoveryStep) -> None:
        """Add a recovery step to the plan."""
        self.steps.append(step)
    
    def is_empty(self) -> bool:
        """Check if the recovery plan contains any steps."""
        return len(self.steps) == 0
    
    def get_step_count(self) -> int:
        """Get the number of steps in the recovery plan."""
        return len(self.steps)

class FaultReport(msgspec.Struct):
    worker_rank: int
    fault_type: str
    recovery_plan: RecoveryPlan
    timestamp: float
    context: dict = msgspec.field(default_factory=dict)

class StepResult(msgspec.Struct):
    step_name: str
    worker_rank: int
    is_success: bool
    engine_index: Optional[int] = None
    error: Optional[str] = None