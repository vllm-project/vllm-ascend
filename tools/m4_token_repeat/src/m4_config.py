"""Frozen M4 alert reasons, in native rule evaluation order."""

from dataclasses import dataclass

REASONS = (
    "vllm[32,500]x5",
    "vllm[32,2000]x3",
    "run>=500",
    "A3_n4_W1024>=0.6 S8 N_n4_W512<=0.05",
    "A5_n4_W512>=0.7 S4 N_n4_W512<=0.1",
    "A5_n4_W512>=0.5 S4 sep<=0.5 N_n4_W512<=0.05",
    "P_n8_W1024>=0.6 S24 N_n4_W512<=0.02",
    "wrep_W2048>=0.95 S16 & mark_W1024>=0.05",
    "wrep_W2048>=0.99 S24",
    "wrep_W1024>=0.99 S8 & mark_W1024>=0.04",
    "wrep_W1024>=0.98 S8 & mark_W1024>=0.05",
    "sdup_N16>=0.7 S24 & wrep_W2048>=0.9 & mark_W1024>=0.03",
    "sdup_N16>=0.6 S40 & mark_W1024>=0.03",
    "wrep_W2048>=0.95 S48 & mark_W1024>=0.03",
)


@dataclass
class Result:
    is_ill: bool = False
    fired_at: int = -1  # raw-token boundary at which the alert is raised
    reason: str = ""
