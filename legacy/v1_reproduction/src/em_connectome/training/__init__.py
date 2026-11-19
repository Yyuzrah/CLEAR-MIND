"""Configuration-driven training and evaluation."""

from .engine import RunResult, resolve_device, run_training, set_global_seed

__all__ = ["RunResult", "resolve_device", "run_training", "set_global_seed"]
