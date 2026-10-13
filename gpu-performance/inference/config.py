"""Configuration loading and validation for the benchmark pipeline."""

from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "input_dim": None,
    "batch_size": 1,
    "device": "auto",
    "execution_provider": "CPUExecutionProvider",
    "warmup": 10,
    "iterations": 100,
    "repeats": 3,
    "precision": "float32",
    "output_dir": "benchmarks/results",
}


def validate_config(values: dict[str, Any]) -> dict[str, Any]:
    """Return a validated config with defaults filled in."""
    if not isinstance(values, dict):
        raise ValueError("Benchmark config must be a YAML mapping.")
    config = {**DEFAULTS, **values}
    if not config.get("model_path"):
        raise ValueError("Config must define a non-empty model_path.")
    for name in ("batch_size", "warmup", "iterations", "repeats"):
        value = config[name]
        minimum = 0 if name == "warmup" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if config["input_dim"] is not None and (
        isinstance(config["input_dim"], bool)
        or not isinstance(config["input_dim"], int)
        or config["input_dim"] <= 0
    ):
        raise ValueError("input_dim must be a positive integer or null.")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if config["execution_provider"] not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
        raise ValueError("execution_provider must be CPUExecutionProvider or CUDAExecutionProvider.")
    if config["precision"] not in {"float32", "float16"}:
        raise ValueError("precision must be float32 or float16.")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML settings and resolve file paths relative to that YAML file."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"Could not read benchmark config {config_path}: {exc}") from exc
    config = validate_config(values)
    for key in ("model_path", "output_dir"):
        candidate = Path(config[key]).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        config[key] = candidate.resolve()
    return config