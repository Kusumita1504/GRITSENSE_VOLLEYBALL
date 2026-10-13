"""Reusable latency statistics and CSV/JSON result serialization."""

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


SUMMARY_FIELDS = (
    "runtime",
    "execution_provider",
    "device",
    "model_path",
    "cpu",
    "gpu",
    "python_version",
    "torch_version",
    "numpy_version",
    "onnxruntime_version",
    "device_requested",
    "requested_onnx_execution_provider",
    "precision",
    "input_shape",
    "batch_size",
    "warmup",
    "iterations_per_repeat",
    "repeats",
    "mean_ms",
    "median_ms",
    "p90_ms",
    "p95_ms",
    "throughput_inferences_per_second",
    "peak_gpu_memory_mb",
)


def summarize_latencies(latencies_ms: list[float], batch_size: int) -> dict[str, float]:
    if (
        not latencies_ms
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("At least one latency and a positive batch_size are required.")
    values = np.asarray(latencies_ms, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Latency values must be finite and positive.")
    average = mean(values.tolist())
    return {
        "mean_ms": float(average),
        "median_ms": float(median(values.tolist())),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "throughput_inferences_per_second": float(batch_size * 1000.0 / average) if average > 0 else float("inf"),
    }


def write_results(output_dir: str | Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "benchmark_results.json"
    csv_path = destination / "benchmark_results.csv"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        metadata = payload.get("metadata", {})
        software = metadata.get("software", {})
        csv_metadata = {
            "model_path": metadata.get("model_path"),
            "cpu": metadata.get("cpu"),
            "gpu": metadata.get("gpu"),
            "python_version": software.get("python"),
            "torch_version": software.get("torch"),
            "numpy_version": software.get("numpy"),
            "onnxruntime_version": software.get("onnxruntime"),
            "device_requested": metadata.get("device_requested"),
            "requested_onnx_execution_provider": metadata.get("requested_onnx_execution_provider"),
        }
        writer.writerows({**csv_metadata, **row} for row in payload["results"])
    return csv_path, json_path