"""Run repeatable model-only inference benchmarks for available runtimes."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from inference.benchmark_runner import summarize_latencies, write_results
from inference.config import load_config
from inference.model_utils import DEFAULT_MODEL_PATH, load_epv_bundle, resolve_device
from inference.onnx_runner import ONNXRunner
from inference.pytorch_runner import PyTorchRunner
from inference.torchscript_runner import TorchScriptRunner


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _measure(name, runner, states, actions, settings, device, provider, gpu_memory_reader=None,
             synchronize_torch=True, device_label=None):
    for _ in range(settings["warmup"]):
        runner.infer(states, actions)
    use_torch_cuda_sync = synchronize_torch and device.type == "cuda"
    if use_torch_cuda_sync:
        torch.cuda.synchronize(device)
        if gpu_memory_reader is not None:
            torch.cuda.reset_peak_memory_stats(device)

    elapsed_ms = []
    for _ in range(settings["repeats"]):
        for _ in range(settings["iterations"]):
            if use_torch_cuda_sync:
                torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            runner.infer(states, actions)
            if use_torch_cuda_sync:
                torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    peak_memory = None
    if gpu_memory_reader is not None:
        peak_memory = gpu_memory_reader() / (1024.0 * 1024.0)
    summary = summarize_latencies(elapsed_ms, settings["batch_size"])
    return {
        "runtime": name,
        "execution_provider": provider,
        "device": device_label or str(device),
        "precision": settings["precision"],
        "input_shape": [settings["batch_size"], settings["input_dim"]],
        "batch_size": settings["batch_size"],
        "warmup": settings["warmup"],
        "iterations_per_repeat": settings["iterations"],
        "repeats": settings["repeats"],
        "peak_gpu_memory_mb": peak_memory,
        "latency_samples_ms": elapsed_ms,
        **summary,
    }


def _metadata(bundle, settings, torch_device, provider):
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch_device if torch_device.type == "cuda" else 0)
    return {
        "model_path": str(bundle.checkpoint_path),
        "model": "EPVModel (checkpoint-backed repository implementation)",
        "model_input_dimension": bundle.state_dim,
        "model_output": "scalar EPV for one candidate action",
        "mock_inputs": True,
        "input_note": "Seeded random raw state vectors; checkpoint normalization is applied inside each model path.",
        "device_requested": settings["device"],
        "resolved_torch_device": str(torch_device),
        "requested_onnx_execution_provider": provider,
        "precision": settings["precision"],
        "batch_size": settings["batch_size"],
        "warmup": settings["warmup"],
        "iterations_per_repeat": settings["iterations"],
        "repeats": settings["repeats"],
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu_name,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "onnxruntime": _version("onnxruntime"),
        },
        "latency_definition": (
            "Host wall time per invocation, including host/device input and output transfers; "
            "CUDA is synchronized around each timed invocation. Not end-to-end video latency."
        ),
    }


def run_benchmarks(settings: dict, runtime: str = "all", onnx_model: str | None = None,
                   torchscript_model: str | None = None, seed: int = 0) -> dict:
    bundle = load_epv_bundle(settings["model_path"])
    if settings["input_dim"] is not None and settings["input_dim"] != bundle.state_dim:
        raise ValueError(f"--input-dim {settings['input_dim']} conflicts with checkpoint state_dim {bundle.state_dim}.")
    settings = {**settings, "input_dim": bundle.state_dim}
    torch_device = resolve_device(settings["device"])
    if settings["precision"] == "float16" and torch_device.type != "cuda":
        raise RuntimeError("float16 is only supported for a CUDA PyTorch benchmark; no CPU fallback is permitted.")
    if runtime == "onnx" and not onnx_model:
        raise ValueError("--runtime onnx requires --onnx-model.")
    if runtime == "torchscript" and not torchscript_model:
        raise ValueError("--runtime torchscript requires --torchscript-model.")
    if runtime not in {"all", "pytorch", "onnx", "torchscript"}:
        raise ValueError("runtime must be all, pytorch, onnx, or torchscript.")

    generator = np.random.default_rng(seed)
    states = generator.standard_normal((settings["batch_size"], bundle.state_dim), dtype=np.float32)
    actions = generator.integers(0, bundle.num_actions, size=settings["batch_size"], dtype=np.int64)
    metadata = _metadata(bundle, settings, torch_device, settings["execution_provider"])
    results = []
    outputs = {}
    skipped = []

    if runtime in {"all", "pytorch"}:
        pytorch = PyTorchRunner(bundle.checkpoint_path, str(torch_device), settings["precision"])
        row = _measure(
            "pytorch", pytorch, states, actions, settings, torch_device,
            "PyTorch CUDA" if torch_device.type == "cuda" else "CPU",
            torch.cuda.max_memory_allocated if torch_device.type == "cuda" else None,
        )
        outputs["pytorch"] = pytorch.infer(states, actions)
        results.append(row)

    if runtime in {"all", "onnx"} and onnx_model:
        if settings["precision"] != "float32":
            if runtime == "onnx":
                raise ValueError("This ONNX export is float32; float16 ONNX benchmarking is not supported.")
            skipped.append({"runtime": "onnx", "reason": "ONNX export is float32; precision does not match float16."})
        else:
            onnx_runner = ONNXRunner(onnx_model, settings["execution_provider"])
            if onnx_runner.state_dim not in {None, bundle.state_dim}:
                raise ValueError(f"ONNX model input dimension {onnx_runner.state_dim} != checkpoint {bundle.state_dim}.")
            provider_device = torch.device("cuda" if settings["execution_provider"] == "CUDAExecutionProvider" else "cpu")
            results.append(_measure("onnx", onnx_runner, states, actions, settings, provider_device,
                                    onnx_runner.execution_provider, synchronize_torch=False,
                                    device_label=onnx_runner.execution_provider))
            outputs["onnx"] = onnx_runner.infer(states, actions)

    if runtime in {"all", "torchscript"} and torchscript_model:
        if settings["precision"] != "float32":
            if runtime == "torchscript":
                raise ValueError("This TorchScript export is float32; float16 benchmarking is not supported.")
            skipped.append({"runtime": "torchscript", "reason": "TorchScript export is float32; precision does not match float16."})
        else:
            scripted = TorchScriptRunner(torchscript_model, str(torch_device), bundle.state_dim,
                                         bundle.num_actions)
            results.append(_measure("torchscript", scripted, states, actions, settings, torch_device,
                                    "PyTorch CUDA" if torch_device.type == "cuda" else "CPU",
                                    torch.cuda.max_memory_allocated if torch_device.type == "cuda" else None))
            outputs["torchscript"] = scripted.infer(states, actions)

    if runtime in {"onnx", "torchscript"} and settings["precision"] == "float32" and "pytorch" not in outputs:
        reference = PyTorchRunner(bundle.checkpoint_path, "cpu", "float32")
        outputs["pytorch"] = reference.infer(states, actions)

    if runtime in {"onnx", "torchscript"} and not results:
        raise ValueError(f"No runtime was benchmarked. Check the requested model path and precision.")

    correctness = []
    if "pytorch" in outputs:
        for name, output in outputs.items():
            if name == "pytorch":
                continue
            if settings["precision"] != "float32":
                correctness.append({"reference": "pytorch", "candidate": name, "compatible": False,
                                    "reason": "precision mismatch; numerical comparison omitted"})
                continue
            difference = np.abs(outputs["pytorch"] - output)
            correctness.append({
                "reference": "pytorch",
                "candidate": name,
                "compatible": True,
                "allclose": bool(np.allclose(outputs["pytorch"], output, rtol=1e-4, atol=1e-5)),
                "max_absolute_error": float(difference.max(initial=0.0)),
                "rtol": 1e-4,
                "atol": 1e-5,
            })
    payload = {"metadata": metadata, "results": results, "correctness": correctness, "skipped": skipped}
    csv_path, json_path = write_results(settings["output_dir"], payload)
    console_results = [
        {key: value for key, value in row.items() if key != "latency_samples_ms"}
        for row in results
    ]
    print(json.dumps({"results": console_results, "correctness": correctness, "skipped": skipped,
                      "csv": str(csv_path), "json": str(json_path)}, indent=2, allow_nan=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--model-path")
    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--execution-provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"))
    parser.add_argument("--precision", choices=("float32", "float16"))
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--runtime", choices=("all", "pytorch", "onnx", "torchscript"), default="all")
    parser.add_argument("--onnx-model")
    parser.add_argument("--torchscript-model")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_config(args.config)
    for key in ("model_path", "input_dim", "batch_size", "device", "execution_provider",
                "precision", "warmup", "iterations", "repeats", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = Path(value).expanduser().resolve() if key in {"model_path", "output_dir"} else value
    if args.model_path is None and not settings.get("model_path"):
        settings["model_path"] = DEFAULT_MODEL_PATH
    for option in ("onnx_model", "torchscript_model"):
        value = getattr(args, option)
        if value:
            setattr(args, option, str(Path(value).expanduser().resolve()))
    run_benchmarks(settings, args.runtime, args.onnx_model, args.torchscript_model, args.seed)


if __name__ == "__main__":
    main()