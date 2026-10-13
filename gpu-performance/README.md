# GritSense GPU Performance Engineering

This is an isolated performance-engineering workbench. It leaves the volleyball application and its model files untouched. Its checked-in model target is the repository's EPV MLP (`epv_phase3/epv/model.py`) and its committed checkpoint (`epv_phase3/epv_model.pt`). The checkpoint describes a 69-feature input, six action IDs, hidden widths `[128, 64]`, and stores the training normalization arrays. Inference accepts raw state features, applies that stored normalization, and returns one scalar EPV value per state/action pair.

The EPV MLP was selected because it is a small, reconstructible `torch.nn.Module` with a local checkpoint contract and fixed tensor shapes. The root YOLO script is a stateful video/tracking loop, not a standalone tensor model interface. Cappy GAT/HRN requires graph-shaped inputs, PyG operators, and model checkpoints that do not share one stable exported-inference contract. Temporal models have no checked-in trained checkpoint. Those remain separate workloads rather than being silently reshaped into an artificial benchmark.

## Architecture

- `inference/model_utils.py` reconstructs the existing EPV model from its checkpoint and wraps checkpoint normalization into the model graph.
- `inference/pytorch_runner.py`, `onnx_runner.py`, and `torchscript_runner.py` run the same raw state/action interface. CUDA requests fail explicitly when unavailable; an ONNX CUDA request is never replaced with CPU execution.
- `models/export_torchscript.py` traces and reloads TorchScript, then checks output parity. `models/export_onnx.py` exports dynamic batch inputs, checks ONNX structure, runs ONNX Runtime CPU inference, and validates outputs against PyTorch.
- `benchmarks/run_benchmarks.py` measures host elapsed time per model invocation, reports repeated summary statistics and records the input, precision, runtime/provider, software, and device configuration. It compares outputs only when paths have matching FP32 inputs.
- `profiling/profile_inference.py` uses PyTorch Profiler for the EPV model, with CPU operator time and CUDA device activity separated. Optional video profiling records synchronized host elapsed times for decoding, resizing, YOLO tracking, optional pose inference, result postprocessing, and an optional court homography transform.
- `cuda/` contains an independent CPU/CUDA vector-add learning example. It makes no claim about accelerating GritSense.

## Install

Use a Python environment with Python 3.10 or newer and a PyTorch build suitable for your machine. For CPU development and tests, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r gpu-performance/requirements.txt
```

The base requirements are NumPy, PyYAML, and PyTorch. ONNX export and inference are optional:

```powershell
python -m pip install onnx onnxruntime
```

For ONNX CUDA execution, install the `onnxruntime-gpu` package that matches the local CUDA/cuDNN environment instead of `onnxruntime`, and verify `CUDAExecutionProvider` is present. Install a CUDA-enabled PyTorch build separately if PyTorch CUDA benchmarking/profiling is also wanted. TensorRT is not required or integrated.

The default checkpoint is already in the repository; no weights are downloaded. Checkpoints are loaded with `weights_only=False` because this repository's training artifact contains NumPy normalization arrays. Only use trusted local checkpoint files.

## Export

Run commands from `gpu-performance/`:

```powershell
python -m models.export_torchscript --model-path ..\epv_phase3\epv_model.pt --output models\epv_model.ts
python -m models.export_onnx --model-path ..\epv_phase3\epv_model.pt --output models\epv_model.onnx
```

TorchScript uses `torch.jit.script` for the EPV MLP's two tensor inputs and supports dynamic batch sizes. Both export commands compare test outputs to the original model; ONNX export additionally requires both `onnx` and `onnxruntime` and validates on the CPU provider. The checked-in EPV module was scripted and executed successfully with the available PyTorch build. PyTorch 2.12 emits a deprecation warning for TorchScript APIs, so this working export is a legacy-compatible path and may need replacement by `torch.export` in a future PyTorch release. ONNX conversion depends on the installed PyTorch exporter and ONNX package versions; an exporter/runtime incompatibility is surfaced as an error rather than producing a claimed-valid model. Exported artifacts are ignored by Git.

## Inference And Benchmarks

CPU-only smoke test:

```powershell
python -m benchmarks.run_benchmarks --runtime pytorch --model-path ..\epv_phase3\epv_model.pt --input-dim 69 --device cpu --batch-size 1 --warmup 3 --iterations 10 --repeats 2 --output-dir benchmarks\results\cpu-smoke
```

To compare PyTorch with a validated TorchScript export:

```powershell
python -m benchmarks.run_benchmarks --runtime all --model-path ..\epv_phase3\epv_model.pt --torchscript-model models\epv_model.ts --device cpu --batch-size 8 --warmup 10 --iterations 100 --repeats 5 --output-dir benchmarks\results\torchscript
```

With optional ONNX packages and a previously exported model:

```powershell
python -m benchmarks.run_benchmarks --runtime all --model-path ..\epv_phase3\epv_model.pt --torchscript-model models\epv_model.ts --onnx-model models\epv_model.onnx --execution-provider CPUExecutionProvider --device cpu --batch-size 8 --warmup 10 --iterations 100 --repeats 5 --output-dir benchmarks\results\cpu-comparison
```

`--runtime` accepts `pytorch`, `torchscript`, `onnx`, or `all`. For a CUDA run, request it explicitly:

```powershell
python -m benchmarks.run_benchmarks --runtime pytorch --device cuda --precision float32 --warmup 20 --iterations 200 --repeats 5 --output-dir benchmarks\results\cuda
```

An ONNX CUDA run must also explicitly request `--execution-provider CUDAExecutionProvider`; it errors if that provider is not installed or active. CPU and CUDA provider rows have different execution configurations and must not be treated as like-for-like results. FP16 autocast is supported only by the PyTorch CUDA runner; ONNX and TorchScript exports in this workbench are FP32 and are skipped (or rejected if explicitly selected) for FP16 to prevent mismatched comparisons.

The CLI reads `configs/benchmark.yaml` by default. CLI values override YAML values. The checkpoint fixes the actual feature width, so `--input-dim` is an assertion and a mismatched value is rejected, not a request to reshape the model. Inputs are seeded synthetic raw state vectors, and each runner applies the checkpoint's normalization. Supply the exact same checkpoint, batch size, seed, precision, and input dimension when comparing runtimes.

Each run writes `benchmark_results.csv` and `benchmark_results.json`. The files include mean, median, P90, P95, throughput, peak PyTorch CUDA allocated memory when measurable, raw timing samples, input dimensions, batch size, warmup/iteration/repeat counts, requested and active provider, software versions, and detected hardware. ONNX Runtime CUDA memory is `null` because its allocator is not visible through PyTorch's memory API. Latency is host wall time around one invocation, including input transfer and output staging; CUDA is synchronized around measured calls. This is model-only inference latency, not end-to-end video latency. GPU kernel timing belongs to the profiler, not this host timer.

## Profiling

Profile the EPV model and optionally export a Chrome trace:

```powershell
python -m profiling.profile_inference --model-path ..\epv_phase3\epv_model.pt --device cpu --batch-size 8 --steps 20 --trace benchmarks\results\epv-trace.json --output benchmarks\results\epv-profile.json
```

PyTorch Profiler reports CPU operators, shapes and memory; on CUDA-enabled PyTorch it additionally records device activity/kernel time. The profiled region explicitly includes host-to-device input copies and output copy/synchronization. CPU operator time is not called GPU kernel time.

The existing EPV perception path can also be profiled with a local video and Ultralytics weights. The report separates video decode, resize, tracking, pose, result postprocessing and court transform stage host times, and includes per-frame host end-to-end elapsed time:

```powershell
python -m profiling.profile_inference --device auto --video path\to\match.mp4 --yolo-model ..\epv_phase3\player_detection.pt --limit 300 --output benchmarks\results\video-profile.json
```

Optional `--pose-model` profiles an explicitly supplied pose model; pose is not enabled by default in the repository EPV main path. Optional `--input-width` and `--input-height` measure a separate resize stage. Optional `--court-matrix homography.json` accepts a JSON 3x3 matrix for timing the OpenCV perspective transform. The tracker call timing includes Ultralytics' own preprocessing, inference, postprocessing and tracking because its repository API does not expose those internal stages as separate timers. All video values are synchronized host elapsed time; they are not isolated GPU kernel time. No video is bundled for repeatable end-to-end measurements.

The profiler report detects whether `nsys` is on `PATH`. Nsight Systems is optional; if installed, use it around a CUDA-enabled benchmark to inspect CUDA API calls, kernels, copies and CPU/GPU scheduling:

```powershell
nsys profile --trace=cuda,nvtx,osrt --output benchmarks\results\nsys-epv python -m benchmarks.run_benchmarks --runtime pytorch --device cuda --warmup 20 --iterations 200 --repeats 5 --output-dir benchmarks\results\nsys-inner
```

Nsight is not required for the Python profiler or CPU path.

## CUDA Example

See [cuda/README.md](cuda/README.md). The example requires an NVIDIA GPU, compatible driver, CUDA Toolkit/`nvcc`, and supported C++17 host compiler. It validates its output against a CPU implementation and prints separate CPU host, CUDA event kernel-only, and CUDA allocation/transfer-inclusive end-to-end timings. It is an instructional vector operation only.

## Correctness And Limitations

- PyTorch uses the model implementation and checkpoint as-is; no weights are changed or downloaded. Inputs use the checkpoint's saved normalization consistently across runtimes.
- Export parity tolerances are `rtol=1e-5, atol=1e-6` for TorchScript and `rtol=1e-4, atol=1e-5` for ONNX Runtime.
- EPV training targets and action labels in this repository are heuristic proxies, not validated volleyball outcomes/actions. These benchmarks measure implementation/runtime behavior, not prediction quality.
- The EPV model is a compact MLP; its measurements do not characterize the full YOLO video workflow, GAT/HRN, or temporal models.
- GAT/HRN export is not provided because the available model/checkpoint interfaces and graph inputs are not a unified supported inference contract. Temporal weights are not checked in. The video path is opt-in and requires local video/model assets.
- GPU timing, CUDA memory, ONNX Runtime, and CUDA C++ execution require corresponding hardware/toolkits and could not be tested on CPU-only installations.
- Results vary with GPU/CPU, drivers, clocks, runtime/provider versions, batch size, and workload. Reproduce on another machine by installing compatible providers, using the same checkpoint and CLI settings, preserving generated JSON/CSV, and recording the software/hardware metadata. Compare only compatible rows.

## Performance Results

One CPU-only smoke measurement was collected in the current environment. It is included as a local reference, not a comparative performance claim. Its raw latency samples and environment metadata are retained in the Git-ignored `benchmarks/results/local-cpu/` output directory.

| Runtime/provider | Input / batch | Precision | Mean / median (ms) | P90 / P95 (ms) | Throughput (inferences/s) | Hardware |
|---|---:|---|---:|---:|---:|---|
| PyTorch / CPU | 69 features / 1 | FP32 | 0.3822 / 0.3087 | 0.5805 / 0.6635 | 2,616.7 | Intel64 Family 6 Model 142; no GPU available |

This row used 20 warmups and 3 repeats of 100 timed inferences (300 samples), Python 3.10.8, PyTorch 2.12.0+cpu, and NumPy 2.2.6. Reproduce it from this directory with:

```powershell
python -m benchmarks.run_benchmarks --runtime pytorch --model-path ..\epv_phase3\epv_model.pt --input-dim 69 --device cpu --batch-size 1 --precision float32 --warmup 20 --iterations 100 --repeats 3 --output-dir benchmarks\results\local-cpu
```

The reported host-time distribution varies between runs; compare against fresh measurements on the target machine rather than interpreting this single CPU run as a stable service-level figure.