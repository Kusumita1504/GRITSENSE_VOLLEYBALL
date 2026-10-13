"""Profile EPV inference, and optionally the repository's video stages."""

import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from inference.model_utils import DEFAULT_MODEL_PATH, load_epv_bundle, resolve_device


def _stage_summary(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p90_ms": None}
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": len(samples),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def detect_nsight_systems() -> dict[str, str | bool | None]:
    executable = shutil.which("nsys")
    return {"available": executable is not None, "executable": executable}


def profile_epv(model_path: str | Path, device_name: str, batch_size: int, steps: int,
                trace_path: str | Path | None = None) -> dict:
    device = resolve_device(device_name)
    bundle = load_epv_bundle(model_path)
    model = bundle.normalized_model.to(device).eval()
    rng = np.random.default_rng(12)
    host_states = rng.standard_normal((batch_size, bundle.state_dim), dtype=np.float32)
    host_actions = rng.integers(0, bundle.num_actions, size=batch_size, dtype=np.int64)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiler_options = {
        "activities": activities,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": False,
    }
    if "acc_events" in inspect.signature(torch.profiler.profile).parameters:
        profiler_options["acc_events"] = True
    with torch.profiler.profile(**profiler_options) as profiler:
        with torch.inference_mode():
            for _ in range(steps):
                with torch.profiler.record_function("input_host_to_device"):
                    states = torch.from_numpy(host_states).to(device)
                    actions = torch.from_numpy(host_actions).to(device)
                with torch.profiler.record_function("epv_model_forward"):
                    outputs = model(states, actions)
                with torch.profiler.record_function("output_device_to_host_and_sync"):
                    outputs.detach().cpu().numpy()

    device_kernel_time_us = None
    event_rows = []
    for event in profiler.key_averages():
        self_device_time = getattr(event, "self_device_time_total", None)
        if self_device_time is None:
            self_device_time = getattr(event, "self_cuda_time_total", None)
        if device.type == "cuda" and self_device_time is not None:
            device_kernel_time_us = (device_kernel_time_us or 0.0) + float(self_device_time)
        event_rows.append({
            "operator": event.key,
            "calls": event.count,
            "cpu_self_time_us": float(event.self_cpu_time_total),
            "device_self_time_us": (
                float(self_device_time) if device.type == "cuda" and self_device_time is not None else None
            ),
            "cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
            "device_memory_bytes": int(getattr(event, "self_device_memory_usage", 0)),
        })
    result = {
        "workload": "EPVModel forward using seeded synthetic raw state inputs",
        "device": str(device),
        "batch_size": batch_size,
        "steps": steps,
        "host_input_transfer_included_in_trace": True,
        "host_output_transfer_and_sync_included_in_trace": True,
        "gpu_kernel_time_us_sum": device_kernel_time_us,
        "gpu_kernel_time_note": (
            "PyTorch Profiler device activity aggregate; null on CPU. It is distinct from host elapsed latency."
        ),
        "events": event_rows,
    }
    print(profiler.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total",
                                        row_limit=25))
    if trace_path:
        trace = Path(trace_path).expanduser().resolve()
        trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace))
        result["chrome_trace"] = str(trace)
    return result


def profile_video(video_path: str | Path, yolo_model_path: str | Path, device_name: str,
                  limit: int | None, pose_model_path: str | Path | None = None,
                  input_size: tuple[int, int] | None = None,
                  court_matrix_path: str | Path | None = None) -> dict:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Video profiling requires the root application's opencv-python and ultralytics dependencies.") from exc

    device = resolve_device(device_name)
    yolo_device = 0 if device.type == "cuda" else "cpu"
    detector = YOLO(str(yolo_model_path))
    pose_model = YOLO(str(pose_model_path)) if pose_model_path else None
    transform = None
    if court_matrix_path:
        transform = np.asarray(json.loads(Path(court_matrix_path).read_text(encoding="utf-8")), dtype=np.float32)
        if transform.shape != (3, 3):
            raise ValueError("Court matrix JSON must contain a 3x3 homography matrix.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    stages: dict[str, list[float]] = {
        "frame_end_to_end_host_elapsed_ms": [],
        "video_decode_host_ms": [],
        "resize_preprocess_host_ms": [],
        "detector_tracking_host_elapsed_ms": [],
        "pose_model_host_elapsed_ms": [],
        "tracking_postprocess_host_ms": [],
        "court_transform_host_ms": [],
    }
    frame_count = 0
    try:
        while limit is None or frame_count < limit:
            frame_start = time.perf_counter_ns()
            start = time.perf_counter_ns()
            ok, frame = capture.read()
            stages["video_decode_host_ms"].append((time.perf_counter_ns() - start) / 1e6)
            if not ok:
                stages["video_decode_host_ms"].pop()
                break
            if input_size:
                start = time.perf_counter_ns()
                frame = cv2.resize(frame, input_size)
                stages["resize_preprocess_host_ms"].append((time.perf_counter_ns() - start) / 1e6)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            tracked = detector.track(
                frame, persist=True, conf=0.4, tracker="botsort.yaml", verbose=False,
                device=yolo_device, half=device.type == "cuda",
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stages["detector_tracking_host_elapsed_ms"].append((time.perf_counter_ns() - start) / 1e6)

            if pose_model:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                pose_model.predict(frame, verbose=False, device=yolo_device, half=device.type == "cuda")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                stages["pose_model_host_elapsed_ms"].append((time.perf_counter_ns() - start) / 1e6)

            points = []
            start = time.perf_counter_ns()
            for result in tracked:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    points.extend(((box[0] + box[2]) * 0.5, box[3]) for box in xyxy)
            stages["tracking_postprocess_host_ms"].append((time.perf_counter_ns() - start) / 1e6)
            if transform is not None:
                start = time.perf_counter_ns()
                if points:
                    cv2.perspectiveTransform(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), transform)
                stages["court_transform_host_ms"].append((time.perf_counter_ns() - start) / 1e6)
            stages["frame_end_to_end_host_elapsed_ms"].append(
                (time.perf_counter_ns() - frame_start) / 1e6
            )
            frame_count += 1
    finally:
        capture.release()

    return {
        "workload": "GritSense EPV YOLO tracking stages; model call includes Ultralytics preprocessing, inference, postprocessing, and tracking",
        "video": str(Path(video_path).resolve()),
        "yolo_model": str(Path(yolo_model_path).resolve()),
        "pose_model": str(Path(pose_model_path).resolve()) if pose_model_path else None,
        "frames": frame_count,
        "device": str(device),
        "gpu_kernel_time_reported": False,
        "stages": {name: _stage_summary(samples) for name, samples in stages.items()},
        "court_transform_measured": transform is not None,
        "note": "Stage timings are synchronized host elapsed times, not isolated GPU kernel times.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--trace", help="Optional Chrome trace JSON output path")
    parser.add_argument("--video")
    parser.add_argument("--yolo-model", help="Ultralytics weights used by the GritSense tracking path")
    parser.add_argument("--pose-model", help="Optional Ultralytics pose weights; no pose model is wired by default")
    parser.add_argument("--court-matrix", help="Optional JSON 3x3 homography for court-transform profiling")
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--input-height", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="profile_results.json")
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1:
        parser.error("--batch-size and --steps must be positive")
    results = {
        "external_tools": {"nsight_systems": detect_nsight_systems()},
        "epv_profiler": profile_epv(args.model_path, args.device, args.batch_size, args.steps, args.trace),
    }
    if args.video:
        if not args.yolo_model:
            parser.error("--video requires --yolo-model")
        size = (args.input_width, args.input_height) if args.input_width and args.input_height else None
        if (args.input_width is None) != (args.input_height is None):
            parser.error("--input-width and --input-height must be given together")
        results["video_pipeline"] = profile_video(args.video, args.yolo_model, args.device, args.limit,
                                                  args.pose_model, size, args.court_matrix)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Profile summary written to {output}")


if __name__ == "__main__":
    main()