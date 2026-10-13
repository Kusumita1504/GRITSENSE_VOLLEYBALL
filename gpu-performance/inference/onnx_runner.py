"""ONNX Runtime runner with explicit provider selection and input validation."""

from pathlib import Path

import numpy as np


class ONNXRunner:
    INPUT_NAMES = {"states", "actions"}
    OUTPUT_NAME = "epv"

    def __init__(self, model_path: str | Path, execution_provider: str = "CPUExecutionProvider") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is not installed. Install onnxruntime or onnxruntime-gpu.") from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        if execution_provider not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
            raise ValueError("execution_provider must be CPUExecutionProvider or CUDAExecutionProvider.")
        available = ort.get_available_providers()
        if execution_provider not in available:
            raise RuntimeError(
                f"Requested {execution_provider}, but installed ONNX Runtime providers are {available}. "
                "No provider fallback was attempted."
            )

        options = ort.SessionOptions()
        if execution_provider == "CUDAExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=[execution_provider])
        self.execution_provider = execution_provider
        effective = self.session.get_providers()
        if not effective or effective[0] != execution_provider:
            raise RuntimeError(
                f"ONNX Runtime did not activate requested provider {execution_provider}; active providers: {effective}."
            )

        inputs = {item.name: item for item in self.session.get_inputs()}
        if set(inputs) != self.INPUT_NAMES:
            raise ValueError(f"Expected ONNX inputs {sorted(self.INPUT_NAMES)}, got {sorted(inputs)}.")
        self.state_dim = self._validate_tensor(inputs["states"], "tensor(float)", rank=2)
        self._validate_tensor(inputs["actions"], "tensor(int64)", rank=1)
        outputs = self.session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != self.OUTPUT_NAME or len(outputs[0].shape) != 1:
            raise ValueError("Expected one rank-1 ONNX output named 'epv'.")
        print(f"[ONNX] requested provider={execution_provider}; active providers={effective}")

    @staticmethod
    def _validate_tensor(item, expected_type: str, rank: int) -> int | None:
        if item.type != expected_type or len(item.shape) != rank:
            raise ValueError(
                f"ONNX input {item.name!r} must have type {expected_type} and rank {rank}; "
                f"got type={item.type}, shape={item.shape}."
            )
        if item.name == "states" and isinstance(item.shape[-1], int):
            return item.shape[-1]
        return None

    def infer(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        if states.ndim != 2 or (self.state_dim is not None and states.shape[1] != self.state_dim):
            raise ValueError(f"states must have shape [batch, {self.state_dim or 'input_dim'}].")
        if actions.shape != (states.shape[0],):
            raise ValueError("actions must have shape [batch] matching states.")
        return self.session.run([self.OUTPUT_NAME], {"states": states, "actions": actions})[0]