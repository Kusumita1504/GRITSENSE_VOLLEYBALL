"""PyTorch inference runner for the repository EPV checkpoint."""

from pathlib import Path

import numpy as np
import torch

from inference.model_utils import DEFAULT_MODEL_PATH, load_epv_bundle, resolve_device


class PyTorchRunner:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        precision: str = "float32",
    ) -> None:
        if precision not in {"float32", "float16"}:
            raise ValueError("precision must be float32 or float16.")
        self.device = resolve_device(device)
        if precision == "float16" and self.device.type != "cuda":
            raise RuntimeError("float16 benchmarking is supported only on CUDA; no CPU fallback is performed.")
        self.precision = precision
        self.bundle = load_epv_bundle(model_path)
        self.model = self.bundle.normalized_model.to(self.device).eval()

    def infer(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        if states.ndim != 2 or states.shape[1] != self.bundle.state_dim:
            raise ValueError(f"states must have shape [batch, {self.bundle.state_dim}].")
        if actions.shape != (states.shape[0],):
            raise ValueError("actions must have shape [batch] matching states.")
        if np.any(actions < 0) or np.any(actions >= self.bundle.num_actions):
            raise ValueError(f"actions must be in [0, {self.bundle.num_actions - 1}].")

        state_tensor = torch.from_numpy(states).to(self.device)
        action_tensor = torch.from_numpy(actions).to(self.device)
        with torch.inference_mode():
            if self.precision == "float16":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(state_tensor, action_tensor)
            else:
                output = self.model(state_tensor, action_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return output.float().cpu().numpy()