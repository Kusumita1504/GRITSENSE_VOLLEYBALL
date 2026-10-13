"""TorchScript inference runner for exported normalized EPV models."""

from pathlib import Path

import numpy as np
import torch

from inference.model_utils import resolve_device


class TorchScriptRunner:
    def __init__(self, model_path: str | Path, device: str = "auto", state_dim: int | None = None,
                 num_actions: int | None = None) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TorchScript model not found: {path}")
        self.device = resolve_device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        self.state_dim = state_dim
        self.num_actions = num_actions

    def infer(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        if states.ndim != 2 or (self.state_dim is not None and states.shape[1] != self.state_dim):
            raise ValueError(f"states must have shape [batch, {self.state_dim or 'input_dim'}].")
        if actions.shape != (states.shape[0],):
            raise ValueError("actions must have shape [batch] matching states.")
        if self.num_actions is not None and (np.any(actions < 0) or np.any(actions >= self.num_actions)):
            raise ValueError(f"actions must be in [0, {self.num_actions - 1}].")
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        with torch.inference_mode():
            output = self.model(states, actions)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        values = output.float().cpu().numpy()
        if values.shape != (states.shape[0],):
            raise ValueError(f"TorchScript output must have shape [batch], got {values.shape}.")
        return values