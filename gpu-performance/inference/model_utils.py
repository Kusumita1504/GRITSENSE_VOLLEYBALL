"""Adapter for the repository's checkpoint-backed EPV PyTorch model."""

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import Tensor, nn


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "epv_phase3" / "epv_model.pt"
EPV_SOURCE = Path(__file__).resolve().parents[2] / "epv_phase3"


@dataclass
class ModelBundle:
    model: nn.Module
    normalized_model: nn.Module
    state_dim: int
    num_actions: int
    checkpoint_path: Path
    checkpoint: dict


class NormalizedEPV(nn.Module):
    """Apply the training checkpoint's normalization before EPV inference."""

    def __init__(self, model: nn.Module, mean: Tensor, std: Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std", std)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.model((states - self.norm_mean) / self.norm_std, actions)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation has no available CUDA device.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    return torch.device(requested)


def load_epv_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> ModelBundle:
    """Load the checked-in EPV architecture and its local checkpoint metadata."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"EPV checkpoint not found: {path}")
    if str(EPV_SOURCE) not in sys.path:
        sys.path.insert(0, str(EPV_SOURCE))
    try:
        from epv.model import EPVModel
    except ImportError as exc:
        raise RuntimeError(f"Could not import the repository EPVModel from {EPV_SOURCE}.") from exc

    try:
        # Existing EPV checkpoints contain NumPy normalization arrays; only load trusted local artifacts.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = EPVModel(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        mean = torch.as_tensor(checkpoint["norm_mean"], dtype=torch.float32)
        std = torch.as_tensor(checkpoint["norm_std"], dtype=torch.float32)
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid or incompatible EPV checkpoint {path}: {exc}") from exc

    state_dim = int(model.state_dim)
    if mean.shape != (1, state_dim) or std.shape != (1, state_dim):
        raise ValueError(
            f"Checkpoint normalization must have shape (1, {state_dim}); "
            f"got mean={tuple(mean.shape)}, std={tuple(std.shape)}."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std == 0):
        raise ValueError("Checkpoint normalization contains non-finite values or zero standard deviations.")

    model.eval()
    normalized_model = NormalizedEPV(model, mean, std).eval()
    return ModelBundle(
        model=model,
        normalized_model=normalized_model,
        state_dim=state_dim,
        num_actions=int(model.num_actions),
        checkpoint_path=path,
        checkpoint=checkpoint,
    )