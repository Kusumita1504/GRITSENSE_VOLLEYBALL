"""Trace the repository EPV model for TorchScript and validate its outputs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from inference.model_utils import DEFAULT_MODEL_PATH, load_epv_bundle
from inference.torchscript_runner import TorchScriptRunner


def export(model_path: str | Path, output_path: str | Path, batch_size: int = 4) -> Path:
    bundle = load_epv_bundle(model_path)
    wrapper = bundle.normalized_model.cpu().eval()
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(batch_size, bundle.state_dim, generator=generator)
    actions = torch.arange(batch_size, dtype=torch.int64) % bundle.num_actions
    with torch.inference_mode():
        expected = wrapper(states, actions)
        scripted = torch.jit.script(wrapper)
        actual = scripted(states, actions)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    reloaded = TorchScriptRunner(output, device="cpu", state_dim=bundle.state_dim,
                                 num_actions=bundle.num_actions)
    np.testing.assert_allclose(reloaded.infer(states.numpy(), actions.numpy()), expected.numpy(), rtol=1e-5, atol=1e-6)
    print(f"TorchScript output validation passed; model saved to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", default="models/epv_model.ts")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    export(args.model_path, args.output, args.batch_size)


if __name__ == "__main__":
    main()