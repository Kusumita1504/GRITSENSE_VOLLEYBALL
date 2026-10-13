"""Export the repository EPV model to ONNX with CPU-runtime parity checks."""

import argparse
from pathlib import Path

import numpy as np
import torch

from inference.model_utils import DEFAULT_MODEL_PATH, load_epv_bundle
from inference.onnx_runner import ONNXRunner


def export(model_path: str | Path, output_path: str | Path, batch_size: int = 4) -> Path:
    try:
        import onnx
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError("ONNX export validation requires both onnx and onnxruntime; install them optionally.") from exc

    bundle = load_epv_bundle(model_path)
    wrapper = bundle.normalized_model.cpu().eval()
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(batch_size, bundle.state_dim, generator=generator)
    actions = torch.arange(batch_size, dtype=torch.int64) % bundle.num_actions
    with torch.inference_mode():
        expected = wrapper(states, actions).numpy()

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (states, actions),
        str(output),
        input_names=["states", "actions"],
        output_names=["epv"],
        dynamic_axes={"states": {0: "batch"}, "actions": {0: "batch"}, "epv": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output)))
    runner = ONNXRunner(output, execution_provider="CPUExecutionProvider")
    actual = runner.infer(states.numpy(), actions.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    print(f"ONNX CPU output validation passed; model saved to {output}; runtime={onnxruntime.__version__}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", default="models/epv_model.onnx")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    export(args.model_path, args.output, args.batch_size)


if __name__ == "__main__":
    main()