import csv
import json
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from inference.benchmark_runner import summarize_latencies, write_results
from inference.model_utils import NormalizedEPV
from inference.onnx_runner import ONNXRunner
from inference.model_utils import resolve_device


class TinyEPV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)

    def forward(self, states, actions):
        action_features = torch.nn.functional.one_hot(actions, num_classes=3).to(states.dtype)
        return self.linear(torch.cat((states, action_features), dim=1)).squeeze(-1)


class TestBenchmarkMetrics(unittest.TestCase):
    def test_summary_statistics(self):
        summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0], batch_size=2)
        self.assertEqual(summary["mean_ms"], 3.0)
        self.assertEqual(summary["median_ms"], 3.0)
        self.assertEqual(summary["p90_ms"], 4.6)
        self.assertEqual(summary["p95_ms"], 4.8)
        self.assertAlmostEqual(summary["throughput_inferences_per_second"], 2000.0 / 3.0)

    def test_rejects_empty_or_nonfinite_samples(self):
        with self.assertRaises(ValueError):
            summarize_latencies([], batch_size=1)
        with self.assertRaises(ValueError):
            summarize_latencies([float("nan")], batch_size=1)
        with self.assertRaises(ValueError):
            summarize_latencies([0.0], batch_size=1)

    def test_csv_and_json_results(self):
        payload = {
            "metadata": {"device": "cpu"},
            "results": [{"runtime": "pytorch", "device": "cpu", "mean_ms": 1.25}],
            "correctness": [],
            "skipped": [],
        }
        with TemporaryDirectory() as temp_dir:
            csv_path, json_path = write_results(temp_dir, payload)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), payload)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["runtime"], "pytorch")
            self.assertEqual(rows[0]["mean_ms"], "1.25")
            self.assertIn("python_version", rows[0])

    def test_torchscript_output_matches_normalized_pytorch(self):
        model = TinyEPV().eval()
        wrapped = NormalizedEPV(model, torch.ones(1, 5), torch.full((1, 5), 2.0)).eval()
        states = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        actions = torch.tensor([0, 1, 2], dtype=torch.int64)
        with torch.inference_mode():
            expected = wrapped(states, actions)
            scripted = torch.jit.script(wrapped)
            actual = scripted(states, actions)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_cpu_device_resolution(self):
        self.assertEqual(str(resolve_device("cpu")), "cpu")
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                resolve_device("cuda")

    def test_missing_onnx_runtime_has_actionable_error(self):
        with patch.dict(sys.modules, {"onnxruntime": None}):
            with self.assertRaisesRegex(RuntimeError, "ONNX Runtime is not installed"):
                ONNXRunner("not-opened.onnx")


if __name__ == "__main__":
    unittest.main()