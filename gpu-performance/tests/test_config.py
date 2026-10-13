import unittest

from inference.config import load_config, validate_config


class TestBenchmarkConfig(unittest.TestCase):
    def test_defaults_and_values(self):
        config = validate_config({"model_path": "model.pt"})
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["device"], "auto")
        self.assertEqual(config["precision"], "float32")

    def test_rejects_invalid_iteration_counts(self):
        with self.assertRaisesRegex(ValueError, "iterations"):
            validate_config({"model_path": "model.pt", "iterations": 0})

    def test_loads_yaml_and_resolves_paths(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "benchmark.yaml"
            config_file.write_text("model_path: weights/model.pt\niterations: 4\n", encoding="utf-8")
            config = load_config(config_file)
            self.assertEqual(config["iterations"], 4)
            self.assertEqual(config["model_path"], (Path(temp_dir) / "weights/model.pt").resolve())


if __name__ == "__main__":
    unittest.main()