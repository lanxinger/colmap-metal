import importlib.util
from pathlib import Path


def load_benchmark_module():
    script_path = Path(__file__).parents[1] / "scripts/benchmark_sift_metal.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_sift_metal", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_largest_model_analyzer_selects_most_registered_images(
    tmp_path, monkeypatch
):
    benchmark = load_benchmark_module()
    sparse_path = tmp_path / "sparse"
    for name in ("0", "1", "diagnostics"):
        (sparse_path / name).mkdir(parents=True)

    model_metrics = {
        "0": (2, 10),
        "1": (77, 21000),
        "diagnostics": (0, 0),
    }

    def fake_run_command(label, command, cwd):
        del label, cwd
        model_name = Path(command[-1]).name
        images, points = model_metrics[model_name]
        return {
            "seconds": float(images),
            "output": (
                f"[I] Registered images: {images}\n[I] Points: {points}\n"
            ),
        }

    monkeypatch.setattr(benchmark, "run_command", fake_run_command)

    result = benchmark.run_largest_model_analyzer(
        Path("colmap"), sparse_path, tmp_path
    )

    assert result["model_path"] == str(sparse_path / "1")
    assert result["num_models"] == 2
    assert result["model_analyzer"]["registered_images"] == "77"
    assert result["model_analyzer"]["points"] == "21000"
