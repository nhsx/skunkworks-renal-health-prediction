from pathlib import Path

from aki_predictions.evaluation.performance_metrics import main as evaluate_main


class TestPerformanceMetricsMain:
    def test_main_function(self, tmpdir):
        output_dir = Path(tmpdir) / "test_main_log"

        if not Path.exists(output_dir):
            Path.mkdir(output_dir)

        evaluate_main(str(output_dir), Path("tests") / "fixtures" / "mock_metrics.json")

        # Performance plots
        assert Path.exists(output_dir / "mortality-performance.png")
        assert Path.exists(output_dir / "mortality-pr.png")
        assert Path.exists(output_dir / "mortality-roc.png")
        assert Path.exists(output_dir / "itu-performance.png")
        assert Path.exists(output_dir / "itu-pr.png")
        assert Path.exists(output_dir / "itu-roc.png")
        assert Path.exists(output_dir / "dialysis-performance.png")
        assert Path.exists(output_dir / "dialysis-pr.png")
        assert Path.exists(output_dir / "dialysis-roc.png")
