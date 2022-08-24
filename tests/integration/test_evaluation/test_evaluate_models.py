from pathlib import Path
import os

from aki_predictions.evaluation.evaluate_models import main as evaluate_main


class TestEvaluateModelsMain:
    def test_main_function(self, tmpdir):
        output_dir = Path(tmpdir) / "test_main_log"

        if not Path.exists(output_dir):
            Path.mkdir(output_dir)

        evaluate_main(str(output_dir), [Path("tests") / "fixtures" / "mock_metrics.json"])

        output_files = os.listdir(output_dir)

        assert "_performance_comparison_scoring.json" in output_files[0]
