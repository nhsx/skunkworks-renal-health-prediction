import shutil
from pathlib import Path

from aki_predictions.data_processing.normalise_data import main as normalise_main


class TestNormaliseDataMain:
    def test_main_function(self, tmpdir):
        output_dir = Path(str(tmpdir))
        # Make fake debug directory
        Path.mkdir(output_dir / "debug")

        # Copy across ingest output files
        file_names = ["numerical_feature_mapping.json", "feature_statistics.json", "ingest_records_output_lines.jsonl"]
        for file_name in file_names:
            shutil.copy(Path("tests/fixtures/test_data_ingest_output") / file_name, output_dir)

        # Run normalisation
        normalise_main(output_dir)

        # Check for mappings
        assert Path.exists(Path(output_dir) / "numerical_feature_mapping.json")
        assert Path.exists(Path(output_dir) / "feature_statistics.json")
        # Check for main output file.
        assert Path.exists(Path(output_dir) / "ingest_records_output_lines.jsonl")

        # Check for new files
        assert Path.exists(Path(output_dir) / "ingest_records_output_lines_normalised_uncapped.jsonl")
