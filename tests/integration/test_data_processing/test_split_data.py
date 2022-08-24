import shutil
from pathlib import Path

import jsonlines

from aki_predictions.data_processing.split_data import main as split_main
from aki_predictions.file_operations import load_jsonl


class TestSplitDataMain:
    def test_main_function(self, tmpdir):
        output_dir = Path(str(tmpdir))
        # Make fake debug directory
        Path.mkdir(output_dir / "debug")

        ingest_file_path = output_dir / "ingest_records_output_lines_normalised_uncapped.jsonl"

        # Copy across ingest output files (assume pre-normalised - not required for this test)
        shutil.copy(
            Path("tests/fixtures/test_data_ingest_output/ingest_records_output_lines.jsonl"),
            ingest_file_path,
        )

        # Make ingest file long enough to get non-zero length output files for small splits
        ingest_jsonl = load_jsonl(ingest_file_path)
        with jsonlines.open(
            ingest_file_path,
            "a",
        ) as f:
            for _ in range(99):
                f.write(ingest_jsonl[0])

        # Run normalisation
        split_main(output_dir, splits=[0.8, 0.1, 0.05, 0.05])

        # Check for new files
        assert Path.exists(Path(output_dir) / "ingest_records_output_lines_train_uncapped.jsonl")
        assert Path.exists(Path(output_dir) / "ingest_records_output_lines_validate_uncapped.jsonl")
        assert Path.exists(Path(output_dir) / "ingest_records_output_lines_calib_uncapped.jsonl")
        assert Path.exists(Path(output_dir) / "ingest_records_output_lines_test_uncapped.jsonl")
