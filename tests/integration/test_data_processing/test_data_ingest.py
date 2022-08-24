from pathlib import Path

from aki_predictions.data_processing.ingest_data import main as ingest_main


class TestIngestDataMain:
    def test_main_function(self, tmpdir):
        output_dir = ingest_main(str(Path(tmpdir) / "test_main_log"), "tests/fixtures/mock_config.json")

        # Check output directory exists
        assert Path.exists(output_dir)

        # Check that version files are present
        assert Path.exists(Path(output_dir) / "REVISION.txt")
        assert Path.exists(Path(output_dir) / "mock_config.json")

        # Check for master key entries
        assert Path.exists(Path(output_dir) / "master_keys.csv")
        assert Path.exists(Path(output_dir) / "master_keys_filtered.csv")

        # Check for mappings
        assert Path.exists(Path(output_dir) / "feature_mapping.json")
        assert Path.exists(Path(output_dir) / "numerical_feature_mapping.json")
        assert Path.exists(Path(output_dir) / "category_mapping.json")
        assert Path.exists(Path(output_dir) / "metadata_mapping.json")
        assert Path.exists(Path(output_dir) / "label_mapping.json")

        # Check for main output file.
        assert Path.exists(Path(output_dir) / "ingest_records_output_lines.jsonl")

        assert Path.exists(Path(output_dir) / "debug" / "numerical_feature_invalid.json")
        assert Path.exists(Path(output_dir) / "feature_statistics.json")
