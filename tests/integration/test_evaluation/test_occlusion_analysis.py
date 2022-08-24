from pathlib import Path

from aki_predictions.file_operations import load_json
from aki_predictions.evaluation.occlusion_analysis import main as occlusion_main


class TestMain:
    def test_occlusion_analysis_without_context_main(self, tmpdir):
        output_dir = tmpdir
        feature_mapping_path = str(Path("tests") / "fixtures" / "test_data_ingest_output" / "feature_mapping.json")
        occlusion_output_path = str(Path("tests") / "fixtures" / "test_data_ingest_output" / "occlusion_analysis.jsonl")

        expected = {"unoccluded_unoccluded": 0.0, "sequence_feature": 2.0}

        occlusion_main(output_dir, feature_mapping_path, occlusion_output_path)

        output = load_json(Path(tmpdir) / "occlusion_analysis_output_mapped.json")

        assert output == expected

    def test_occlusion_analysis_with_context_main(self, tmpdir):
        output_dir = tmpdir
        feature_mapping_path = str(Path("tests") / "fixtures" / "test_data_ingest_output" / "feature_mapping.json")
        metadata_mapping_path = str(Path("tests") / "fixtures" / "test_data_ingest_output" / "metadata_mapping.json")
        occlusion_output_path = str(
            Path("tests") / "fixtures" / "test_data_ingest_output" / "occlusion_analysis_with_context.jsonl"
        )

        expected = {"unoccluded_unoccluded": 0.0, "sequence_feature": 2.0, "context_year_of_birth": 5.0}

        occlusion_main(output_dir, feature_mapping_path, occlusion_output_path, metadata_mapping_path)

        output = load_json(Path(tmpdir) / "occlusion_analysis_output_mapped.json")

        assert output == expected
