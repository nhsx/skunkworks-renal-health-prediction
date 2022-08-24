from argparse import Namespace
import os
from pathlib import Path

import mock

import aki_predictions.__main__ as main_module
from aki_predictions.__main__ import parse_arguments, main


class TestParseArguments:
    def test_parse_args(self):
        parse_arguments([])

    def test_returns_namespace_object(self):
        assert isinstance(parse_arguments([]), Namespace)

    # Test ingest arguments
    def test_sets_ingest_flag(self):
        args = ["ingest"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "ingest"

    def test_sets_output_directory_short(self):
        args = ["ingest", "-o", "output_dir_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.output_dir == "output_dir_location"

    def test_sets_output_directory_long(self):
        args = ["ingest", "--output_dir", "output_dir_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.output_dir == "output_dir_location"

    def test_set_ingest_config(self):
        args = ["ingest", "-c", "ingest_config"]
        parsed_args = parse_arguments(args)
        assert parsed_args.config == "ingest_config"

    def test_sets_ingest_config_long(self):
        args = ["ingest", "--config", "ingest_config"]
        parsed_args = parse_arguments(args)
        assert parsed_args.config == "ingest_config"

    # Test survey arguments
    def test_sets_survey_flag(self):
        args = ["survey"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "survey"

    def test_sets_output_directory_short_survey(self):
        args = ["survey", "-o", "output_dir_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.output_dir == "output_dir_location"

    def test_sets_output_directory_long_survey(self):
        args = ["survey", "--output_dir", "output_dir_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.output_dir == "output_dir_location"

    def test_set_survey_data(self):
        args = ["survey", "-d", "data_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.data == "data_location"

    def test_sets_survey_data_long(self):
        args = ["survey", "--data", "data_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.data == "data_location"

    # Test training arguments
    def test_sets_training_flag(self):
        args = ["training"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "training"

    # Test training type
    def test_training_type_flag(self):
        args = ["training", "default"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "training"
        assert parsed_args.training_command == "default"

    def test_training_type_flag_hyperp(self):
        args = ["training", "hyperp"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "training"
        assert parsed_args.training_command == "hyperp"

    def test_training_type_flag_recurrent_cell(self):
        args = ["training", "recurrent_cell"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "training"
        assert parsed_args.training_command == "recurrent_cell"

    # Test default training args
    def test_sets_output_directory_short_training(self):
        args = ["training", "default", "-o", "output_dir_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.output_dir == "output_dir_location"

    def test_sets_output_directory_long_training(self):
        args = ["training", "default", "--output_dir", "output_dir_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.output_dir == "output_dir_location"

    def test_set_training_data(self):
        args = ["training", "default", "-d", "training_data_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.data == "training_data_location"

    def test_sets_training_data_long(self):
        args = ["training", "default", "-d", "training_data_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.data == "training_data_location"

    def test_set_training_steps(self):
        args = ["training", "default", "-s", "2"]
        parsed_args = parse_arguments(args)
        assert parsed_args.steps == 2

    def test_set_training_steps_different(self):
        args = ["training", "default", "-s", "200"]
        parsed_args = parse_arguments(args)
        assert parsed_args.steps == 200

    def test_sets_training_steps_long(self):
        args = ["training", "default", "--steps", "2"]
        parsed_args = parse_arguments(args)
        assert parsed_args.steps == 2

    def test_sets_training_context(self):
        args = ["training", "default"]
        parsed_args = parse_arguments(args)
        assert parsed_args.context is False

    def test_sets_training_context_true(self):
        args = ["training", "default", "-c", "True"]
        parsed_args = parse_arguments(args)
        assert parsed_args.context is True

    # Test evaluation arguments
    def test_sets_evaluation_flag(self):
        args = ["evaluation"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "evaluation"

    # Test evaluation options
    def test_evaluation_type_performance_evaluation(self):
        args = ["evaluation", "performance_evaluation"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "evaluation"
        assert parsed_args.evaluation_command == "performance_evaluation"

    def test_evaluation_type_flag_occlusion_processing(self):
        args = ["evaluation", "occlusion_processing"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "evaluation"
        assert parsed_args.evaluation_command == "occlusion_processing"

    def test_evaluation_type_flag_occlusion_reporting(self):
        args = ["evaluation", "occlusion_reporting"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "evaluation"
        assert parsed_args.evaluation_command == "occlusion_reporting"

    def test_evaluation_type_performance_comparison(self):
        args = ["evaluation", "performance_comparison"]
        parsed_args = parse_arguments(args)
        assert parsed_args.command == "evaluation"
        assert parsed_args.evaluation_command == "performance_comparison"

    # Test evaluation specific arguments
    def test_sets_output_directory_short_evaluation(self):
        args = ["evaluation", "performance_evaluation", "-o", "output_dir_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.output_dir == "output_dir_location"

    def test_sets_output_directory_long_evaluation(self):
        args = ["evaluation", "performance_evaluation", "--output_dir", "output_dir_location"]
        parsed_args = parse_arguments(args)
        assert parsed_args.output_dir == "output_dir_location"


class TestMain:
    def test_main_real_config(self, tmpdir):
        main(
            [
                "main",
                "ingest",
                "--output_dir",
                str(Path(tmpdir) / "test_main_log"),
                "--config",
                "tests/fixtures/mock_config.json",
            ]
        )
        # For debugging manually uncomment these lines instead of the above call.
        # main(
        #     [
        #         "main",
        #         "ingest",
        #         "--output_dir",
        #         str(Path("none") / "test_main_log"),
        #         "--config",
        #         "tests/fixtures/mock_config.json",
        #     ]
        # )

    def test_calls_parse_args(self, tmpdir):
        with mock.patch.object(main_module, "run_ingest") and mock.patch.object(
            main_module, "run_normalise"
        ) and mock.patch.object(main_module, "run_split"):

            with mock.patch.object(main_module, "parse_arguments") as mock_parse:
                with mock.patch.object(main_module, "setup_logging"):
                    args = ["main", "ingest", "--output_dir", str(Path(tmpdir) / "test_main_log")]
                    main(args)
                    mock_parse.assert_called_with(["ingest", "--output_dir", str(Path(tmpdir) / "test_main_log")])

    def test_generates_log_file(self, tmpdir):
        with mock.patch.object(main_module, "run_ingest") and mock.patch.object(
            main_module, "run_normalise"
        ) and mock.patch.object(main_module, "run_split"):

            args = [
                "main",
                "ingest",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--config",
                "tests/fixtures/mock_config.json",
            ]
            main(args)
            file_names = [
                filename
                for filename in os.listdir(Path(tmpdir) / "test_log")
                if os.path.isfile(Path(Path(tmpdir) / "test_log" / filename))
            ]
            assert "_aki_predictions_tool_log.txt" in file_names[0]

    def test_runs_ingest_with_config_and_paths(self, tmpdir):
        with mock.patch.object(main_module, "run_ingest") as mock_run_ingest:
            with mock.patch.object(main_module, "run_normalise") as mock_run_normalise:
                with mock.patch.object(main_module, "run_split") as mock_run_split:
                    args = [
                        "main",
                        "ingest",
                        "--output_dir",
                        str(Path(tmpdir) / "test_log"),
                        "--config",
                        "mock_config_path",
                    ]
                    main(args)
                    mock_run_ingest.assert_called_with(str(Path(tmpdir) / "test_log"), "mock_config_path")
                    mock_run_normalise.assert_called_with(mock_run_ingest.return_value)
                    mock_run_split.assert_called_with(mock_run_ingest.return_value)

    def test_runs_training_with_correct_arguments(self, tmpdir):
        with mock.patch.object(main_module, "run_training") as mock_run_training:
            args = [
                "main",
                "training",
                "default",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--data",
                "mock_data_path",
                "--steps",
                "200",
            ]
            main(args)
            mock_run_training.assert_called_with(
                str(Path(tmpdir) / "test_log"), "mock_data_path", 200, None, None, None
            )

    def test_runs_training_with_correct_arguments_with_context(self, tmpdir):
        with mock.patch.object(main_module, "run_training_with_context") as mock_run_training:
            args = [
                "main",
                "training",
                "default",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--data",
                "mock_data_path",
                "--steps",
                "200",
                "--context",
                "True",
            ]
            main(args)
            mock_run_training.assert_called_with(
                str(Path(tmpdir) / "test_log"), "mock_data_path", 200, None, None, None
            )

    def test_runs_training_with_correct_arguments_hyper_parameter_scan(self, tmpdir):
        with mock.patch.object(main_module, "run_hyper_parameter_scan") as mock_run_training:
            args = [
                "main",
                "training",
                "hyperp",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--data",
                "mock_data_path",
                "--steps",
                "200",
            ]
            main(args)
            mock_run_training.assert_called_with(str(Path(tmpdir) / "test_log"), "mock_data_path", 200)

    def test_runs_training_with_correct_arguments_recurrent_cell(self, tmpdir):
        with mock.patch.object(main_module, "run_recurrent_cell_scan") as mock_run_training:
            args = [
                "main",
                "training",
                "recurrent_cell",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--data",
                "mock_data_path",
                "--steps",
                "200",
            ]
            main(args)
            mock_run_training.assert_called_with(str(Path(tmpdir) / "test_log"), "mock_data_path", 200)

    def test_runs_data_survey_with_correct_arguments(self, tmpdir):
        with mock.patch.object(main_module, "run_survey") as mock_run_survey:
            args = [
                "main",
                "survey",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--data",
                "mock_data_path",
            ]
            main(args)
            mock_run_survey.assert_called_with(str(Path(tmpdir) / "test_log"), "mock_data_path")

    def test_runs_performance_evaluation_with_correct_arguments(self, tmpdir):
        with mock.patch.object(main_module, "run_performance_analysis") as mock_run_performance_analysis:
            args = [
                "main",
                "evaluation",
                "performance_evaluation",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--metrics",
                "mock_metrics",
            ]
            main(args)
            mock_run_performance_analysis.assert_called_with(str(Path(tmpdir) / "test_log"), "mock_metrics")

    def test_runs_occlusion_processing_with_correct_arguments(self, tmpdir):
        with mock.patch.object(main_module, "run_occlusion_processing") as mock_run_occlusion_processing:
            with mock.patch.object(main_module, "run_occlusion_reporting") as mock_run_occlusion_reporting:
                mock_run_occlusion_processing.return_value = ("feature_file", "metrics_file")
                args = [
                    "main",
                    "evaluation",
                    "occlusion_processing",
                    "--output_dir",
                    str(Path(tmpdir) / "test_log"),
                    "--data",
                    "mock_data_path",
                    "--steps",
                    "200",
                ]
                main(args)
                mock_run_occlusion_processing.assert_called_with(str(Path(tmpdir) / "test_log"), "mock_data_path", 200)
                mock_run_occlusion_reporting.assert_called_with(
                    str(Path(tmpdir) / "test_log"),
                    mock_run_occlusion_processing.return_value[0],
                    mock_run_occlusion_processing.return_value[1],
                    None,
                )

    def test_runs_occlusion_processing_with_correct_arguments_with_context(self, tmpdir):
        with mock.patch.object(main_module, "run_occlusion_processing_context") as mock_run_occlusion_processing:
            with mock.patch.object(main_module, "run_occlusion_reporting") as mock_run_occlusion_reporting:
                mock_run_occlusion_processing.return_value = ("feature_file", "metadata_file", "metrics_file")
                args = [
                    "main",
                    "evaluation",
                    "occlusion_processing",
                    "--output_dir",
                    str(Path(tmpdir) / "test_log"),
                    "--data",
                    "mock_data_path",
                    "--steps",
                    "200",
                    "-c",
                    "True",
                ]
                main(args)
                mock_run_occlusion_processing.assert_called_with(str(Path(tmpdir) / "test_log"), "mock_data_path", 200)
                mock_run_occlusion_reporting.assert_called_with(
                    str(Path(tmpdir) / "test_log"),
                    mock_run_occlusion_processing.return_value[0],
                    mock_run_occlusion_processing.return_value[2],
                    mock_run_occlusion_processing.return_value[1],
                )

    def test_runs_occlusion_reporting_with_correct_arguments(self, tmpdir):
        with mock.patch.object(main_module, "run_occlusion_reporting") as mock_run_occlusion_reporting:
            args = [
                "main",
                "evaluation",
                "occlusion_reporting",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--feature",
                "mock_feature_mapping",
                "--metrics",
                "mock_metrics_file",
            ]
            main(args)
            mock_run_occlusion_reporting.assert_called_with(
                str(Path(tmpdir) / "test_log"), "mock_feature_mapping", "mock_metrics_file", None
            )

    def test_runs_occlusion_reporting_with_correct_arguments_with_context(self, tmpdir):
        with mock.patch.object(main_module, "run_occlusion_reporting") as mock_run_occlusion_reporting:
            args = [
                "main",
                "evaluation",
                "occlusion_reporting",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "--feature",
                "mock_feature_mapping",
                "--metrics",
                "mock_metrics_file",
                "--metadata",
                "mock_metadata_file",
                "-c",
                "True",
            ]
            main(args)
            mock_run_occlusion_reporting.assert_called_with(
                str(Path(tmpdir) / "test_log"), "mock_feature_mapping", "mock_metrics_file", "mock_metadata_file"
            )

    def test_runs_performance_comparison_with_correct_arguments(self, tmpdir):
        with mock.patch.object(main_module, "run_performance_comparison") as mock_run_performance_comparison:
            args = [
                "main",
                "evaluation",
                "performance_comparison",
                "--output_dir",
                str(Path(tmpdir) / "test_log"),
                "-a",
                "itu",
                "--metrics",
                "mock_metrics_1",
                "mock_metrics_2",
            ]
            main(args)
            mock_run_performance_comparison.assert_called_with(
                str(Path(tmpdir) / "test_log"), ["mock_metrics_1", "mock_metrics_2"], "itu"
            )
