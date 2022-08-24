import os
from pathlib import Path

import pandas as pd
import xlsxwriter
import pytest

from aki_predictions import data_loading


@pytest.fixture
def generate_xlsx(tmp_path):
    def _generate(name):
        path = Path(os.path.join(tmp_path, f"{name}.xlsx"))
        workbook = xlsxwriter.Workbook(path)
        workbook.add_worksheet()
        workbook.close()
        return path

    return _generate


class TestLoadRawData:
    def test_returns_dictionary(self):
        assert isinstance(data_loading.load_raw_data({}), dict)

    def test_assert_loads_data_entry(self):
        sources = [{"filepath": "tests/fixtures/simple.csv", "name": "simple"}]
        expected = pd.DataFrame(data={"comma": ["0"], "separated": ["1"], "values": ["2"]})
        data = data_loading.load_raw_data(sources)
        assert data["simple"].equals(expected)

    def test_assert_loads_multiple_data_entry(self):
        sources = [
            {"filepath": "tests/fixtures/simple.csv", "name": "simple"},
            {"filepath": "tests/fixtures/simple2.csv", "name": "simple2"},
        ]
        expected = {
            "simple": pd.DataFrame(data={"comma": ["0"], "separated": ["1"], "values": ["2"]}),
            "simple2": pd.DataFrame(data={"comma": ["3"], "separated": ["4"], "values": ["5"]}),
        }
        data = data_loading.load_raw_data(sources)
        assert data["simple"].equals(expected["simple"])
        assert data["simple2"].equals(expected["simple2"])

    def test_concatenates_columns(self):
        sources = [
            {
                "filepath": "tests/fixtures/simple.csv",
                "name": "simple",
                "concatenate_columns": [{"field": "output", "values": ["comma", "separated"], "seperator": "/"}],
            }
        ]
        expected = pd.DataFrame(data={"comma": ["0"], "separated": ["1"], "values": ["2"], "output": "0/1"})
        data = data_loading.load_raw_data(sources)
        assert data["simple"].equals(expected)

    def test_splits_columns(self):
        sources = [
            {
                "filepath": "tests/fixtures/simple3.csv",
                "name": "simple",
                "split_columns": [{"value": "comma", "fields": ["com", "ma"], "delimiter": "-"}],
            }
        ]
        expected = pd.DataFrame(data={"comma": ["3-6"], "separated": ["4"], "values": ["5"], "com": "3", "ma": "6"})
        data = data_loading.load_raw_data(sources)
        assert data["simple"].equals(expected)


class TestLoadSingleDataEntry:
    def test_load_single_data_entry(self):
        assert data_loading.load_single_data_entry({"filepath": ""}) is None

    def test_returns_data_frame_csv(self):
        config = {"filepath": "tests/fixtures/simple.csv"}
        assert isinstance(data_loading.load_single_data_entry(config), pd.DataFrame)

    def test_returns_data_frame_contents_csv(self):
        config = {"filepath": "tests/fixtures/simple.csv"}
        expected = pd.DataFrame(data={"comma": ["0"], "separated": ["1"], "values": ["2"]})
        assert data_loading.load_single_data_entry(config).equals(expected)

    def test_returns_data_frame_xlsx(self, generate_xlsx):
        filepath = generate_xlsx("empty_sheet")
        config = {"filepath": filepath}
        assert isinstance(data_loading.load_single_data_entry(config), pd.DataFrame)

    def test_reads_skip_rows_entry_for_xlsx(self, tmp_path):
        source_data = {"top_row": ["a", "b", "c"], "top_row2": ["d", "e", "f"]}
        data_frame = pd.DataFrame(data=source_data)
        filepath = Path(os.path.join(tmp_path, "skip_rows.xlsx"))
        data_frame.to_excel(filepath, index=False)

        expected_data = {"a": ["b", "c"], "d": ["e", "f"]}
        data_frame_expected = pd.DataFrame(data=expected_data)

        config = {"filepath": filepath, "skiprows": 1}
        result = data_loading.load_single_data_entry(config)
        print(result)
        print(data_frame)
        assert result.equals(data_frame_expected)

    def test_drop_columns(self):
        config = {"filepath": "tests/fixtures/simple.csv", "drop_columns": ["separated"]}
        expected = pd.DataFrame(data={"comma": ["0"], "values": ["2"]})
        assert data_loading.load_single_data_entry(config).equals(expected)


class TestLoadMultipleDataEntry:
    def test_load_single_data_entry(self):
        assert data_loading.load_multiple_data_entry({"filepath": ""}) == []

    def test_returns_data_frame_csv(self):
        config = {"filepath": ["tests/fixtures/simple.csv", "tests/fixtures/simple2.csv"]}
        assert isinstance(data_loading.load_multiple_data_entry(config), pd.DataFrame)

    def test_returns_data_frame_contents_multiple_csv(self):
        config = {"filepath": ["tests/fixtures/simple.csv", "tests/fixtures/simple2.csv"]}
        expected = pd.DataFrame(data={"comma": ["0", "3"], "separated": ["1", "4"], "values": ["2", "5"]})
        assert data_loading.load_multiple_data_entry(config).equals(expected)

    def test_returns_data_frame_mixed(self, generate_xlsx):
        filepath = generate_xlsx("empty_sheet")
        config = {"filepath": ["tests/fixtures/simple.csv", filepath]}
        assert isinstance(data_loading.load_multiple_data_entry(config), pd.DataFrame)

    def test_returns_data_frame_contents_mixed(self, tmp_path):
        source_data = {"x": ["comma", "3"], "y": ["separated", "4"], "z": ["values", "5"]}
        data_frame = pd.DataFrame(data=source_data)
        filepath = Path(os.path.join(tmp_path, "skip_rows.xlsx"))
        data_frame.to_excel(filepath, index=False)

        data_frame_expected = pd.DataFrame(data={"comma": ["0", "3"], "separated": ["1", "4"], "values": ["2", "5"]})

        config = {"filepath": ["tests/fixtures/simple.csv", filepath], "skiprows": 1}
        assert data_loading.load_multiple_data_entry(config).equals(data_frame_expected)

    def test_drop_columns(self):
        config = {
            "filepath": ["tests/fixtures/simple.csv", "tests/fixtures/simple2.csv"],
            "drop_columns": ["separated"],
        }
        data_frame_expected = pd.DataFrame(data={"comma": ["0", "3"], "values": ["2", "5"]})
        assert data_loading.load_multiple_data_entry(config).equals(data_frame_expected)


class TestConcatenateColumns:
    def test_returns_modified(self):
        source_data = {"x": ["1", "2"], "y": ["3", "4"], "z": ["5", "6"]}
        data_entry = {
            "name": "raw_data",
            "concatenate_columns": [{"field": "output", "values": ["x", "y"], "seperator": "/"}],
        }
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        data_frame_expected = pd.DataFrame(
            data={"x": ["1", "2"], "y": ["3", "4"], "z": ["5", "6"], "output": ["1/3", "2/4"]}
        )
        assert data_loading.concatenate_columns(data_entry, raw_data)["raw_data"].equals(data_frame_expected)

    def test_returns_modified_multi(self):
        source_data = {"x": ["1", "2"], "y": ["3", "4"], "z": ["5", "6"]}
        data_entry = {
            "name": "raw_data",
            "concatenate_columns": [
                {"field": "output", "values": ["x", "y"], "seperator": "/"},
                {"field": "output_2", "values": ["y", "z"], "seperator": "+"},
            ],
        }
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        data_frame_expected = pd.DataFrame(
            data={
                "x": ["1", "2"],
                "y": ["3", "4"],
                "z": ["5", "6"],
                "output": ["1/3", "2/4"],
                "output_2": ["3+5", "4+6"],
            }
        )
        assert data_loading.concatenate_columns(data_entry, raw_data)["raw_data"].equals(data_frame_expected)

    def test_returns_unmodified(self):
        source_data = {"x": ["1", "2"], "y": ["3", "4"], "z": ["5", "6"]}
        data_entry = {"name": "raw_data"}
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        assert data_loading.concatenate_columns(data_entry, raw_data)["raw_data"].equals(data_frame)


class TestSplitColumns:
    def test_returns_modified(self):
        source_data = {"x": ["1/True", "2/False"], "y": ["3/True", "4/False"], "z": ["5/True", "6/False"]}
        data_entry = {
            "name": "raw_data",
            "split_columns": [{"value": "x", "fields": ["output", "output_2"], "delimiter": "/"}],
        }
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        data_frame_expected = pd.DataFrame(
            data={
                "x": ["1/True", "2/False"],
                "y": ["3/True", "4/False"],
                "z": ["5/True", "6/False"],
                "output": ["1", "2"],
                "output_2": ["True", "False"],
            }
        )
        assert data_loading.split_columns(data_entry, raw_data)["raw_data"].equals(data_frame_expected)

    def test_returns_modified_multi(self):
        source_data = {"x": ["1/True", "2/False"], "y": ["3/True", "4/False"], "z": ["5/True", "6/False"]}
        data_entry = {
            "name": "raw_data",
            "split_columns": [
                {"value": "x", "fields": ["output", "output_2"], "delimiter": "/"},
                {"value": "y", "fields": ["output_3", "output_4"], "delimiter": "/"},
            ],
        }
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        data_frame_expected = pd.DataFrame(
            data={
                "x": ["1/True", "2/False"],
                "y": ["3/True", "4/False"],
                "z": ["5/True", "6/False"],
                "output": ["1", "2"],
                "output_2": ["True", "False"],
                "output_3": ["3", "4"],
                "output_4": ["True", "False"],
            }
        )
        assert data_loading.split_columns(data_entry, raw_data)["raw_data"].equals(data_frame_expected)

    def test_returns_unmodified(self):
        source_data = {"x": ["1/True", "2/False"], "y": ["3/True", "4/False"], "z": ["5/True", "6/False"]}
        data_entry = {"name": "raw_data"}
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        assert data_loading.concatenate_columns(data_entry, raw_data)["raw_data"].equals(data_frame)


class TestRemoveDuplicates:
    def test_returns_modified_all(self):
        source_data = {"x": ["1", "2", "2"], "y": ["3", "3", "4"], "z": ["3", "3", "4"], "i": ["1", "3", "7"]}
        data_entry = {"name": "raw_data", "remove_duplicates": ["y", "z"]}
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        data_frame_expected = pd.DataFrame(data={"x": ["2"], "y": ["4"], "z": ["4"], "i": ["7"]})
        assert data_loading.eval_duplicates(data_entry, raw_data, remove_all_dupe_rows=True)["raw_data"].equals(
            data_frame_expected
        )

    def test_returns_modified_keep(self):
        source_data = {
            "x": ["1", "1", "2", "2"],
            "y": ["3", "3", "3", "4"],
            "z": ["3", "3", "3", "4"],
            "i": ["1", "1", "3", "7"],
        }
        data_entry = {"name": "raw_data", "remove_duplicates": ["y", "z"]}
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        data_frame_expected = pd.DataFrame(data={"x": ["1", "2"], "y": ["3", "4"], "z": ["3", "4"], "i": ["1", "7"]})
        assert data_loading.eval_duplicates(data_entry, raw_data, remove_all_dupe_rows=False)["raw_data"].equals(
            data_frame_expected
        )

    def test_returns_unmodified(self):
        source_data = {"x": ["1/True", "2/False"], "y": ["3/True", "4/False"], "z": ["5/True", "6/False"]}
        data_entry = {"name": "raw_data"}
        data_frame = pd.DataFrame(data=source_data)
        raw_data = {"raw_data": data_frame}
        assert data_loading.eval_duplicates(data_entry, raw_data, remove_all_dupe_rows=False)["raw_data"].equals(
            data_frame
        )


class TestGroupEvents:
    def test_group_events_returns_list(self):
        assert isinstance(data_loading.group_events([]), list)

    def test_group_events_returns_same_event(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "5"}],
            }
        ]
        grouped_events = data_loading.group_events(events)
        assert grouped_events == events

    def test_group_events_returns_both_events(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "5"}],
            },
            {
                "patient_age": 2,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "4", "feature_idx": "5", "feature_value": "6"}],
            },
        ]
        grouped_events = data_loading.group_events(events)
        assert grouped_events == events

    def test_group_events_groups_both_events(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "5"}],
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "6", "feature_idx": "7", "feature_value": "8"}],
            },
        ]
        grouped_events = data_loading.group_events(events)
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [
                    {"feature_category_idx": "3", "feature_idx": "4", "feature_value": "5"},
                    {"feature_category_idx": "6", "feature_idx": "7", "feature_value": "8"},
                ],
            }
        ]
        assert grouped_events == expected

    def test_group_events_groups_both_events_and_sorts(self):
        events = [
            {
                "patient_age": 2,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "10", "feature_idx": "11", "feature_value": "12"}],
            },
            {
                "patient_age": 2,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "4", "feature_idx": "5", "feature_value": "6"}],
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "1", "feature_idx": "2", "feature_value": "3"}],
            },
            {
                "patient_age": 2,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "7", "feature_idx": "8", "feature_value": "9"}],
            },
        ]
        grouped_events = data_loading.group_events(events)
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [
                    {"feature_category_idx": "1", "feature_idx": "2", "feature_value": "3"},
                ],
            },
            {
                "patient_age": 2,
                "time_of_day": 1,
                "entries": [
                    {"feature_category_idx": "4", "feature_idx": "5", "feature_value": "6"},
                    {"feature_category_idx": "7", "feature_idx": "8", "feature_value": "9"},
                ],
            },
            {
                "patient_age": 2,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "10", "feature_idx": "11", "feature_value": "12"}],
            },
        ]
        assert grouped_events == expected


class TestLabelEvents:
    def test_returns_events_with_labels(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_1h": "0",
                    "adverse_outcome_outcome_within_2h": "0",
                    "adverse_outcome_outcome_within_3h": "0",
                    "adverse_outcome_outcome_within_4h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            }
        ]
        record = {"202": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1}
        feature_mapping = {"outcome_field_outcome_value": 4}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "1",
                "max_look_ahead": "4",
                "labels": {"outcome": {"name": "outcome_field", "values": ["outcome_value"]}},
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_1h": "0",
            "adverse_outcome_outcome_within_2h": "0",
            "adverse_outcome_outcome_within_3h": "0",
            "adverse_outcome_outcome_within_4h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [1, 2, 3, 4]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_1h": "1",
                    "adverse_outcome_outcome_within_2h": "1",
                    "adverse_outcome_outcome_within_3h": "1",
                    "adverse_outcome_outcome_within_4h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            }
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_lookahead_longer_than_one_day(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_1h": "0",
                    "adverse_outcome_outcome_within_2h": "0",
                    "adverse_outcome_outcome_within_3h": "0",
                    "adverse_outcome_outcome_within_4h": "0",
                    "adverse_outcome_outcome_within_5h": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_7h": "0",
                    "adverse_outcome_outcome_within_8h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            }
        ]
        record = {"202": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1}
        feature_mapping = {"outcome_field_outcome_value": 4}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "1",
                "max_look_ahead": "8",
                "labels": {"outcome": {"name": "outcome_field", "values": ["outcome_value"]}},
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_1h": "0",
            "adverse_outcome_outcome_within_2h": "0",
            "adverse_outcome_outcome_within_3h": "0",
            "adverse_outcome_outcome_within_4h": "0",
            "adverse_outcome_outcome_within_5h": "0",
            "adverse_outcome_outcome_within_6h": "0",
            "adverse_outcome_outcome_within_7h": "0",
            "adverse_outcome_outcome_within_8h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [1, 2, 3, 4, 5, 6, 7, 8]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_1h": "1",
                    "adverse_outcome_outcome_within_2h": "1",
                    "adverse_outcome_outcome_within_3h": "1",
                    "adverse_outcome_outcome_within_4h": "1",
                    "adverse_outcome_outcome_within_5h": "1",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_7h": "1",
                    "adverse_outcome_outcome_within_8h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            }
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_with_metadata(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_1h": "0",
                    "adverse_outcome_outcome_within_2h": "0",
                    "adverse_outcome_outcome_within_3h": "0",
                    "adverse_outcome_outcome_within_4h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            }
        ]
        record = {"202": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1}
        feature_mapping = {"outcome_field_outcome_value": 4}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "1",
                "max_look_ahead": "4",
                "labels": {"outcome": {"name": "outcome_field", "values": ["outcome_value"], "metadata": ["X5"]}},
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_1h": "0",
            "adverse_outcome_outcome_within_2h": "0",
            "adverse_outcome_outcome_within_3h": "0",
            "adverse_outcome_outcome_within_4h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [1, 2, 3, 4]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_1h": "1",
                    "adverse_outcome_outcome_within_2h": "1",
                    "adverse_outcome_outcome_within_3h": "1",
                    "adverse_outcome_outcome_within_4h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            }
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_with_metadata_false(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_1h": "0",
                    "adverse_outcome_outcome_within_2h": "0",
                    "adverse_outcome_outcome_within_3h": "0",
                    "adverse_outcome_outcome_within_4h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            }
        ]
        record = {"200": "1", "201": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1}
        feature_mapping = {"outcome_field_outcome_value": 4}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "1",
                "max_look_ahead": "4",
                "labels": {"outcome": {"name": "outcome_field", "values": ["outcome_value"], "metadata": ["X5"]}},
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_1h": "0",
            "adverse_outcome_outcome_within_2h": "0",
            "adverse_outcome_outcome_within_3h": "0",
            "adverse_outcome_outcome_within_4h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [1, 2, 3, 4]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_1h": "0",
                    "adverse_outcome_outcome_within_2h": "0",
                    "adverse_outcome_outcome_within_3h": "0",
                    "adverse_outcome_outcome_within_4h": "0",
                    "adverse_outcome_in_spell": "0",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            }
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_with_single_lookahead(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
        ]
        record = {"202": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1}
        feature_mapping = {"outcome_field_outcome_value": 4}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "6",
                "max_look_ahead": "4",
                "labels": {"outcome": {"name": "outcome_field", "values": ["outcome_value"], "metadata": ["X5"]}},
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_6h": "0",
            "adverse_outcome_outcome_within_12h": "0",
            "adverse_outcome_outcome_within_18h": "0",
            "adverse_outcome_outcome_within_24h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [6, 12, 18, 24]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_with_single_lookahead_with_outcome_in_additional_bin(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 4,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
        ]
        record = {"202": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1}
        feature_mapping = {"outcome_field_outcome_value": 4}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "6",
                "max_look_ahead": "4",
                "labels": {"outcome": {"name": "outcome_field", "values": ["outcome_value"], "metadata": ["X5"]}},
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_6h": "0",
            "adverse_outcome_outcome_within_12h": "0",
            "adverse_outcome_outcome_within_18h": "0",
            "adverse_outcome_outcome_within_24h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [6, 12, 18, 24]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 4,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_with_next_lookahead(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
        ]
        record = {"202": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1}
        feature_mapping = {"outcome_field_outcome_value": 4}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "6",
                "max_look_ahead": "4",
                "labels": {"outcome": {"name": "outcome_field", "values": ["outcome_value"], "metadata": ["X5"]}},
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_6h": "0",
            "adverse_outcome_outcome_within_12h": "0",
            "adverse_outcome_outcome_within_18h": "0",
            "adverse_outcome_outcome_within_24h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [6, 12, 18, 24]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_with_multiple_lookahead(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "7", "feature_idx": "8", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "8", "feature_idx": "9", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "9", "feature_idx": "10", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 2,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
        ]
        record = {"202": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1}
        feature_mapping = {"outcome_field_outcome_value": 4}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "6",
                "max_look_ahead": "4",
                "labels": {"outcome": {"name": "outcome_field", "values": ["outcome_value"], "metadata": ["X5"]}},
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_6h": "0",
            "adverse_outcome_outcome_within_12h": "0",
            "adverse_outcome_outcome_within_18h": "0",
            "adverse_outcome_outcome_within_24h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [6, 12, 18, 24]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "7", "feature_idx": "8", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "8", "feature_idx": "9", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "9", "feature_idx": "10", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 2,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_with_multiple_outcomes(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "7", "feature_idx": "8", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "8", "feature_idx": "9", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {  # Outcome 1
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "9", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 4,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {  # Outcome 2
                "patient_age": 2,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "20", "feature_value": "outcome_2_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 2,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
        ]
        record = {"202": "1", "201": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1, "outcome_2": 2}
        feature_mapping = {"outcome_field_outcome_value": 4, "outcome_2_field_outcome_2_value": 20}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "6",
                "max_look_ahead": "4",
                "labels": {
                    "outcome": {"name": "outcome_field", "values": ["outcome_value"], "metadata": ["X5"]},
                    "outcome_2": {"name": "outcome_2_field", "values": ["outcome_2_value"], "metadata": ["X4"]},
                },
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_6h": "0",
            "adverse_outcome_outcome_within_12h": "0",
            "adverse_outcome_outcome_within_18h": "0",
            "adverse_outcome_outcome_within_24h": "0",
            "adverse_outcome_outcome_2_within_6h": "0",
            "adverse_outcome_outcome_2_within_12h": "0",
            "adverse_outcome_outcome_2_within_18h": "0",
            "adverse_outcome_outcome_2_within_24h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [6, 12, 18, 24]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "7", "feature_idx": "8", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "1",
                    "adverse_outcome_outcome_2_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "8", "feature_idx": "9", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "1",
                    "adverse_outcome_outcome_2_within_18h": "1",
                    "adverse_outcome_outcome_2_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {  # Outcome 1
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "9", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "1",
                    "adverse_outcome_outcome_2_within_12h": "1",
                    "adverse_outcome_outcome_2_within_18h": "1",
                    "adverse_outcome_outcome_2_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 4,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "1",
                    "adverse_outcome_outcome_2_within_12h": "1",
                    "adverse_outcome_outcome_2_within_18h": "1",
                    "adverse_outcome_outcome_2_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
            {  # Outcome 2
                "patient_age": 2,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "20", "feature_value": "outcome_2_value"}],
                "labels": {
                    "adverse_outcome": "2",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "1",
                    "adverse_outcome_outcome_2_within_12h": "1",
                    "adverse_outcome_outcome_2_within_18h": "1",
                    "adverse_outcome_outcome_2_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 2,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "2",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "1",
                    "adverse_outcome_outcome_2_within_12h": "1",
                    "adverse_outcome_outcome_2_within_18h": "1",
                    "adverse_outcome_outcome_2_within_24h": "1",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events

    def test_returns_events_with_labels_with_multiple_outcomes_only_one_outcome_positive(self):
        events = [
            {
                "patient_age": 1,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "7", "feature_idx": "8", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "8", "feature_idx": "9", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
            {  # Outcome 1
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "9", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "0",
                    "adverse_outcome_outcome_within_24h": "0",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "0",
                },
                "numerical_labels": {},
            },
        ]
        record = {"202": "1", "201": "1", "episodes": [{"events": events}]}
        label_mapping = {"outcome": 1, "outcome_2": 2}
        feature_mapping = {"outcome_field_outcome_value": 4, "outcome_2_field_outcome_2_value": 20}
        metadata_mapping = {"diagnosis_X3": 200, "diagnosis_X4": 201, "diagnosis_X5": 202}
        label_config = {
            "of_interest": "outcome",
            "adverse_outcome": {
                "time_step": "6",
                "max_look_ahead": "4",
                "labels": {
                    "outcome": {"name": "outcome_field", "values": ["outcome_value"], "metadata": ["X5"]},
                    "outcome_2": {"name": "outcome_2_field", "values": ["outcome_2_value"], "metadata": ["X4"]},
                },
            },
        }
        label_template = {
            "adverse_outcome": "0",
            "adverse_outcome_outcome_within_6h": "0",
            "adverse_outcome_outcome_within_12h": "0",
            "adverse_outcome_outcome_within_18h": "0",
            "adverse_outcome_outcome_within_24h": "0",
            "adverse_outcome_outcome_2_within_6h": "0",
            "adverse_outcome_outcome_2_within_12h": "0",
            "adverse_outcome_outcome_2_within_18h": "0",
            "adverse_outcome_outcome_2_within_24h": "0",
            "adverse_outcome_in_spell": "0",
            "segment_mask": "0",
        }
        window_times = [6, 12, 18, 24]
        expected = [
            {
                "patient_age": 1,
                "time_of_day": 0,
                "entries": [{"feature_category_idx": "3", "feature_idx": "5", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "0",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 1,
                "entries": [{"feature_category_idx": "7", "feature_idx": "8", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "0",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {
                "patient_age": 1,
                "time_of_day": 2,
                "entries": [{"feature_category_idx": "8", "feature_idx": "9", "feature_value": "normal"}],
                "labels": {
                    "adverse_outcome": "0",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "0",
                },
                "numerical_labels": {},
            },
            {  # Outcome 1
                "patient_age": 1,
                "time_of_day": 3,
                "entries": [{"feature_category_idx": "9", "feature_idx": "4", "feature_value": "outcome_value"}],
                "labels": {
                    "adverse_outcome": "1",
                    "adverse_outcome_outcome_within_6h": "1",
                    "adverse_outcome_outcome_within_12h": "1",
                    "adverse_outcome_outcome_within_18h": "1",
                    "adverse_outcome_outcome_within_24h": "1",
                    "adverse_outcome_outcome_2_within_6h": "0",
                    "adverse_outcome_outcome_2_within_12h": "0",
                    "adverse_outcome_outcome_2_within_18h": "0",
                    "adverse_outcome_outcome_2_within_24h": "0",
                    "adverse_outcome_in_spell": "1",
                    "segment_mask": "1",
                },
                "numerical_labels": {},
            },
        ]
        labelled_events, _ = data_loading.label_events(
            events,
            label_config,
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            {},
            record,
            window_times,
        )
        assert expected == labelled_events


class TestIdentifyNextTimeBin:
    def test_returns_tuple(self):
        assert isinstance(data_loading.identify_next_time_bin(0, 0, 0), tuple)

    @pytest.mark.parametrize(
        "day, bin, num_bins, expected",
        [
            (1, 0, 2, (1, 1, 2, False)),
            (1, 1, 2, (2, 0, 2, True)),
            (1, 2, 2, (2, 0, 2, True)),
            (2, 0, 4, (2, 1, 4, False)),
            (2, 3, 4, (3, 0, 4, True)),
        ],
    )
    def test_returns_next_bin(self, day, bin, num_bins, expected):
        assert data_loading.identify_next_time_bin(day, bin, num_bins) == expected


class TestDateTimeBins:
    @pytest.mark.parametrize(
        "date_time, bin, expected_bin",
        [
            ("21/5/1995 12:30:00", 2, 1),
            ("21/5/1995 15:33:00", 4, 2),
            ("21/5/1995 2:00:00", 5, 0),
            ("21/5/1995 7:10:00", 24, 7),
            ("21/5/1995 00:00:00", 24, 24),
            ("21/5/1995 23:00:00", 24, 23),
            ("21/5/1995 23:30:00", 24, 23),
            ("21/5/1995 00:10:00", 24, 0),
            ("21/5/1995 01:10:00", 3, 0),
            ("21/5/1995 01:10:00", 24, 1),
            #  Test single hours
            ("21/5/1995 00:01:00", 24, 0),
            ("21/5/1995 01:01:00", 24, 1),
            ("21/5/1995 02:01:00", 24, 2),
            ("21/5/1995 03:01:00", 24, 3),
            ("21/5/1995 04:01:00", 24, 4),
            ("21/5/1995 05:01:00", 24, 5),
            ("21/5/1995 06:01:00", 24, 6),
            ("21/5/1995 07:01:00", 24, 7),
            ("21/5/1995 08:01:00", 24, 8),
            ("21/5/1995 09:01:00", 24, 9),
            ("21/5/1995 10:01:00", 24, 10),
            ("21/5/1995 11:01:00", 24, 11),
            ("21/5/1995 12:01:00", 24, 12),
            ("21/5/1995 13:01:00", 24, 13),
            ("21/5/1995 14:01:00", 24, 14),
            ("21/5/1995 15:01:00", 24, 15),
            ("21/5/1995 16:01:00", 24, 16),
            ("21/5/1995 17:01:00", 24, 17),
            ("21/5/1995 18:01:00", 24, 18),
            ("21/5/1995 19:01:00", 24, 19),
            ("21/5/1995 20:01:00", 24, 20),
            ("21/5/1995 21:01:00", 24, 21),
            ("21/5/1995 22:01:00", 24, 22),
            ("21/5/1995 23:01:00", 24, 23),
            ("21/5/1995 23:59:00", 24, 23),
            #  Test 6hourly
            ("21/5/1995 00:01:00", 4, 0),
            ("21/5/1995 05:59:00", 4, 0),
            ("21/5/1995 06:01:00", 4, 1),
            ("21/5/1995 11:59:00", 4, 1),
            ("21/5/1995 12:01:00", 4, 2),
            ("21/5/1995 17:59:00", 4, 2),
            ("21/5/1995 18:01:00", 4, 3),
            ("21/5/1995 23:59:00", 4, 3),
            ("21/5/1995 00:00:00", 4, 4),
        ],
    )
    def test_returns_bin(self, date_time, bin, expected_bin):
        assert data_loading.datetime_bins(date_time, bin) == expected_bin


class TestRelativeDate:
    @pytest.mark.parametrize(
        "year_birth, date_event, expected_days",
        [("1995", "1/1/1996 12:30:00", 365), ("1995", "30/1/1997 12:30:00", 760), ("1995", "1/3/2005 12:30:00", 3712)],
    )
    def test_returns_days(self, year_birth, date_event, expected_days):
        assert data_loading.relative_date(year_birth, date_event) == expected_days
