import os
from pathlib import Path
import json

import pandas as pd
import xlsxwriter
import pytest

from aki_predictions import file_operations


@pytest.fixture
def generate_xlsx(tmp_path):
    def _generate(name):
        path = Path(os.path.join(tmp_path, f"{name}.xlsx"))
        workbook = xlsxwriter.Workbook(path)
        workbook.add_worksheet()
        workbook.close()
        return path

    return _generate


class TestLoadXlsx:
    def test_load_xlsx_returns_data_frame_from_xlsx(self, generate_xlsx):
        assert isinstance(file_operations.load_xlsx(generate_xlsx("empty_sheet")), pd.DataFrame)


class TestLoadCSV:
    def test_load_csv_returns_dataframe(self):
        assert isinstance(
            file_operations.load_csv(Path(Path(os.getcwd()), "tests", "fixtures", "simple.csv")), pd.DataFrame
        )

    def test_data_frame_columns(self):
        assert file_operations.load_csv(
            Path(Path(os.getcwd()), "tests", "fixtures", "simple.csv")
        ).columns.values.tolist() == ["comma", "separated", "values"]


class TestLoadJson:
    def test_load_json_from_file(self):
        assert isinstance(file_operations.load_json(Path(Path(os.getcwd()), "tests", "fixtures", "simple.json")), dict)

    def test_loads_json(self):
        assert file_operations.load_json(Path(Path(os.getcwd()), "tests", "fixtures", "simple.json")) == {
            "example": "json file"
        }


class TestLoadJsonl:
    def test_load_json_from_file(self):
        assert isinstance(
            file_operations.load_jsonl(Path(Path(os.getcwd()), "tests", "fixtures", "simple.jsonl")), list
        )

    def test_loads_jsonl(self):
        assert file_operations.load_jsonl(Path(Path(os.getcwd()), "tests", "fixtures", "simple.jsonl")) == [
            {"example": "json file"},
            {"example2": "second line of jsonl file"},
        ]


class TestSaveDictionaryJson:
    def test_save_dictionary_json_saves_file(self, tmp_path):
        data = {"test_data": "json data"}
        path = Path(os.path.join(tmp_path), "test_json_data.json")
        file_operations.save_dictionary_json(path, data)
        assert Path.exists(path)

    def test_save_json_dictionary_contents(self, tmp_path):
        data = {"test_data": "json data"}
        path = Path(os.path.join(tmp_path), "test_json_data.json")
        file_operations.save_dictionary_json(path, data)
        with open(path, "r") as f:
            saved_json = json.loads(f.read())
        assert saved_json == data


class TestAppendJSONL:
    def test_append_jsonl_saves_file(self, tmp_path):
        data = {"test_data": "json data"}
        path = Path(os.path.join(tmp_path), "test_json_data.jsonl")
        file_operations.append_jsonl(path, data)
        assert Path.exists(path)

    def test_append_jsonl_contents(self, tmp_path):
        data = {"test_data": "json data"}
        path = Path(os.path.join(tmp_path), "test_json_data.jsonl")
        file_operations.append_jsonl(path, data)
        saved_data = []
        with open(path, "r") as f:
            for line in f:
                saved_data.append(json.loads(line))
        assert saved_data[0] == data

    def test_append_multi_lines(self, tmp_path):
        data1 = {"test_data": "json data"}
        data2 = {"test_data2": "json data2"}
        path = Path(os.path.join(tmp_path), "test_json_data.jsonl")
        file_operations.append_jsonl(path, data1)
        file_operations.append_jsonl(path, data2)
        saved_data = []
        with open(path, "r") as f:
            for line in f:
                saved_data.append(json.loads(line))
        assert saved_data == [data1, data2]
