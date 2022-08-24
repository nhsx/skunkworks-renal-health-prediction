import sys
from pathlib import Path

from tqdm import tqdm

from aki_predictions.file_operations import load_xlsx, load_json


def main(args):
    # Load data index
    data_index = load_json(Path(args[0]))

    for data_entry in data_index["sources"]:
        for filepath in tqdm(data_entry["filepath"]):
            skiprows = data_entry["skiprows"] if "skiprows" in data_entry else 0
            data_tables = load_xlsx(filepath, sheet_name=None, skiprows=int(skiprows), dtype=str)
            for key, table in tqdm(data_tables.items()):
                if "Report" not in key:
                    print(table.info())
                    filepath_stem = Path(filepath).stem
                    path = Path(args[0]).parent / f"{filepath_stem}_{key}.csv"
                    print(path)
                    table.to_csv(path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
