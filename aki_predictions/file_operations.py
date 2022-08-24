import json
import csv

import pandas as pd
import jsonlines
from tqdm import tqdm


def load_xlsx(*args, **kwargs):
    """Wrapper for pd.read_excel to load excel file from provided Path."""
    return pd.read_excel(*args, **kwargs)


def load_csv(*args, **kwargs):
    """Wrapper for pd.read_csv to load csv file from provided Path."""
    return pd.read_csv(*args, **kwargs)


def load_json(path):
    """Load json from file.

    Args:
        path (Path or str): Path to file.

    Returns:
        (dict): dictionary of loaded json data.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    """Load jsonl from file into dictionary.

    Args:
        path (Path or str): Path to file.

    Returns:
        (list of dict): list of dictionary of loaded jsonl data.
    """
    with open(path, "r") as f:
        return [json.loads(line) for line in tqdm(f.read().splitlines())]


def save_dictionary_json(path, data, sort_keys=True):
    """Save python dictionary to json file.

    Contents of the file must be serialisable as a json.

    Args:
        path (Path): path of file to save
        data (dict): data dictionary
    """
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=sort_keys, indent=4)


def save_csv(path, data, header):
    """Save data to csv.

    Args:
        path (Path): path of file to save
        data (list): list of lists (rows, [values])
        header (list): list of headings
    """
    with open(path, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for value in data:
            writer.writerow(value)


def append_jsonl(path, data):
    """Save line to jsonl file.

    Appends to existing file if present.

    Args:
        path (Path): path of file to save
        data (dict): data as dictionary
    """
    with jsonlines.open(
        path,
        "a",
    ) as f:
        f.write(data)
