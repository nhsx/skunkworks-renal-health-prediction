# Script to load a json description of a dataset and save a json file containing
# an extract all of the field names, and any additional attributes desired.
#
# Expects data index of form:
#   {
#    "files": [
#        {
#            "filepath": "<data path>",
#            "skiprows": <number of rows to skip> (excel only)
#            "relevant_fields": [ (this will provide output of unique values from the dataset in that field)
#                {
#                    "name": "<field name>",
#                    "split": "," (optional split for CSV cells)
#                    "remove_chars": "|" (remove set of characters from the string.
#                                         NOTE: this is done before splitting CSV strings)
#                }
#            ]
#        }
#        ... (can be list of files)
#    ]
#   }
#
# Provide data index file path via CLI argument on execution.
#
import sys
import re
from pathlib import Path
from collections import Iterable

import pandas as pd

from aki_predictions.file_operations import load_json, load_xlsx, load_csv, save_dictionary_json


# https://stackoverflow.com/a/17865033
def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


def main(args):
    # Load data index
    data_index = load_json(Path(args[0]))
    print(f"Processing: {Path(args[0])}")
    print(f"Index: {data_index}")

    survey_data = {"files": []}
    # Load data files
    for data_entry in data_index["files"]:
        # Load data file
        if Path(data_entry["filepath"]).suffix == ".xlsx":
            data_table = load_xlsx(data_entry["filepath"], skiprows=int(data_entry["skiprows"]))
        elif Path(data_entry["filepath"]).suffix == ".csv":
            data_table = load_csv(data_entry["filepath"])

        # Print data file info
        print(f"Surveying: {data_entry['filepath']}")
        print(data_table.info())

        # Create schema information from data
        schema = pd.io.json.build_table_schema(data_table)

        survey = {}
        # Extract relevant field information
        survey["relevant_fields"] = []
        if "relevant_fields" in data_entry:
            # Extract column for specific field
            for field in data_entry["relevant_fields"]:
                # If csv then extract values into list
                if "remove_chars" in field:
                    data_table[field["name"]] = data_table[field["name"]].apply(
                        lambda x: re.sub(field["remove_chars"], "", str(x))
                    )
                if "split" in field:
                    # Split on single character
                    if len(field["split"]) == 1:
                        data_table[field["name"]] = data_table[field["name"]].apply(
                            lambda x: x.split(field["split"]) if field["split"] in x else x
                        )
                    # Split on regex
                    else:
                        pass
                        # data_table[field["name"]] = data_table[field["name"]].apply(
                        #     lambda x: re.split(field["split"], x)
                        # )
                        # data_table[field["name"]] = data_table[field["name"]].str.findall(field["split"])

                # compile list of values
                values = data_table[field["name"]].tolist()
                list_values = list(flatten(values))

                # down select unique values
                unique_values = sorted(list(set(list_values)))

                # add unique values to survey data for relevant fields
                survey["relevant_fields"].append({"name": field["name"], "unique_values": unique_values})

        # Compile data survey information
        survey["filepath"] = data_entry["filepath"]
        survey["data_schema"] = schema
        survey_data["files"].append(survey)

    # Save data survey output
    output_location = Path(args[0]).parent / f"{Path(args[0]).stem}_survey_output.json"
    save_dictionary_json(output_location, survey_data)


if __name__ == "__main__":
    main(sys.argv[1:])
