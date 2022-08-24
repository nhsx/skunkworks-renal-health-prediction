# Script to load a json description of a dataset and then work through data using a
# mapping definition to re-structure into specific data entries.
#
# Expects data index of form:
#   {
#    "sources": [
#        {
#            "filepath": "<data path>" (can be list of file paths to concatenate- for files with matching schemas),
#            "skiprows": <number of rows to skip>, (excel only)
#            "name": "<short name for data file, used as data key>",
#            "master_key": "<field to use as master key field>"
#            "secondary_key": "<used to state that the secondary key should be used for this source>"
#            "concatenate_columns": [{ (used to concatenate a combination of columns on data load - assumes strings)
#                "field": "<new field name>",
#                "values": [ (assumes concatenating in order of list)
#                    ... (column names to concatenate)
#                ],
#                "seperator": <seperator to add between string columns>
#            }, ... (can be list of column pairs to combine with concatenate_columns) ],
#            "drop_columns": [] (list of column names from source to drop on load - for efficiency of indexing)
#            "split_columns": [ {
#                "value": "<column name>",
#                "fields": [
#                    ... (list of fields in order to split)
#                ],
#                "delimiter": "-"
#                }, ... (can be list of columns to split with split_columns)
#            ],
#            "str_lookup": {
#                "type": "metadata",
#                "filepath": "<filepath>", (lookup table)
#                "name": "<name of table>",
#                "primary_key": "<primary lookup column>",
#                "secondary_key": "<secondary lookp column>",
#                "value": "<column to look up values from>",
#                "field": "<column name>", (new column to populate with detected primary keys)
#                "delimiter": "," (delimiter to use when detecting more than one value from lookup list in value column)
#            },
#            "remove_duplicates": [column name, column name]
#        }
#        ... (can be list of files)
#    ],
#    "mapping": {
#        "master_key": {
#            "source": "<source for master key",
#            "field": "<master key for unique data record", (master key to use to select row in source table)
#            "filter_rules": [ (rules on leaving out records relating to specific master keys)
#                {
#                    "description": "<description>",
#                    "source": "<source name>",
#                    "field": "<master key field>",
#                    "query_field": "<query field to look in>",
#                    "drop_values": [
#                       ... (list of values to detect)
#                    ]
#                },
#                {
#                    "description": "<description>",
#                    "source": "<source name>",
#                    "field": "<master key field>",
#                    "query_field": "<query column name>",
#                    "remove_if_populated": "True"
#                }
#            ]
#        }
#        "secondary_key": {
#           "source": "<source for secondary key - must be same as master key>",
#           "field": "<secondary key for unique data record>", (must be from same source as master key, mapped together)
#        }
#        "numerical_features": [
#            <list of numerical output feature names in any order,
#             these will map to feature indexes depending on the order>
#        ],
#        "numerical_feature_exclude_list": {
#            "<feature_name>": [
#                (list of string values to ignore when finding feature entries)
#            ],
#            ... (allows multiple features)
#        }
#        "categorical_features": [ (these features will have name and value
#                                   used to assign an index for the value in the categorical feature
#                                   value mapping table)
#            {
#                "name": "<feature name>",
#                "source": "<source sheet>",
#                "field": "<field in source>"
#            },
#        ]
#        "metadata": {
#            "<field name in new data record>": {
#                "source": "<short name of data file (see above)",
#                "field": "<name of field to extract>"
#            },
#            ... (can have multiple metadata keys)
#        }
#        "events": [
#             {
#                 "category": "<category of event>",
#                 "source": "<short name of data file (see above)",
#                 "datetime": "<field in source to use as timestamp>"
#                 "features": [
#                     {
#                         "name": "<feature name>",
#                         "field": "<field to access in data>",
#                         "type": "<numerical or categorical, this will determine how to handle one hot encoding>"
#
#                     },
#                     ... (multiple allowed)
#                 ]
#             },
#             ... (multiple allowed)
#        ],
#        "labelling": { (labelling is largely based on categorical feature values)
#            "labels": {
#                "of_interest": "<adverse outcome of interest>",
#                "output_of_interest": "<Source data to export for keys of interest>",
#                "master_key_of_interest": "<Master key column for specific source>",
#                "adverse_outcome": {
#                    "time_step": "6", (gap in hours between outcome labels)
#                    "max_look_ahead": "8", (number of periods to look ahead for labelling)
#                    "labels": {
#                        "<adverse outcome name>": {
#                            "name": "<feature name to look at>",
#                            "values": ["<value of categorical feature>"],
#                            "metadata": [
#                                <list of metadata feature names to presence check as secondary rule>
#                            ]
#                        },
#                        ... (multiple adverse outcomes allowed)
#                    }
#                }
#            },
#            "numerical_labels": {
#            }
#        }
#    }
#   }
#
# Provide data index file path via CLI argument on execution.
#
# Input file(s):
#    <config filename>.json (see above for format)
#
# Output file(s):
#    Directory in same location as config -> Named <config filename>_%Y-%m-%d-%H%M%S
#    Containing:
#       <date timestamp>_log.txt (log of console outputs for tracking)
#       <config filename>.json (backup of config used)
#       REVISION.txt (git revision used for ingest run)
#       category_mapping.json
#       feature_mapping.json
#       metadata_mapping.json
#       numerical_feature_mapping.json
#       feature_statistics.json
#       master_keys.csv
#       master_keys_filtered.csv
#       label_mapping.json
#       debug/numerical_feature_invalid.json
#       debug/<feature name>_unique_values.csv
#           (csv for each categorical feature stating unique values.)
#       debug/ingest_records_output_single_line_<n>.json
#           (output for first n records - default of 10, in structured pretty json)
#       debug/data_of_interest.xlsx (raw data output for specific keys of interest)
#       ingest_records_output_lines.jsonl (formatted data output)
import sys
from pathlib import Path
import time
import subprocess
import logging
import io
from collections import Iterable, Counter

import pandas as pd
from tqdm import tqdm
import numpy as np

from aki_predictions.file_operations import load_json, save_dictionary_json, append_jsonl, save_csv
from aki_predictions.data_loading import load_raw_data, relative_date, datetime_bins, group_events, label_events


# https://stackoverflow.com/a/17865033
def flatten(coll):
    """Flattens list of lists into single list while handling strings."""
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


# https://stackoverflow.com/a/47113836
def is_valid_decimal(s):
    """Perform a value conversion check for variable to float."""
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


# https://stackoverflow.com/a/21901260
def get_git_revision_hash() -> str:
    """Get git revision hash for logging."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def clean_feature_string(value):
    """Convert strings to lowercase and strip any double quotes."""
    new_value = value.lower().replace('"', "").replace(",", "")
    return new_value


def extract_keys(data_index, raw_data, root_logger):
    """Extract master keys from dataset prototype."""
    # Identify and extract master keys
    unique_master_keys = (
        raw_data[data_index["mapping"]["master_key"]["source"]][data_index["mapping"]["master_key"]["field"]]
        .astype(str)
        .unique()
        .tolist()
    )
    unique_master_keys = [i for i in unique_master_keys if i != "nan"]

    root_logger.info(f"Number of unique keys: {len(unique_master_keys)}")

    # Extract secondary keys against master key set
    root_logger.info("Extracting key mappings...")
    key_mapping_list = []
    for master_key in tqdm(unique_master_keys):
        source = raw_data[data_index["mapping"]["secondary_key"]["source"]]

        master_key_column = data_index["mapping"]["master_key"]["field"]
        secondary_key_column = data_index["mapping"]["secondary_key"]["field"]
        secondary_key = (
            source.loc[source[master_key_column] == master_key, secondary_key_column].astype(str).unique().tolist()
        )

        pair = {"master_key": master_key, "secondary_key": secondary_key[0]}
        key_mapping_list.append(pair)

    return unique_master_keys, pd.DataFrame(key_mapping_list)


def filter_keys(data_index, raw_data, master_keys, root_logger):  # noqa: C901
    """Filter master keys by alternate field values."""
    filtered_keys = master_keys

    rules = data_index["mapping"]["master_key"]["filter_rules"]

    keys_to_exclude = []
    for rule in rules:
        rule_description = rule["description"]
        root_logger.info(f"Processing key filtering rule... {rule_description}")

        source_sheet = rule["source"]
        source_field = rule["field"]
        query_field = rule["query_field"]

        source_data = raw_data[source_sheet]

        for key in tqdm(filtered_keys):
            entries = source_data.loc[source_data[source_field] == key]

            if "drop_values" in rule:
                if len(entries) == 1:
                    query_value = entries[query_field].astype(str).to_list()[0]
                    drop_values = rule["drop_values"]
                    if query_value in drop_values:
                        keys_to_exclude.append(key)
                elif len(entries) > 1:
                    if "sort_by" in rule:
                        entries = entries.sort_values(by=[rule["sort_by"]["field"]])

                    if "apply_first" in rule["sort_by"]:
                        if rule["sort_by"]["apply_first"] == "True":
                            for entry in entries.itertuples():
                                query_value = getattr(entry, query_field)
                                drop_values = rule["drop_values"]
                                if query_value in drop_values:
                                    keys_to_exclude.append(key)
                                break
                    else:
                        for entry in entries.itertuples():
                            query_value = getattr(entry, query_field)
                            drop_values = rule["drop_values"]
                            if query_value in drop_values:
                                keys_to_exclude.append(key)

            if "remove_if_populated" in rule:
                for entry in entries.itertuples():
                    if rule["remove_if_populated"] == "True":
                        query_value = getattr(entry, query_field)
                        if not pd.isna(query_value):
                            keys_to_exclude.append(key)

    for key in list(set(keys_to_exclude)):
        filtered_keys.remove(key)

    return filtered_keys


def extract_feature_mapping(debug_dir, data_index, raw_data, root_logger):
    """Extract and save feature and category mappings prototype."""
    root_logger.info("Extracting feature mappings...")
    # Compile feature category mappings
    feature_mapping = {}
    category_mapping = {}

    # Extract feature names and assign ids to each feature for substitution into data output.
    numerical_feature_names = data_index["mapping"]["numerical_features"]
    numerical_feature_mapping = {}

    i = 0  # Feature index mapping
    j = 0
    for name in numerical_feature_names:
        feature_mapping[name] = i
        category_mapping[name] = i  # Change later to give mappings dependent on the category
        numerical_feature_mapping[name] = i
        i += 1

    j = i  # initialise categorical feature indexes from end of numerical index numbering
    # Work through categoricals and extract feature names for indexing.
    categorical_feature_names = []
    for feature in data_index["mapping"]["categorical_features"]:
        feature_name = feature["name"]
        source = raw_data[feature["source"]]
        possible_values = source[feature["field"]].astype(str).unique().tolist()

        # Clean up value strings
        cleaned_values = [clean_feature_string(val) for val in possible_values]
        unique_values = list(set(cleaned_values))

        # Create new feature name for unique categorical value
        for unique_value in unique_values:
            feature_mapping_name = f"{feature_name}_{unique_value}"
            categorical_feature_names.append(feature_mapping_name)
            feature_mapping[feature_mapping_name] = j
            category_mapping[feature_mapping_name] = i
            j += 1  # Increment feature index value

        # Save csv of unique categorical values
        output_location = debug_dir / f"feature_{feature_name}_unique_values.csv"
        save_csv(output_location, [[i] for i in sorted(unique_values)], ["unique values"])

        i += 1  # Increment category value.

    root_logger.info("Extracting metadata mappings...")
    # Generate metadata mapping ids
    metadata_mapping = {}

    i = 0  # Initialise metadata feature index
    for key, value in data_index["mapping"]["metadata"].items():
        metadata_feature_name = key
        metadata_feature_source = value["source"]
        metadata_feature_field = value["field"]
        metadata_feature_type = value["type"]

        if metadata_feature_type == "numerical":
            metadata_mapping[metadata_feature_name] = i
            i += 1
        elif metadata_feature_type == "categorical":
            if "delimiter" in value:
                tree_list = raw_data[metadata_feature_source][metadata_feature_field].astype(str).to_list()
                flat_list = []
                for entry in tree_list:
                    split_entry = entry.split(value["delimiter"])
                    flat_list.append(split_entry)
                possible_values = list(set(flatten(flat_list)))
            else:
                value_list = raw_data[metadata_feature_source][metadata_feature_field].astype(str).to_list()
                possible_values = list(set(value_list))

            for unique_value in possible_values:
                new_mapping_name = f"{metadata_feature_name}_{unique_value}"
                metadata_mapping[new_mapping_name] = i
                i += 1

    return feature_mapping, category_mapping, metadata_mapping, numerical_feature_names, numerical_feature_mapping


def generate_label_template(data_index):
    """Generate template for labelling."""
    template = {}
    numerical_template = {}
    label_mapping = {}

    config = data_index["mapping"]["labelling"]

    # Generate static labels and mapping
    template["adverse_outcome"] = "0"
    template["adverse_outcome_in_spell"] = "0"
    template["segment_mask"] = "0"
    for i, key in enumerate(config["labels"]["adverse_outcome"]["labels"].keys()):
        label_mapping[key] = i + 1

    window_times = []

    for key in config["labels"]["adverse_outcome"]["labels"].keys():
        label_key = f"adverse_outcome_{key}"
        start_time = int(config["labels"]["adverse_outcome"]["time_step"])
        end_time = int(config["labels"]["adverse_outcome"]["time_step"]) * int(
            config["labels"]["adverse_outcome"]["max_look_ahead"]
        )
        interval = int(config["labels"]["adverse_outcome"]["time_step"])
        for i in range(start_time, end_time, interval):
            label_key = f"adverse_outcome_{key}_within_{i}h"
            template[label_key] = "0"
            window_times.append(i)

    window_times = sorted(list(set(window_times)))

    return template, numerical_template, label_mapping, window_times


def main(output_dir, config_path):  # noqa: C901
    """Main ingest prototype.

    Args:
        output_dir (str): directory to locate output files
        config_path (str): path to configuration file

    Returns:
        (Path): Path to ingested data directory (timestamped)
    """
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    artifacts_dir = Path(output_dir) / (str(Path(config_path).stem) + "_" + timestamp)

    debug_dir = artifacts_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger(__name__)

    # Load data index
    data_index = load_json(Path(config_path))
    root_logger.info(f"Processing: {Path(config_path)}")
    # root_logger.info(f"Index: {data_index}")

    # Write file to output directory with git hash
    git_hash = get_git_revision_hash()
    with open(artifacts_dir / "REVISION.txt", "w") as f:
        f.write(git_hash)

    # Write data index to output as backup
    save_dictionary_json(artifacts_dir / (str(Path(config_path).stem) + ".json"), data_index)

    # Load data files
    raw_data = load_raw_data(data_index["sources"])

    # Print information about dataset
    for key, value in raw_data.items():
        root_logger.info(f"Loaded: {key}")
        buf = io.StringIO()
        value.info(buf=buf)
        s = buf.getvalue()
        root_logger.info(s)

    # Identify and extract master keys
    unique_master_keys, key_mapping = extract_keys(data_index, raw_data, root_logger)

    # Master keys
    number_of_master_keys = len(unique_master_keys)
    root_logger.info(f"Number of spells: {number_of_master_keys}")

    output_location = artifacts_dir / "master_keys.csv"
    save_csv(output_location, [[i] for i in unique_master_keys], ["master_key"])

    # Filter master keys by specified rules (such as specific values in specific fields)
    unique_master_keys = filter_keys(data_index, raw_data, unique_master_keys, root_logger)

    number_of_master_keys = len(unique_master_keys)
    root_logger.info(f"Number of spells post-filter: {number_of_master_keys}")

    output_location = artifacts_dir / "master_keys_filtered.csv"
    save_csv(output_location, [[i] for i in unique_master_keys], ["filtered_master_key"])

    # Iterate through master keys and apply data mapping, selecting data from each data entry.
    root_logger.info("Key Mappings Table:")
    buf = io.StringIO()
    key_mapping.info(buf=buf)
    s = buf.getvalue()
    root_logger.info(s)

    # Compile feature category mappings
    (
        feature_mapping,
        category_mapping,
        metadata_mapping,
        numerical_feature_names,
        numerical_feature_mapping,
    ) = extract_feature_mapping(debug_dir, data_index, raw_data, root_logger)

    # Save output dictionary of key index pairings
    output_location = artifacts_dir / "feature_mapping.json"
    save_dictionary_json(output_location, feature_mapping)

    # Save output dictionary of key index pairings for only numerical features
    output_location = artifacts_dir / "numerical_feature_mapping.json"
    save_dictionary_json(output_location, numerical_feature_mapping)

    # Save output dictionary of category index pairings
    output_location = artifacts_dir / "category_mapping.json"
    save_dictionary_json(output_location, category_mapping)

    # Save output dictionary of key index pairings
    output_location = artifacts_dir / "metadata_mapping.json"
    save_dictionary_json(output_location, metadata_mapping)

    # Extract feature statistics
    numerical_feature_cache = {feature_name: [] for feature_name in numerical_feature_names}
    numerical_feature_invalid_values = {feature_name: [] for feature_name in numerical_feature_names}

    label_template, numerical_label_template, label_mapping, window_times = generate_label_template(data_index)

    root_logger.info(f"Prediction and binning window times: {window_times}")
    # Number of bins based upon time step within the adverse outcome labelling.
    number_of_bins = int(24 / int(data_index["mapping"]["labelling"]["labels"]["adverse_outcome"]["time_step"]))
    root_logger.info(f"Splitting events into {number_of_bins} bins per day.")

    root_logger.info(label_template)

    # Save output dictionary of key index pairings for labels
    output_location = artifacts_dir / "label_mapping.json"
    save_dictionary_json(output_location, label_mapping)

    # Keys of interest
    # Gather keys relating to specific criteria to export relevant raw data
    interest_keys = []

    total_invalid_numerical_values_flagged = 0
    total_numerical_values_evaluated = 0
    total_numerical_values_ignored = 0

    # Extract records
    root_logger.info("Extracting data for each key...")
    for i, master_key in enumerate(tqdm(unique_master_keys)):
        # Create record
        record = {"record_number": i, "master_key": master_key, "episodes": [{"events": []}]}

        # Extract metadata using master key
        for key, value in data_index["mapping"]["metadata"].items():
            master_key_slice = raw_data[value["source"]][
                raw_data[value["source"]][data_index["mapping"]["master_key"]["field"]] == master_key
            ]
            if len(master_key_slice[value["field"]].values):
                if value["type"] == "numerical":
                    if str(master_key_slice[value["field"]].values[0]) != "nan":
                        record[str(metadata_mapping[key])] = str(master_key_slice[value["field"]].values[0])
                elif value["type"] == "categorical":
                    if str(master_key_slice[value["field"]].values[0]) != "nan":
                        categorical_value = master_key_slice[value["field"]].values[0]
                        # Distinguish between delimited categorical metadata values and standard ones.
                        if "delimiter" in data_index["mapping"]["metadata"][key]:
                            for code in categorical_value.split(data_index["mapping"]["metadata"][key]["delimiter"]):
                                combined_key = f"{key}_{code}"
                                record[str(metadata_mapping[combined_key])] = "1"
                        else:
                            combined_key = f"{key}_{categorical_value}"
                            record[str(metadata_mapping[combined_key])] = "1"

        # Extract event data for each key
        for event_def in data_index["mapping"]["events"]:
            event_source = raw_data[event_def["source"]]

            # Find key for specific source
            master_key_for_source = None
            master_key_value_for_source = None
            for data_source in data_index["sources"]:
                if data_source["name"] == event_def["source"]:
                    if "master_key" in data_source:
                        master_key_for_source = data_source["master_key"]
                        master_key_value_for_source = master_key
                    elif "secondary_key" in data_source:
                        master_key_for_source = data_source["secondary_key"]
                        master_key_value_for_source = (
                            key_mapping.loc[key_mapping["master_key"] == master_key, "secondary_key"]
                            .astype(str)
                            .tolist()[0]
                        )

            # Extract event data from source
            events = event_source.loc[event_source[master_key_for_source] == master_key_value_for_source].astype(str)
            for event in events.itertuples():
                # Convert datetime
                timestamp = getattr(event, event_def["datetime"])

                if pd.isnull(timestamp) or str(timestamp) == "nan" or timestamp is None or timestamp == "None":
                    # Events with missing timestamps are dropped when grouping events.
                    patient_age = ""
                    time_bin = ""
                else:
                    # Calculate relative datetime
                    year_of_birth = int(
                        record[str(metadata_mapping[data_index["mapping"]["date_reference_field_name"]])]
                    )
                    patient_age = relative_date(year_of_birth, timestamp, string_format="%Y-%m-%d %H:%M:%S")
                    # Calculate time of day bin
                    time_bin = datetime_bins(timestamp, number_of_bins, string_format="%Y-%m-%d %H:%M:%S")

                new_event = {
                    # "category": event_def["category"],
                    "patient_age": patient_age,
                    "time_of_day": time_bin,
                    "entries": [],
                    "labels": {},
                    "numerical_labels": {},
                }
                for feature in event_def["features"]:
                    # Flag for tracking if feature entry should be ignored.
                    skip_feature_entry = False
                    value = ""

                    if feature["type"] == "numerical":
                        key = feature["name"]
                        raw_value = getattr(event, feature["field"])

                        total_numerical_values_evaluated += 1
                        root_logger.info(f"Total numerical values evaluated {total_numerical_values_evaluated}")

                        if is_valid_decimal(raw_value) and pd.isnull(float(raw_value)) is False:
                            numerical_feature_cache[key].append(float(raw_value))
                            value = float(raw_value)
                        else:
                            # root_logger.info(
                            #     f"WARNING: Feature {key} as non-numerical value {raw_value},"
                            #     " and will be filtered out or included as empty entry."
                            # )

                            numerical_feature_invalid_values[key].append(raw_value)

                            ignore_list = data_index["mapping"]["numerical_feature_exclude_list"][key]
                            if raw_value in ignore_list or raw_value == "nan" or raw_value == "None":
                                # print(f"Ignoring value {raw_value} for feature {key}")
                                skip_feature_entry = True

                                total_numerical_values_ignored += 1
                                root_logger.info(f"Total numerical values ignored {total_numerical_values_ignored}")

                            else:
                                print(f"Setting value to empty for {raw_value} for feature {key}")
                                total_invalid_numerical_values_flagged += 1
                                root_logger.info(
                                    "Total invalid numerical values flagged (included as empty)"
                                    + f"{total_invalid_numerical_values_flagged}"
                                )

                                value = ""
                    elif feature["type"] == "categorical":
                        temp_key = feature["name"]
                        raw_value = getattr(event, feature["field"])
                        clean_value = clean_feature_string(raw_value)
                        key = f"{temp_key}_{clean_value}"
                        value = 1

                    # Add entry to entries
                    new_entry = {
                        "feature_category_idx": str(category_mapping[key]),
                        "feature_idx": str(feature_mapping[key]),
                        "feature_value": str(value),
                    }
                    if not skip_feature_entry:
                        new_event["entries"].append(new_entry)

                record["episodes"][0]["events"].append(new_event)

        # Export subset of structured records (with ungrouped events)
        if i < 10:
            output_temp_location = debug_dir / f"records_output_single_line_ungrouped_{i}.json"
            save_dictionary_json(output_temp_location, record)

        # Group and sort events (dropping events without a timestamp)
        record["episodes"][0]["events"] = group_events(record["episodes"][0]["events"])

        # Export subset of structured records (with grouped events)
        if i < 10:
            output_temp_location = debug_dir / f"records_output_single_line_grouped_{i}.json"
            save_dictionary_json(output_temp_location, record)

        # Work through sorted events and populate labels for record
        record["episodes"][0]["events"], key_of_interest = label_events(
            record["episodes"][0]["events"],
            data_index["mapping"]["labelling"]["labels"],
            feature_mapping,
            metadata_mapping,
            label_mapping,
            label_template,
            numerical_label_template,
            record,
            window_times,
        )

        # Add key to interest keys if matching label keyword type
        if "of_interest" in data_index["mapping"]["labelling"]["labels"].keys():
            if key_of_interest == data_index["mapping"]["labelling"]["labels"]["of_interest"]:
                interest_keys.append(record["master_key"])
                root_logger.info(f"Detected positive of interest case for key {record['master_key']}")

        output_temp_location = artifacts_dir / "ingest_records_output_lines.jsonl"
        append_jsonl(output_temp_location, record)

        # Export subset of structured records
        if i < 50:
            output_temp_location = debug_dir / f"ingest_records_output_single_line_{i}.json"
            save_dictionary_json(output_temp_location, record)

        # Uncomment for more ganular logging on the numerical values during processing.
        # root_logger.info({str(v): len(k) for v, k in numerical_feature_invalid_values.items()})
        # root_logger.info({str(v): k for v, k in dict(Counter(numerical_feature_invalid_values)).items()})

    # Save invalid numerical feature statistics
    formatted_invalid_feature_values = {}

    root_logger.info(f"End total numerical values evaluated {total_numerical_values_evaluated}")
    root_logger.info(f"End total invalid numerical values ignored {total_numerical_values_ignored}")
    root_logger.info(
        f"End total invalid numerical values flagged but still included {total_invalid_numerical_values_flagged}"
    )

    output_location = debug_dir / "numerical_feature_invalid_stats.json"
    save_dictionary_json(
        output_location, {str(v): k for v, k in dict(Counter(numerical_feature_invalid_values)).items()}
    )

    for key, values in numerical_feature_invalid_values.items():
        formatted_invalid_feature_values[key] = list(set(values))

    output_location = debug_dir / "numerical_feature_invalid.json"
    save_dictionary_json(output_location, formatted_invalid_feature_values)

    # Calculate feature statistics
    root_logger.info("Calculating statistics for numerical features...")
    feature_statistics = {}
    for feature_name in numerical_feature_names:
        # output_location = debug_dir / f"feature_statistics_{feature_name}.csv"
        # save_csv(output_location, [[i] for i in numerical_feature_cache[feature_name]], ["value"])

        if len(numerical_feature_cache[feature_name]):
            mean = np.mean(numerical_feature_cache[feature_name])
            std = np.std(numerical_feature_cache[feature_name])
            maximum = np.max(numerical_feature_cache[feature_name])
            minimum = np.min(numerical_feature_cache[feature_name])
            centile_1 = np.percentile(numerical_feature_cache[feature_name], 1)
            centile_99 = np.percentile(numerical_feature_cache[feature_name], 99)

            feature_statistics[feature_name] = {
                "mean": mean,
                "std": std,
                "max": maximum,
                "min": minimum,
                "centile_1": centile_1,
                "centile_99": centile_99,
                "values": numerical_feature_cache[feature_name],
            }
        else:
            feature_statistics[feature_name] = {
                "mean": "",
                "std": "",
                "max": "",
                "min": "",
                "centile_1": "",
                "centile_99": "",
                "values": [],
            }

    # Save feature statistics
    output_location = artifacts_dir / "feature_statistics.json"
    save_dictionary_json(output_location, feature_statistics)

    # Save out relevant data for keys of interest

    root_logger.info(f"Number of interest keys: {len(interest_keys)}")

    if (
        "output_of_interest" in data_index["mapping"]["labelling"]["labels"].keys()
        and "master_key_of_interest" in data_index["mapping"]["labelling"]["labels"].keys()
    ):
        sheet_of_interest = data_index["mapping"]["labelling"]["labels"]["output_of_interest"]
        column_filter = data_index["mapping"]["labelling"]["labels"]["master_key_of_interest"]
        subset_of_data = raw_data[sheet_of_interest][raw_data[sheet_of_interest][column_filter].isin(interest_keys)]

        path = debug_dir / "data_of_interest.xlsx"
        subset_of_data.to_excel(path)

    return artifacts_dir


if __name__ == "__main__":
    configuration_path = sys.argv[1]

    # Setup artifacts directory
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    artifacts_dir = Path(configuration_path).parent / (str(Path(configuration_path).stem) + "_" + timestamp)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(artifacts_dir, f"{timestamp}_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    root_logger.info(f"Output directory: {artifacts_dir}")

    main(artifacts_dir, configuration_path)
