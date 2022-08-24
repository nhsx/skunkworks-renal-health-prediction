# Methods for loading datasets.
from pathlib import Path
import operator
import datetime
import copy
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from aki_predictions.file_operations import load_xlsx, load_csv


logger = logging.getLogger(__name__)


def load_raw_data(data_sources: dict) -> dict:
    """Load dataset.

    data_sources schema: [
       {
           "filepath": "<data path>" (can be list of file paths to concatenate - for files with matching schemas),
           "skiprows": <number of rows to skip>, (excel only)
           "name": "<short name for data file, used as data key>",
           "master_key": "<field to use as master key field>"
           "secondary_key": "<used to state that the secondary key should be used for this source>"
           "concatenate_columns": [{ (used to concatenate a combination of columns on data load - assumes strings)
               "field": "datetime",
               "values": [ (assumes concatenating in order of list)
                   "Date",
                   "Time"
               ],
               "seperator": <seperator to add between string columns>
           }, ... (can be list of column pairs to combine with concatenate_columns) ],
           "drop_columns": [] (list of column names from source to drop on load - for efficiency of indexing)
           "split_columns": [ {
               "value": "Discharge Date",
               "fields": [
                   "Year",
                   "Month",
                   "Day"
               ],
               "delimiter": "-"
               }, ... (can be list of columns to split with split_columns)
           ]
       }
       ... (can be list of files)
    ]

    Args:
        data_sources (dict): dictionary (json) stating sources and settings.

    Returns:
        (dict): dictionary of raw loaded data tables.
    """
    raw_data = {}
    for data_entry in tqdm(data_sources):
        source_name = data_entry["name"]
        if logger is not None:
            logger.info(f"Loading data source... {source_name}")
        if isinstance(data_entry["filepath"], list):
            raw_data[data_entry["name"]] = load_multiple_data_entry(data_entry)
        else:
            raw_data[data_entry["name"]] = load_single_data_entry(data_entry)
        raw_data = concatenate_columns(data_entry, raw_data)
        raw_data = split_columns(data_entry, raw_data)
        raw_data = rename_columns(data_entry, raw_data)
        raw_data = eval_duplicates(data_entry, raw_data, remove_all_dupe_rows=False)
        raw_data = format_datetimes(data_entry, raw_data)

        if "str_lookup" in data_entry.keys():
            # Load lookup table
            lookup_table = load_single_data_entry(data_entry["str_lookup"])
            lookup_primary_key = data_entry["str_lookup"]["primary_key"]
            # Create new column in raw data of csv list containing primary keys
            # apprearing in value column as new field column
            new_column_name = data_entry["str_lookup"]["field"]
            source_column = data_entry["str_lookup"]["value"]
            # Iterate through data rows in source and extract values
            new_values = []

            lookup_primary_keys = lookup_table[lookup_primary_key].astype(str).unique().tolist()

            for row in tqdm(raw_data[data_entry["name"]].itertuples(), total=len(raw_data[data_entry["name"]].index)):
                # Extract values from row in specified source column
                entries = getattr(row, source_column)
                seen_keys = []
                if "nan" not in str(entries).lower():
                    for key in lookup_primary_keys:
                        if str(key) in str(entries):
                            seen_keys.append(str(key))
                # Convert seen keys to string
                string_keys = data_entry["str_lookup"]["delimiter"].join(seen_keys)
                # Add values to list
                new_values.append(string_keys)

            # Add column to raw data for access
            raw_data[data_entry["name"]][new_column_name] = new_values

    return raw_data


def load_single_data_entry(data_entry: dict) -> pd.DataFrame:
    """Load set of data files from index into dataframes.

    data_entry format = {
        "filepath": "<filepath>",
        "skiprows": "<rows to skip on load for xlsx>"
    }

    Args:
        data_entry (dict): Dictionary defining data entry.

    Returns:
        (pd.DataFrame): Loaded dataframe.
    """
    filepath = data_entry["filepath"]
    data_table = None
    if filepath:
        if Path(filepath).suffix == ".xlsx":
            skiprows = data_entry["skiprows"] if "skiprows" in data_entry else 0
            data_table = load_xlsx(filepath, skiprows=int(skiprows), dtype=str)
        elif Path(filepath).suffix == ".csv":
            data_table = load_csv(filepath, dtype=str)
    if "drop_columns" in data_entry:
        return data_table.drop(columns=data_entry["drop_columns"])
    else:
        return data_table


def load_multiple_data_entry(data_entry: dict) -> pd.DataFrame:
    """Load multiple data files of the same structure into a single data frame.

    data_entry format = {
        "filepath": [
            "<filepath1>",
            "<filepath2>"
        ]
        "skiprows": "<rows to skip on load for xlsx>"
    }

    Args:
        data_entry (dict): Dictionary defining data entry.

    Returns:
        (pd.DataFrame): Loaded dataframe.
    """
    tables = []
    for filepath in data_entry["filepath"]:
        if Path(filepath).suffix == ".xlsx":
            skiprows = data_entry["skiprows"] if "skiprows" in data_entry else 0
            data_table = load_xlsx(filepath, skiprows=int(skiprows), dtype=str)
        elif Path(filepath).suffix == ".csv":
            data_table = load_csv(filepath, dtype=str)
        if "drop_columns" in data_entry:
            tables.append(data_table.drop(columns=data_entry["drop_columns"]))
        else:
            tables.append(data_table)
    if tables:
        tables = pd.concat(tables, axis=0, ignore_index=True)
    return tables


def concatenate_columns(data_entry: dict, raw_data: dict) -> dict:
    """Concatenate columns in raw data specified in data entry.

    data_entry format = {
        "concatenate_columns": [ {
                "field": "<field name>",
                "values": [
                    "<column name>",
                    "<column name>"
                ],
                "seperator": "<seperator character>"
                }, ... (can be list of columns to split with split_columns)
            ],

    Args:
        data_entry (dict): Dictionary defining data entry.
        raw_data (dict): Dictionary defining raw data.

    Returns:
        (dict): Modified raw data.
    """
    if "concatenate_columns" in data_entry:
        for concat in data_entry["concatenate_columns"]:
            target = raw_data[data_entry["name"]][concat["values"]]
            raw_data[data_entry["name"]][concat["field"]] = target.astype(str).agg(concat["seperator"].join, axis=1)
    return raw_data


def split_columns(data_entry: dict, raw_data: dict) -> dict:
    """Split columns in raw data specified in data entry.

    data_entry format = {
       "split_columns": [ {
                 "value": "<column name>",
                 "fields": [
                     "<field name>",
                     "<field name>",
                     "<field name>"
                 ],
                 "delimiter": "<delimiter character>"
                 }, ... (can be list of columns to split with split_columns)
             ]

    Args:
        data_entry (dict): Dictionary defining data entry.
        raw_data (dict): Dictionary defining raw data.

    Returns:
        (dict): Modified raw data.
    """
    if "split_columns" in data_entry:
        for split in data_entry["split_columns"]:
            target = raw_data[data_entry["name"]][split["value"]]
            raw_data[data_entry["name"]][split["fields"]] = target.astype(str).str.split(
                split["delimiter"], expand=True
            )
    return raw_data


def rename_columns(data_entry: dict, raw_data: dict) -> dict:
    """Rename columns using dictionary of old and new names.

    Args:
        data_entry (dict): Dictionary defining data entry.
        raw_data (dict): Dictionary defining raw data.

    Returns:
        (dict): Modified raw data.
    """
    if "rename_columns" in data_entry:
        raw_data[data_entry["name"]].rename(columns=data_entry["rename_columns"], inplace=True)
    return raw_data


def eval_duplicates(data_entry: dict, raw_data: dict, remove_all_dupe_rows: bool) -> dict:
    """Evaluate raw data and either remove duplicates or remove all rows that have duplicates.

    data_entry format = {
        "remove_duplicates": ["<column name>", "<column name>"]
        }

    Args:
        data_entry (dict): Dictionary defining data entry.
        raw_data (dict): Dictionary defining raw data.
        remove_all_dupe_rows (bool): if True, remove all rows that have duplicates else keep first
        duplicate row and remove all other duplicates.

    Returns:
        (dict): Modified raw data.
    """
    if "remove_duplicates" in data_entry:
        dupe_count = (
            raw_data[data_entry["name"]]
            .groupby(raw_data[data_entry["name"]].columns.tolist(), as_index=False, dropna=False)
            .size()
        )
        if logger is not None:
            logger.info("Duplicates (size column is number of duplicates):")
            logger.info(dupe_count[dupe_count["size"] > 1])  # group size greater than 1 implies duplicates.
            logger.info("Number of rows with duplicate(s):")
            logger.info(
                dupe_count[dupe_count["size"] > 1]["size"].sum() - len(dupe_count[dupe_count["size"] > 1].index)
            )
        if remove_all_dupe_rows:
            raw_data[data_entry["name"]] = (
                raw_data[data_entry["name"]]
                .drop_duplicates(subset=data_entry["remove_duplicates"], keep=False)
                .reset_index(drop=True)
            )
        else:
            raw_data[data_entry["name"]] = (
                raw_data[data_entry["name"]]
                .drop_duplicates(subset=data_entry["remove_duplicates"], keep="first")
                .reset_index(drop=True)
            )
    return raw_data


def format_datetimes(data_entry: dict, raw_data: dict, output_format=None) -> dict:
    """Format datetime columns into standard format."""
    if output_format is None:
        output_format = "%Y-%m-%d %H:%M:%S"
    if "datetime_format" in data_entry:
        for entry in data_entry["datetime_format"]:
            if "destination_format" in entry:
                output_format = entry["destination_format"]
            if "alternate_source_formats" in entry:
                raw_data[data_entry["name"]][entry["field"]] = raw_data[data_entry["name"]][entry["field"]].apply(
                    lambda x: convert_time(
                        x,
                        entry["source_format"],
                        entry["alternate_source_formats"],
                        output_format,
                    )
                )
            else:
                raw_data[data_entry["name"]][entry["field"]] = raw_data[data_entry["name"]][entry["field"]].apply(
                    lambda x: convert_time(x, entry["source_format"], "", output_format)
                )
    return raw_data


def convert_time(timestamp, source_format, alternate_source_formats, output_format):
    """Convert timestamp (of some format) into desired format."""
    output = None
    if isinstance(timestamp, str):
        try:
            output = datetime.datetime.strptime(timestamp, source_format).strftime(output_format)
        except ValueError:
            for format in alternate_source_formats:
                try:
                    output = datetime.datetime.strptime(timestamp, format).strftime(output_format)
                    break
                except ValueError:
                    pass
    elif isinstance(timestamp, datetime.datetime):
        output = timestamp.strftime(output_format)
    elif pd.isnull(timestamp):
        output = timestamp
    return output


def group_events(events):
    """Reduce dimensionality of events by common patient age and time bin, combining entry lists.
    Drops events with no timestamp logged.

    Example event format:
        {
            'patient_age': 1,
            'time_bin': 2,
            'entries': [
                {
                    'feature_category_idx': '1',
                    'feature_idx': '2',
                    'feature_value': '3'
                },
            ]
        }

    Args:
        events (list): list of events.

    Returns:
        (list): list of combined events.
    """
    grouped_events = {}

    # Create new structure grouped by age
    for event in events:
        age = event["patient_age"]
        time_bin = event["time_of_day"]
        key = str(f"{age}_{time_bin}")
        if key != "_":
            if key not in grouped_events:
                grouped_events[key] = event.copy()
                grouped_events[key]["entries"] = []

    # Run through events and add relevant entries to correct group
    for event in events:
        age = event["patient_age"]
        time_bin = event["time_of_day"]
        key = str(f"{age}_{time_bin}")
        if key != "_":
            for entry in event["entries"]:
                grouped_events[key]["entries"].append(entry)

    events_list = list(grouped_events.values())
    sorted_events = events_list.copy()
    sorted_events.sort(key=operator.itemgetter("time_of_day"))
    sorted_events.sort(key=operator.itemgetter("patient_age"))

    return sorted_events


def label_events(  # noqa: C901
    events,
    label_config,
    feature_mapping,
    metadata_mapping,
    label_mapping,
    label_template,
    numerical_label_template,
    record,
    window_times,
):
    """Label events using feature mapping and labelling config.

    Args:
        events (list): list of events
        label_config (dict): configuration dictionary for labelling
        feature_mapping (dict): feature mapping
        metadata_mapping (dict): metadata mapping (to match metadata features)
        label_mapping (dict): mapping of outcome label indexes
        label_template (dict): template for labelling (default keys and values)
        numerical_label_template (dict): template for numerical labelling
        record (dict): entire record (for referencing metadata features)
        window_times (list): list of window times for predictions

    Returns:
        (list): list of labelled events.
        (str): if the key is "of interest" relating to specific label rules. Specifies label if interest.
    """
    labelled_events = []
    adverse_outcomes_logged = []
    of_interest = None
    for i, event in enumerate(events):
        # print(f"Processing event {i}")
        # Update template
        updated_label_template = {key: value[:] for key, value in label_template.items()}
        if adverse_outcomes_logged:
            # Persist adverse outcome label for future events.
            updated_label_template["adverse_outcome"] = adverse_outcomes_logged[-1]
            updated_label_template["segment_mask"] = "1"
        for entry in event["entries"]:
            # Check feature index against mapping to compare to outcome label.
            # Process through each outcome.
            for label_key, label_info in label_config["adverse_outcome"]["labels"].items():
                for value in label_info["values"]:
                    # print(f"Asessing {value}")
                    feature_name = label_info["name"]
                    feature_lookup = f"{feature_name}_{value}"
                    if feature_lookup in feature_mapping:
                        #  print("Feature in lookup")
                        if entry["feature_idx"] == str(feature_mapping[feature_lookup]):
                            if "metadata" in label_info:
                                # Check metadata listing
                                metadata_feature_names = []
                                for e in label_info["metadata"]:
                                    metadata_feature_names.append(f"diagnosis_{e}")
                                metadata_ids = []
                                for e in metadata_feature_names:
                                    if e in metadata_mapping:
                                        metadata_ids.append(metadata_mapping[e])
                                presence_map = []
                                for idx in metadata_ids:
                                    if str(idx) in record:
                                        presence_map.append(True)
                                # Check condition
                                if any(presence_map):
                                    idx = str(feature_mapping[feature_lookup])
                                    #  print(
                                    #      f"Positive labelling for {label_key} of {value} from feature {feature_name},"
                                    #      f" of feature_idx of {idx} with presence mapping of metadata keys."
                                    #  )
                                    updated_label_template["adverse_outcome"] = str(label_mapping[label_key])
                                    adverse_outcomes_logged.append(str(label_mapping[label_key]))
                            else:
                                idx = str(feature_mapping[feature_lookup])
                                #  print(
                                #      f"Positive labelling for {label_key} of {value} from feature {feature_name},"
                                #      f" of feature_idx of {idx}"
                                #  )
                                updated_label_template["adverse_outcome"] = str(label_mapping[label_key])
                                adverse_outcomes_logged.append(str(label_mapping[label_key]))
                            # Set of_interest output info
                            if "of_interest" in label_config.keys():
                                if label_key == label_config["of_interest"]:
                                    of_interest = label_key
                if str(label_mapping[label_key]) in adverse_outcomes_logged:
                    # Adverse outcome has occurred, therefore set all lookaheads for that key to 1
                    for lookahead_time in window_times:
                        # print(f"Setting adverse_outcome_{label_key}_within_{lookahead_time}h to 1")
                        updated_label_template[f"adverse_outcome_{label_key}_within_{lookahead_time}h"] = "1"
                if adverse_outcomes_logged:
                    updated_label_template["segment_mask"] = "1"

        updated_event = copy.deepcopy(event)
        updated_event["labels"] = updated_label_template
        # TODO: Populate numerical labels
        updated_event["numerical_labels"] = numerical_label_template
        labelled_events.append(updated_event)

    # Use adverse outcome labelling to do look ahead
    if adverse_outcomes_logged:  # only lookahead if adverse outcome present for spell.
        # Work through n bins in the future and adjust labels for events that fit in that bin.
        number_of_bins = int(24 / int(label_config["adverse_outcome"]["time_step"]))

        lookahead_labelled_events = []

        for i, event in enumerate(labelled_events):
            # Look ahead by n events
            next_day, next_bin, _, overflow = identify_next_time_bin(
                event["patient_age"], event["time_of_day"], number_of_bins
            )

            new_event = event.copy()
            for x, time_period in enumerate(window_times):
                # For each outcome
                if x != 0:  # Already at next time bin
                    next_day, next_bin, _, overflow = identify_next_time_bin(next_day, next_bin, number_of_bins)

                for label_key, label_info in label_config["adverse_outcome"]["labels"].items():
                    # Deal with first period
                    # Search through events for those with next time bin
                    for j, search_event in enumerate(labelled_events):
                        if j != i:  # Ignore current event
                            if search_event["patient_age"] == next_day and search_event["time_of_day"] == next_bin:
                                # Event matches time bin
                                if search_event["labels"]["adverse_outcome"] == str(label_mapping[label_key]):
                                    #  print("Matching event has adverse outcome")
                                    new_event["labels"][f"adverse_outcome_{label_key}_within_{time_period}h"] = "1"
                                    for lookahead_time in window_times:
                                        if lookahead_time > time_period:
                                            new_event["labels"][
                                                f"adverse_outcome_{label_key}_within_{lookahead_time}h"
                                            ] = "1"
                            elif overflow:
                                if (
                                    search_event["patient_age"] == next_day - 1
                                    and search_event["time_of_day"] == number_of_bins
                                ):
                                    # Check final time bin from day before
                                    if search_event["labels"]["adverse_outcome"] == str(label_mapping[label_key]):
                                        #  print("Matching event has adverse outcome")
                                        new_event["labels"][f"adverse_outcome_{label_key}_within_{time_period}h"] = "1"
                                        for lookahead_time in window_times:
                                            if lookahead_time > time_period:
                                                new_event["labels"][
                                                    f"adverse_outcome_{label_key}_within_{lookahead_time}h"
                                                ] = "1"
            lookahead_labelled_events.append(new_event)

        labelled_events = lookahead_labelled_events

    if adverse_outcomes_logged:
        # Add adverse outcome in spell flag.
        updated_labelled_events = []
        for event in labelled_events:
            updated_event = copy.deepcopy(event)
            updated_event["labels"]["adverse_outcome_in_spell"] = "1"
            updated_labelled_events.append(updated_event)
        return updated_labelled_events, of_interest
    else:
        return labelled_events, of_interest


def identify_next_time_bin(day, bin, num_bins):
    """Return next time bin to check for adverse outcomes n bins in future.

    Args:
        day (int): day
        bin (int): bin within day
        lookahead (int): number of bins to lookahead
        num_bins (int): number of bins within a day

    Returns:
        (int, int, int, boolean): next day, bin, num_bins, overflow (to designate checking the 4th bin)
    """
    if bin == num_bins:
        return day + 1, 0, num_bins, True
    elif bin < num_bins:
        if bin + 1 == num_bins:
            return day + 1, 0, num_bins, True
        else:
            return day, bin + 1, num_bins, False


def datetime_bins(date_time, bins: int, include_minutes=False, string_format=None) -> int:
    """Takes a datetime string and returns bin index.

    An additional bin (n_bins + 1) is added to account for all events which do not have a specific time element.
    Times matching 00:00:00 are assumed to be those without specific time element.

    Args:
        date_time (str): Date and time in the format '%d/%m/%Y %H:%M:%S'.
        bins (int): Number of bins.
        include_minutes (bool): Optional, if True then calculates bin position by minutes.

    Returns:
        (int): Bin index between 0 and bins + 1
    """
    if string_format is None:
        string_format = "%d/%m/%Y %H:%M:%S"
    dt_obj = datetime.datetime.strptime(date_time, string_format)

    if dt_obj.hour == 0 and dt_obj.minute == 0 and dt_obj.second == 0:
        return bins

    if include_minutes:
        total_minutes = (dt_obj.hour * 60) + dt_obj.minute
        return get_bin(total_minutes, 1440, bins)
    else:
        hour = dt_obj.hour
        return get_bin(hour, 24, bins)


def get_bin(val, total, bins) -> int:
    """datetime_bin helper function. Returns a bin given the total expected value and the value to be binned."""
    steps = int(total / bins)
    bins_space = [1] * bins
    for i, _ in enumerate(bins_space):
        bins_space[i] = steps * (i + 1)
    return int(np.digitize(val, bins_space))


def relative_date(year_birth, date_event: str, string_format=None) -> int:
    """Find the relative distance in day between a birth year and an event date.

    Args:
        year_birth (str or int or float): Year of birth.
        date_event (str): Date and time of event in the format '%d/%m/%Y %H:%M:%S'.

    Returns:
        (int): Relative distance in days.
    """
    if string_format is None:
        string_format = "%d/%m/%Y %H:%M:%S"
    y = datetime.datetime(int(year_birth), 1, 1)
    x = datetime.datetime.strptime(date_event, string_format)
    delta = x - y
    return delta.days
