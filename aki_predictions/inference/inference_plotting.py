import sys
import time
import logging
from pathlib import Path
import operator

import matplotlib.pyplot as plt

from aki_predictions.file_operations import load_jsonl, load_json


def main(output_dir, data_file, inference_file):
    """Main inference plotting function."""
    # Load data file
    root_logger = logging.getLogger(__name__)

    artifacts_dir = Path(output_dir)

    # Load artifacts
    root_logger.info(f"Artifacts directory: {artifacts_dir}")

    # load data
    root_logger.info("Loading data file...")
    data = load_jsonl(Path(data_file))[0]

    # load data
    root_logger.info("Loading data file...")
    inference_output = load_json(Path(inference_file))

    root_logger.info(f"Processing inference for spell: {data['record_number']}")

    time_bins = [6, 12, 18, 24, 30, 36, 42, 48]

    # Extract ground truth
    adverse_outcome = False
    for event in data["episodes"][0]["events"]:
        if event["labels"]["adverse_outcome_in_spell"] == "1":
            adverse_outcome = True

    outcomes = {}

    if adverse_outcome:
        events_list = data["episodes"][0]["events"]
        sorted_events = events_list.copy()
        sorted_events.sort(key=operator.itemgetter("time_of_day"))
        sorted_events.sort(key=operator.itemgetter("patient_age"))

        time_reference = (int(sorted_events[0]["patient_age"]) * 60 * 60 * 24) + (
            (int(sorted_events[0]["time_of_day"]) - 1) * (6 * 3600)
        )

        # Stay outcome mortality
        for event in data["episodes"][0]["events"]:
            if event["labels"]["adverse_outcome_mortality_within_6h"] == "1":
                time_point = (int(event["patient_age"]) * 60 * 60 * 24) + ((int(event["time_of_day"]) - 1) * (6 * 3600))
                outcomes["mortality"] = time_point - time_reference

        # Stay outcome itu
        for event in data["episodes"][0]["events"]:
            if event["labels"]["adverse_outcome_itu_within_6h"] == "1":
                time_point = (int(event["patient_age"]) * 60 * 60 * 24) + ((int(event["time_of_day"]) - 1) * (6 * 3600))
                outcomes["itu"] = time_point - time_reference

        # Stay outcome dialysis
        for event in data["episodes"][0]["events"]:
            if event["labels"]["adverse_outcome_dialysis_within_6h"] == "1":
                time_point = (int(event["patient_age"]) * 60 * 60 * 24) + ((int(event["time_of_day"]) - 1) * (6 * 3600))
                outcomes["dialysis"] = time_point - time_reference

    root_logger.info(outcomes)
    root_logger.info({outcome: val / 86400 for outcome, val in outcomes.items()})

    for outcome, values in inference_output.items():
        root_logger.info(f"Processing outcome: {outcome}")

        fig = plt.figure()

        ax1 = fig.add_subplot()
        ax1.set_xlabel("Time (days)")
        ax1.set_ylabel("Predictive value")
        ax1.set_title(f"Inference output for {outcome} for record {data['record_number']}")

        legend = []
        for interval, outputs in values.items():
            root_logger.info(f"Processing interval: {interval}")
            time_steps = [val / 86400 for val in outputs["timestamps"]]
            predictions = outputs["values"]

            (l,) = ax1.plot(time_steps, predictions)
            legend.append(l)

        if outcome in outcomes:
            ax1.vlines(x=[outcomes[outcome] / 86400], ymin=0, ymax=1, color="b")

        ax1.legend(legend, [str(bin) for bin in time_bins])

        plt.savefig(
            artifacts_dir / f"{outcome}-inference-plot-{data['record_number']}.png", bbox_inches="tight", dpi=200
        )

        plt.close(fig)


if __name__ == "__main__":
    output_dir = sys.argv[1]
    data = sys.argv[2]
    inference = sys.argv[3]
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(output_dir, f"{timestamp}_inference_plotting_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(output_dir, data, inference)
