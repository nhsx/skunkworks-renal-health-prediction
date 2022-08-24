"""
Assuming fake_patients.json has been created using aki_predictions/ehr_prediction_modeling/fake_data/convert_pb_json.py,
this script will generate a small amount of train and eval data used in tests that should be compact enough
to go into the git repo without bloating it
"""

import json
from pathlib import Path

train_data_path = (
    Path(__file__).parents[1] / "aki_predictions" / "ehr_prediction_modeling" / "fake_data" / "fake_patients.json"
)

with open(train_data_path, "rb") as f:
    train_data = json.load(f)

eval_data = {}
eval_data["patients"] = train_data["patients"][:2]

eval_data_path = train_data_path.parent / "eval_fake_patients.json"

with open(eval_data_path, "w") as f:
    json.dump(eval_data, f)

short_train_data = {}
short_train_data["patients"] = train_data["patients"][2:10]
short_train_data_path = train_data_path.parent / "fake_patients_short.json"
with open(short_train_data_path, "w") as f:
    json.dump(short_train_data, f)
