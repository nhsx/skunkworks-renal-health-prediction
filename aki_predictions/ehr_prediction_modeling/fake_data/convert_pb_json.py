# Script to convert fake_data protocol buffers into json objects and output to file.
#
# Requires installation of the protobuf compiler and compilation of the fake data definitions.
#     ```sudo apt-get install protobuf-compiler```
#     ```protoc -I=aki_predictions/ehr_prediction_modeling \
#           --python_out=aki_predictions/ehr_prediction_modeling \
#           aki_predictions/ehr_prediction_modeling/proto/*.proto```
#
# Execute from top level directory.
#     ```python aki_predictions/ehr_prediction_modeling/fake_data/convert_pb_json.py```
import os

from google.protobuf.json_format import MessageToJson

from aki_predictions.ehr_prediction_modeling.proto import (
    fake_patient_pb2,
    fake_records_pb2,
)

# Define data location
data_dirpath = os.path.join(os.getcwd(), "aki_predictions", "ehr_prediction_modeling", "fake_data")

raw_patients_path = os.path.join(data_dirpath, "fake_patients.pb")

# Open patients data
with open(raw_patients_path, "rb") as f:
    patients = fake_patient_pb2.FakePatients.FromString(f.read())  # .records

print(type(patients))

# Save individual patient records
for i, patient in enumerate(patients.patients):
    if i < 10:  # Only export first 10 patients individually
        json_obj = MessageToJson(patient, preserving_proto_field_name=True)
        with open(os.path.join(data_dirpath, "fake_patients", f"fake_patient_{i}.json"), "w") as outfile:
            outfile.write(json_obj)

# Export all patient records
json_obj = MessageToJson(patients, preserving_proto_field_name=True)

with open(os.path.join(data_dirpath, "fake_patients.json"), "w") as outfile:
    outfile.write(json_obj)


# Do the same for the raw records
raw_records_path = os.path.join(data_dirpath, "fake_raw_records.pb")

# Open patients data
with open(raw_records_path, "rb") as f:
    records = fake_records_pb2.FakeRecords.FromString(f.read())  # .records

print(type(records))

# Save individual patient records
for i, record in enumerate(records.records):
    if i < 10:  # Only export first 10 records individually
        json_obj = MessageToJson(record, preserving_proto_field_name=True)
        with open(os.path.join(data_dirpath, "fake_records", f"fake_record_{i}.json"), "w") as outfile:
            outfile.write(json_obj)

# Export all patient records
json_obj = MessageToJson(records, preserving_proto_field_name=True)

with open(os.path.join(data_dirpath, "fake_records.json"), "w") as outfile:
    outfile.write(json_obj)
