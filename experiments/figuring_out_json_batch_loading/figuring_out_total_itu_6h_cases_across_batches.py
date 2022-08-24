# debug into the exception of the while loop in eval_fn_on_data_split in multiple_adverse_outcomes_training,
# then use these code snippets to ensure that the true expected quantity of itu 6h window true cases exist in
# the validation set. The automated metrics differ from this value if you sum TP and FN because the framework
# excludes some results for other reasons (ignore_label, segment_label, various other possibilities during
# eval_mask generation

itu_6h_occurred = [el.max(axis=0) for el in itu_6h_labels]
itu_6h_per_batch = [sum(el) for el in itu_6h_occurred]
itu_6h_sum = sum(itu_6h_per_batch)

itu_6h_targets = [el.max(axis=0) for el in all_itu_6h_binary_targets]
itu_6h_targets_per_batch = [sum(el) for el in itu_6h_targets]
itu_6h_targets_sum = sum(itu_6h_targets_per_batch)

max_target_by_record_number = {}
for rns_i, record_number_set in enumerate(record_numbers_by_batch):
    for rn_i, record_number in enumerate(record_number_set):
        if record_number in max_target_by_record_number:
            max_target_by_record_number[record_number] = max(
                max_target_by_record_number[record_number], itu_6h_occurred[rns_i][rn_i]
            )
        else:
            max_target_by_record_number[record_number] = itu_6h_occurred[rns_i][rn_i]

itu_6h_sum = 0
for key, item in max_target_by_record_number.items():
    itu_6h_sum += item
