import numpy as np

from scripts.build_clean_v4_dataset import (
    apply_decisions_to_encoded,
    apply_label_interval,
)


def test_label_bit_update_preserves_stretch_values():
    stretch = np.asarray([10, 11, 12, 13], dtype=np.int64)
    encoded = (stretch << 1) | np.asarray([0, 1, 0, 1], dtype=np.int64)

    updated = apply_label_interval(encoded, 1, 3, 1)

    assert np.array_equal(updated >> 1, stretch)


def test_set_cough_updates_only_requested_interval():
    encoded = (np.arange(6, dtype=np.int64) << 1)
    decisions = [{"start_sec": 0.002, "end_sec": 0.004, "decision": "set_cough"}]

    updated = apply_decisions_to_encoded(encoded, decisions, fs_audio=1000)

    assert np.array_equal(updated & 1, np.asarray([0, 0, 1, 1, 0, 0]))


def test_set_non_cough_updates_only_requested_interval():
    encoded = (np.arange(6, dtype=np.int64) << 1) | 1
    decisions = [{"start_sec": 0.001, "end_sec": 0.003, "decision": "set_non_cough"}]

    updated = apply_decisions_to_encoded(encoded, decisions, fs_audio=1000)

    assert np.array_equal(updated & 1, np.asarray([1, 0, 0, 1, 1, 1]))


def test_blank_and_keep_decisions_do_not_change_labels():
    encoded = (np.arange(6, dtype=np.int64) << 1) | np.asarray([0, 1, 0, 1, 0, 1])
    decisions = [
        {"start_sec": 0.000, "end_sec": 0.003, "decision": ""},
        {"start_sec": 0.003, "end_sec": 0.006, "decision": "keep"},
    ]

    updated = apply_decisions_to_encoded(encoded, decisions, fs_audio=1000)

    assert np.array_equal(updated, encoded)
