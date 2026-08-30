from dataclasses import fields
from pathlib import Path

import numpy as np

from vista.detections.detector import Detector
from vista.sensors import Sensor

DATA_DIR = Path(__file__).parent / "data"
# The CSV fixtures were exported by the main branch at ca04ae5.


def assert_init_fields_equal(actual, expected) -> None:
    """Compare every constructor field while ignoring generated IDs and caches."""
    assert type(actual) is type(expected)

    for data_field in fields(expected):
        if not data_field.init:
            continue

        actual_value = getattr(actual, data_field.name)
        expected_value = getattr(expected, data_field.name)
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        elif isinstance(expected_value, dict):
            assert actual_value.keys() == expected_value.keys()
            for key, expected_item in expected_value.items():
                actual_item = actual_value[key]
                if isinstance(expected_item, np.ndarray):
                    np.testing.assert_array_equal(actual_item, expected_item)
                else:
                    assert actual_item == expected_item
        else:
            assert actual_value == expected_value, data_field.name


def assert_dataframe_time_round_trip(object_type: type[Detector], timed_sensor: Sensor) -> None:
    original = object_type(
        name="time-round-trip",
        frames=np.array([2, 4, 8], dtype=np.int64),
        rows=np.array([12.5, 24.0, 48.75]),
        columns=np.array([120.0, 240.25, 480.5]),
        sensor=timed_sensor,
    )

    df = original.to_dataframe()
    assert df["Times"].tolist() == [
        "2025-01-02T03:04:05.000000",
        "2025-01-02T03:04:06.000000",
        "2025-01-02T03:04:07.000000",
    ]

    restored = object_type.from_dataframe(df.drop(columns="Frames"), timed_sensor)
    assert_init_fields_equal(restored, original)
