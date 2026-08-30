import datetime

import numpy as np
import pandas as pd
import pytest
from helpers import DATA_DIR, assert_dataframe_time_round_trip, assert_init_fields_equal

from vista.detections.detector import Detector
from vista.sensors import Sensor


def test_detector_deserializes_csv_exported_by_main(sensor: Sensor):
    detector = Detector.from_dataframe(pd.read_csv(DATA_DIR / "main_detector.csv"), sensor)

    assert detector.name == "main-detector"
    np.testing.assert_array_equal(detector.frames, [3, 7, 11])
    np.testing.assert_allclose(detector.rows, [10.25, 20.5, 30.75])
    np.testing.assert_allclose(detector.columns, [101.5, 202.25, 303.0])
    assert detector.sensor is sensor
    assert detector.description == ""
    assert detector.color == "cyan"
    assert detector.marker == "x"
    assert detector.marker_size == 13
    assert detector.line_thickness == 4
    assert bool(detector.visible) is False
    assert bool(detector.complete) is True
    assert detector.labels == [{"confirmed", "vehicle"}, set(), {"review"}]
    assert detector.label_times == [
        datetime.datetime(2025, 1, 2, 3, 4, 5),
        None,
        datetime.datetime(2025, 6, 7, 8, 9, 10),
    ]
    assert detector.labelers == ["alice", None, "bob"]


def test_detector_dataframe_round_trip(sensor: Sensor):
    original = Detector(
        name="round-trip-detector",
        frames=np.array([2, 4, 8], dtype=np.int64),
        rows=np.array([12.5, 24.0, 48.75]),
        columns=np.array([120.0, 240.25, 480.5]),
        sensor=sensor,
        color="yellow",
        marker="+",
        marker_size=17,
        line_thickness=3,
        visible=False,
        complete=True,
        labels=[{"first"}, set(), {"last", "review"}],
        label_times=[
            datetime.datetime(2025, 3, 4, 5, 6, 7),
            None,
            datetime.datetime(2025, 8, 9, 10, 11, 12),
        ],
        labelers=["alice", None, "charlie"],
    )

    restored = Detector.from_dataframe(original.to_dataframe(), sensor)

    assert_init_fields_equal(restored, original)


def test_detector_copy_preserves_fields_and_independence(sensor: Sensor):
    original = Detector(
        name="detector-copy",
        frames=np.array([2, 4], dtype=np.int64),
        rows=np.array([12.5, 24.0]),
        columns=np.array([120.0, 240.25]),
        sensor=sensor,
        description="copied detector",
        color="yellow",
        marker="+",
        marker_size=17,
        line_thickness=3,
        visible=False,
        complete=True,
        labels=[{"first"}, {"last"}],
        label_times=[datetime.datetime(2025, 3, 4, 5, 6, 7), None],
        labelers=["alice", None],
    )

    copied = original.copy()

    assert_init_fields_equal(copied, original)
    assert copied.sensor is original.sensor
    assert copied.uuid != original.uuid
    assert copied.frames is not original.frames
    assert copied.labels[0] is not original.labels[0]

    copied.frames[0] = 99
    copied.labels[0].add("copy only")
    assert original.frames[0] == 2
    assert "copy only" not in original.labels[0]


@pytest.mark.parametrize(
    ("selection", "expected_indices"),
    [
        (slice(1, 4, 2), np.array([1, 3])),
        (np.array([True, False, True, False]), np.array([0, 2])),
        (np.array([3, 1, 1]), np.array([3, 1, 1])),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
    ],
)
def test_detector_selection_keeps_fields_aligned(selection, expected_indices, sensor: Sensor):
    label_times: list[datetime.datetime | None] = [datetime.datetime(2025, 1, day) for day in range(1, 5)]
    detector = Detector(
        name="selected-detector",
        frames=np.array([2, 4, 6, 8]),
        rows=np.array([12.0, 14.0, 16.0, 18.0]),
        columns=np.array([102.0, 104.0, 106.0, 108.0]),
        sensor=sensor,
        labels=[{"zero"}, {"one"}, {"two"}, {"three"}],
        label_times=label_times,
        labelers=["alice", "bob", "charlie", "dana"],
    )
    detector._cached_lons = np.array([-105.0, -104.0, -103.0, -102.0])
    detector._cached_lats = np.array([39.0, 40.0, 41.0, 42.0])

    selected = detector[selection]

    np.testing.assert_array_equal(selected.frames, detector.frames[expected_indices])
    np.testing.assert_array_equal(selected.rows, detector.rows[expected_indices])
    np.testing.assert_array_equal(selected.columns, detector.columns[expected_indices])
    assert selected.labels == [detector.labels[i] for i in expected_indices]
    assert selected.label_times == [detector.label_times[i] for i in expected_indices]
    assert selected.labelers == [detector.labelers[i] for i in expected_indices]
    assert selected._cached_lons is not None
    assert selected._cached_lats is not None
    np.testing.assert_array_equal(selected._cached_lons, detector._cached_lons[expected_indices])
    np.testing.assert_array_equal(selected._cached_lats, detector._cached_lats[expected_indices])

    if len(selected) > 0:
        selected.labels[0].add("selected only")
        assert "selected only" not in detector.labels[expected_indices[0]]


def test_detector_selection_rejects_scalar_indices(sensor: Sensor):
    detector = Detector(
        name="selected-detector",
        frames=np.array([2]),
        rows=np.array([12.0]),
        columns=np.array([102.0]),
        sensor=sensor,
    )

    with pytest.raises(TypeError):
        detector[0]
    with pytest.raises(TypeError):
        detector[np.array(0)]


def test_detector_dataframe_time_round_trip(timed_sensor: Sensor):
    assert_dataframe_time_round_trip(Detector, timed_sensor)
