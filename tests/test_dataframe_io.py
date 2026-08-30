import datetime
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vista.detections.detector import Detector
from vista.sensors import Sensor
from vista.tracks.track import Track

DATA_DIR = Path(__file__).parent / "data"
# The CSV fixtures were exported by the main branch at ca04ae5.


class SensorStub(Sensor):
    def can_geolocate(self):
        return False

    def get_imagery_frames_and_times(self):
        return np.array([], dtype=np.int64), np.array([], dtype="datetime64[ns]")


SENSOR = SensorStub(name="test-sensor")


def assert_init_fields_equal(actual, expected):
    """Compare every constructor field while ignoring generated IDs and caches."""
    assert type(actual) is type(expected)

    for data_field in fields(expected):
        if not data_field.init:
            continue

        actual_value = getattr(actual, data_field.name)
        expected_value = getattr(expected, data_field.name)
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value, data_field.name


def test_detector_deserializes_csv_exported_by_main():
    detector = Detector.from_dataframe(pd.read_csv(DATA_DIR / "main_detector.csv"), SENSOR)

    assert detector.name == "main-detector"
    np.testing.assert_array_equal(detector.frames, [3, 7, 11])
    np.testing.assert_allclose(detector.rows, [10.25, 20.5, 30.75])
    np.testing.assert_allclose(detector.columns, [101.5, 202.25, 303.0])
    assert detector.sensor is SENSOR
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


def test_detector_dataframe_round_trip():
    original = Detector(
        name="round-trip-detector",
        frames=np.array([2, 4, 8], dtype=np.int64),
        rows=np.array([12.5, 24.0, 48.75]),
        columns=np.array([120.0, 240.25, 480.5]),
        sensor=SENSOR,
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

    restored = Detector.from_dataframe(original.to_dataframe(), SENSOR)

    assert_init_fields_equal(restored, original)


def test_track_deserializes_csv_exported_by_main():
    track = Track.from_dataframe(pd.read_csv(DATA_DIR / "main_track.csv"), SENSOR)

    assert track.name == "main-track"
    np.testing.assert_array_equal(track.frames, [5, 6, 9])
    np.testing.assert_allclose(track.rows, [40.0, 41.5, 43.25])
    np.testing.assert_allclose(track.columns, [400.5, 402.0, 405.75])
    assert track.sensor is SENSOR
    assert track.description == ""
    assert track.color == "magenta"
    assert track.marker == "d"
    assert track.marker_size == 15
    assert track.line_thickness == 2
    assert bool(track.visible) is False
    assert bool(track.complete) is True
    assert track.labels == [{"aircraft", "confirmed"}] * 3
    assert track.label_times == [datetime.datetime(2025, 2, 3, 4, 5, 6)] * 3
    assert track.labelers == ["legacy-user"] * 3
    assert track.line_width == 5
    assert track.tail_length == 8
    assert bool(track.show_line) is False
    assert track.line_style == "DashLine"
    assert track.tracker == "legacy-tracker"
    assert track.extraction_metadata is None
    np.testing.assert_allclose(track.covariance_00, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(track.covariance_01, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(track.covariance_11, [7.0, 8.0, 9.0])
    assert bool(track.show_uncertainty) is True


def test_track_dataframe_round_trip():
    original = Track(
        name="round-trip-track",
        frames=np.array([1, 3, 7], dtype=np.int64),
        rows=np.array([15.0, 18.5, 23.25]),
        columns=np.array([150.5, 154.0, 160.75]),
        sensor=SENSOR,
        color="blue",
        marker="s",
        marker_size=14,
        visible=False,
        complete=True,
        labels=[{"aircraft", "review"}] * 3,
        label_times=[datetime.datetime(2025, 4, 5, 6, 7, 8)] * 3,
        labelers=["alice"] * 3,
        line_width=6,
        tail_length=12,
        show_line=False,
        line_style="DotLine",
        tracker="round-trip-tracker",
        covariance_00=np.array([1.0, 2.0, 3.0]),
        covariance_01=np.array([0.1, 0.2, 0.3]),
        covariance_11=np.array([4.0, 5.0, 6.0]),
        show_uncertainty=True,
    )

    restored = Track.from_dataframe(original.to_dataframe(), SENSOR)

    assert_init_fields_equal(restored, original)


def test_track_label_interface_broadcasts_metadata():
    track = Track(
        name="labeled-track",
        frames=np.array([1, 3, 7], dtype=np.int64),
        rows=np.array([15.0, 18.5, 23.25]),
        columns=np.array([150.5, 154.0, 160.75]),
        sensor=SENSOR,
    )
    label_time = datetime.datetime(2025, 4, 5, 6, 7, 8)

    track.set_label({"aircraft", "review"}, label_time, "alice")

    assert track.label == {"aircraft", "review"}
    assert track.label_time == label_time
    assert track.labeler == "alice"
    assert track.labels == [{"aircraft", "review"}] * 3
    assert track.label_times == [label_time] * 3
    assert track.labelers == ["alice"] * 3

    returned_labels = track.label
    returned_labels.add("local change")
    assert track.label == {"aircraft", "review"}


def test_track_rejects_per_point_label_metadata():
    def make_track(
        *,
        labels: list[set[str]] | None = None,
        label_times: list[datetime.datetime | None] | None = None,
        labelers: list[str | None] | None = None,
    ) -> Track:
        return Track(
            name="invalid-track",
            frames=np.array([1, 2], dtype=np.int64),
            rows=np.array([10.0, 20.0]),
            columns=np.array([100.0, 200.0]),
            sensor=SENSOR,
            labels=[] if labels is None else labels,
            label_times=[] if label_times is None else label_times,
            labelers=[] if labelers is None else labelers,
        )

    label_time = datetime.datetime(2025, 4, 5, 6, 7, 8)

    with pytest.raises(ValueError, match="Track labels must be identical"):
        make_track(labels=[{"aircraft"}, {"review"}])
    with pytest.raises(ValueError, match="Track label times must be identical"):
        make_track(label_times=[label_time, None])
    with pytest.raises(ValueError, match="Track labelers must be identical"):
        make_track(labelers=["alice", "bob"])
