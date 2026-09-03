import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import assert_constructor_fields_equal

from vista.detections.detector import Detector
from vista.sensors import Sensor


def test_deserialize_csv(sensor: Sensor):
    csv_path = Path(__file__).parent / "data" / "detector_v1.13.0.csv"
    expected = Detector(
        name="main-detector",
        frames=np.array([3, 7, 11]),
        rows=np.array([10.25, 20.5, 30.75]),
        columns=np.array([101.5, 202.25, 303.0]),
        sensor=sensor,
        color="cyan",
        marker="x",
        marker_size=13,
        line_thickness=4,
        visible=False,
        complete=True,
        labels=[{"confirmed", "vehicle"}, set(), {"review"}],
        label_times=[
            datetime.datetime(2025, 1, 2, 3, 4, 5),
            None,
            datetime.datetime(2025, 6, 7, 8, 9, 10),
        ],
        labelers=["alice", None, "bob"],
    )

    detector = Detector.from_dataframe(pd.read_csv(csv_path), sensor)

    assert_constructor_fields_equal(detector, expected)


def test_dataframe_round_trip(sensor: Sensor):
    expected = Detector(
        name="round-trip-detector",
        frames=np.array([2, 4, 8]),
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

    detector = Detector.from_dataframe(expected.to_dataframe(), sensor)

    assert_constructor_fields_equal(detector, expected)


def test_get_times(timed_sensor: Sensor):
    detector = Detector(
        name="timed-detector",
        frames=np.array([2, 3, 8]),
        rows=np.array([12.5, 24.0, 48.75]),
        columns=np.array([120.0, 240.25, 480.5]),
        sensor=timed_sensor,
    )

    times = detector.get_times()

    np.testing.assert_array_equal(
        times,
        np.array(
            ["2025-01-02T03:04:05", "NaT", "2025-01-02T03:04:07"],
            dtype="datetime64[ns]",
        ),
    )


def test_time_round_trip(timed_sensor: Sensor):
    expected = Detector(
        name="time-round-trip",
        frames=np.array([2, 4, 8]),
        rows=np.array([12.5, 24.0, 48.75]),
        columns=np.array([120.0, 240.25, 480.5]),
        sensor=timed_sensor,
        labels=[set(), set(), set()],
        label_times=[None, None, None],
        labelers=[None, None, None],
    )

    dataframe = expected.to_dataframe()
    detector = Detector.from_dataframe(dataframe.drop(columns="Frames"), timed_sensor)

    assert dataframe["Times"].tolist() == [
        "2025-01-02T03:04:05.000000",
        "2025-01-02T03:04:06.000000",
        "2025-01-02T03:04:07.000000",
    ]
    assert_constructor_fields_equal(detector, expected)


def test_frames_take_precedence(sensor: Sensor):
    dataframe = pd.DataFrame(
        {
            "Detector": ["frames-first", "frames-first"],
            "Frames": [5, 9],
            "Times": ["not a time", "also not a time"],
            "Rows": [10.0, 20.0],
            "Columns": [100.0, 200.0],
        }
    )

    detector = Detector.from_dataframe(dataframe, sensor)

    np.testing.assert_array_equal(detector.frames, [5, 9])


@pytest.mark.parametrize(
    ("times", "use_timed_sensor", "message"),
    [
        (None, False, "must have either 'Frames' or 'Times'"),
        (["2025-01-02T03:04:05"], False, "Sensor imagery times are required"),
        (["2024-01-02T03:04:05"], True, "times are not within the bounds"),
    ],
)
def test_invalid_time_dataframe(times, use_timed_sensor, message, sensor: Sensor, timed_sensor: Sensor):
    data = {
        "Detector": ["invalid-times"],
        "Rows": [10.0],
        "Columns": [100.0],
    }
    if times is not None:
        data["Times"] = times
    selected_sensor = timed_sensor if use_timed_sensor else sensor

    with pytest.raises(ValueError, match=message):
        Detector.from_dataframe(pd.DataFrame(data), selected_sensor)
