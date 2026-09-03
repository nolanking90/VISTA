import datetime

import numpy as np
import pandas as pd
import pytest
from helpers import DATA_DIR, assert_dataframe_time_round_trip, assert_init_fields_equal
from PyQt6.QtCore import QPointF, Qt
from pytestqt.qtbot import QtBot

from vista.detections.detector import Detector
from vista.sensors import Sensor
from vista.widgets.core.data.data_loader import DataLoaderThread
from vista.widgets.core.imagery_viewer import ImageryViewer


def make_editable_detector(sensor: Sensor) -> Detector:
    return Detector(
        name="editable-detector",
        frames=np.array([1, 2], dtype=np.int64),
        rows=np.array([10.0, 20.0]),
        columns=np.array([100.0, 200.0]),
        sensor=sensor,
        labels=[{"first"}, {"second"}],
        label_times=[datetime.datetime(2025, 1, 1), datetime.datetime(2025, 1, 2)],
        labelers=["alice", "bob"],
    )


@pytest.fixture
def imagery_viewer(qtbot: QtBot) -> ImageryViewer:
    viewer = ImageryViewer()
    qtbot.addWidget(viewer)
    viewer.resize(800, 600)
    viewer.show()
    viewer.plot_item.setRange(xRange=(0, 300), yRange=(0, 30), padding=0)
    qtbot.wait(10)
    return viewer


def click_plot(qtbot: QtBot, viewer: ImageryViewer, *, row: float, column: float) -> None:
    scene_position = viewer.plot_item.vb.mapViewToScene(QPointF(column, row))
    viewport_position = viewer.graphics_layout.mapFromScene(scene_position)
    qtbot.mouseClick(
        viewer.graphics_layout.viewport(),
        Qt.MouseButton.LeftButton,
        pos=viewport_position,
    )


def finish_editing(viewer: ImageryViewer, detector: Detector) -> Detector:
    viewer.start_detection_editing(detector)
    edited = viewer.finish_detection_editing()
    assert edited is detector
    return edited


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


def test_finishing_detection_editing_preserves_labels(sensor: Sensor, imagery_viewer: ImageryViewer):
    detector = make_editable_detector(sensor)

    edited = finish_editing(imagery_viewer, detector)

    assert edited.labels == [{"first"}, {"second"}]
    assert edited.label_times == [datetime.datetime(2025, 1, 1), datetime.datetime(2025, 1, 2)]
    assert edited.labelers == ["alice", "bob"]


def test_moving_detection_preserves_its_labels(sensor: Sensor, imagery_viewer: ImageryViewer, qtbot: QtBot):
    detector = make_editable_detector(sensor)
    imagery_viewer.start_detection_editing(detector)
    imagery_viewer.set_frame_number(1)
    click_plot(qtbot, imagery_viewer, row=10.0, column=100.0)
    click_plot(qtbot, imagery_viewer, row=15.0, column=150.0)

    edited = imagery_viewer.finish_detection_editing()

    assert edited is detector
    np.testing.assert_allclose(edited.rows, [15.0, 20.0], atol=0.5)
    np.testing.assert_allclose(edited.columns, [150.0, 200.0], atol=0.5)
    assert edited.labels == [{"first"}, {"second"}]
    assert edited.labelers == ["alice", "bob"]


def test_adding_detection_does_not_clear_existing_labels(
    sensor: Sensor,
    imagery_viewer: ImageryViewer,
    qtbot: QtBot,
):
    detector = make_editable_detector(sensor)
    imagery_viewer.start_detection_editing(detector)
    imagery_viewer.set_frame_number(2)
    click_plot(qtbot, imagery_viewer, row=25.0, column=250.0)

    edited = imagery_viewer.finish_detection_editing()

    assert edited is detector
    assert edited.labels == [{"first"}, {"second"}, set()]
    assert edited.label_times == [datetime.datetime(2025, 1, 1), datetime.datetime(2025, 1, 2), None]
    assert edited.labelers == ["alice", "bob", None]


def test_deleting_detection_removes_only_its_labels(sensor: Sensor, imagery_viewer: ImageryViewer, qtbot: QtBot):
    detector = make_editable_detector(sensor)
    imagery_viewer.start_detection_editing(detector)
    imagery_viewer.set_frame_number(1)
    click_plot(qtbot, imagery_viewer, row=10.0, column=100.0)

    edited = imagery_viewer.finish_detection_editing()

    assert edited is detector
    np.testing.assert_array_equal(edited.frames, [2])
    assert edited.labels == [{"second"}]
    assert edited.label_times == [datetime.datetime(2025, 1, 2)]
    assert edited.labelers == ["bob"]


def test_detector_dataframe_coerces_label_metadata(sensor: Sensor):
    detector = Detector.from_dataframe(
        pd.DataFrame(
            {
                "Detector": ["metadata"],
                "Frames": [1],
                "Rows": [10.0],
                "Columns": [100.0],
                "Labels": [1],
                "Label Time": ["06/07/2025 08:09"],
                "Labeler": [123],
            }
        ),
        sensor,
    )

    assert detector.labels == [{"1"}]
    assert detector.label_times == [datetime.datetime(2025, 6, 7, 8, 9)]
    assert detector.labelers == ["123"]


def test_detector_dataframe_treats_invalid_label_time_as_missing(sensor: Sensor):
    detector = Detector.from_dataframe(
        pd.DataFrame(
            {
                "Detector": ["metadata"],
                "Frames": [1],
                "Rows": [10.0],
                "Columns": [100.0],
                "Label Time": ["not a time"],
            }
        ),
        sensor,
    )

    assert detector.label_times == [None]


def test_detection_csv_loader_converts_numeric_names_to_strings(tmp_path, sensor: Sensor):
    csv_path = tmp_path / "numeric-detector.csv"
    pd.DataFrame(
        {
            "Detector": [101, 101],
            "Frames": [1, 2],
            "Rows": [10.0, 20.0],
            "Columns": [100.0, 200.0],
        }
    ).to_csv(csv_path, index=False)
    loaded_detectors = []
    loader = DataLoaderThread(csv_path, "detections", "csv", sensor=sensor)
    loader.detectors_loaded.connect(loaded_detectors.extend)

    loader.run()

    assert len(loaded_detectors) == 1
    assert loaded_detectors[0].name == "101"


def test_detector_dataframe_time_round_trip(timed_sensor: Sensor):
    assert_dataframe_time_round_trip(Detector, timed_sensor)
