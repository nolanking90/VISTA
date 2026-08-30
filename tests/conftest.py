import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.pop("QT_QPA_PLATFORMTHEME", None)

import numpy as np
import pytest

from vista.sensors import Sensor


class SensorStub(Sensor):
    def can_geolocate(self):
        return False

    def get_imagery_frames_and_times(self):
        return np.array([], dtype=np.int64), np.array([], dtype="datetime64[ns]")


class TimedSensorStub(SensorStub):
    def get_imagery_frames_and_times(self):
        return np.array([2, 4, 8], dtype=np.int64), np.array(
            ["2025-01-02T03:04:05", "2025-01-02T03:04:06", "2025-01-02T03:04:07"],
            dtype="datetime64[ns]",
        )


@pytest.fixture(scope="session")
def sensor() -> Sensor:
    return SensorStub(name="test-sensor")


@pytest.fixture(scope="session")
def timed_sensor() -> Sensor:
    return TimedSensorStub(name="timed-test-sensor")
