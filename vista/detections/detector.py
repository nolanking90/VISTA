import datetime
import pathlib
import uuid as uuid_module
from dataclasses import field, fields
from typing import TYPE_CHECKING, Annotated, ClassVar, Optional, Self, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from numpy.typing import NDArray
from pydantic import AliasChoices, AliasPath, ConfigDict, Field, SkipValidation, TypeAdapter, field_serializer
from pydantic import field_validator as pydantic_field_validator
from pydantic.dataclasses import dataclass

from vista.sensors.sensor import Sensor
from vista.utils.time_mapping import map_times_to_frames

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

PYDANTIC_CONFIG = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_by_name=True)


def dataframe_field(field_name: str, column_name: str, *, scalar: bool = False, **kwargs):
    """Declare how a dataclass field is represented in a DataFrame."""
    dataframe_alias = AliasPath(column_name, 0) if scalar else column_name
    return Field(
        validation_alias=AliasChoices(field_name, dataframe_alias),
        serialization_alias=column_name,
        **kwargs,
    )


@dataclass(config=PYDANTIC_CONFIG)
class Detector:
    """
    Collection of detection points from a detection algorithm or manual creation.

    A Detector represents a set of detected objects or points of interest across
    multiple frames. Unlike Tracks, detections are unassociated points without
    temporal continuity. Each detection point can have its own set of labels.

    Parameters
    ----------
    name : str
        Unique identifier for this detector
    frames : NDArray[np.int_]
        Frame numbers where detections occur
    rows : NDArray[np.float64]
        Row (vertical) pixel coordinates for each detection
    columns : NDArray[np.float64]
        Column (horizontal) pixel coordinates for each detection
    sensor : Sensor
        Sensor object associated with these detections
    description : str, optional
        Description of detection algorithm or method, by default ""

    Attributes
    ----------
    color : str, optional
        Color for detection markers, by default 'r' (red)
    marker : str, optional
        Marker style ('o', 's', 't', 'd', '+', 'x', 'star'), by default 'o' (circle)
    marker_size : int, optional
        Size of detection markers, by default 10
    line_thickness : int, optional
        Thickness of marker outline, by default 2
    visible : bool, optional
        Whether detections are visible in viewer, by default True
    labels : list[set[str]], optional
        List of label sets, one set per detection point, by default empty list

    Methods
    -------
    __getitem__(slice)
        Slice detector by index or boolean mask
    from_dataframe(df, sensor, name)
        Create Detector from pandas DataFrame
    copy()
        Create a deep copy of the detector
    to_csv(file)
        Save detector to CSV file
    to_dataframe()
        Convert detector to pandas DataFrame
    get_unique_labels()
        Get all unique labels across all detections

    Notes
    -----
    - Detections are unassociated points (unlike tracks which represent trajectories)
    - Multiple detections can exist at the same frame
    - Labels are per-detection, allowing individual detection categorization
    - Detection coordinates are always in pixel space (row/column)
    """

    if TYPE_CHECKING:
        __pydantic_fields__: ClassVar[dict[str, FieldInfo]]

    name: Annotated[str, dataframe_field("name", "Detector", scalar=True)]
    frames: Annotated[NDArray[np.int_], dataframe_field("frames", "Frames")]
    rows: Annotated[NDArray[np.float64], dataframe_field("rows", "Rows")]
    columns: Annotated[NDArray[np.float64], dataframe_field("columns", "Columns")]
    sensor: Annotated[SkipValidation[Sensor], Field(exclude=True)]
    description: str = Field(default="", exclude=True)

    # Styling attributes
    color: str = dataframe_field("color", "Color", scalar=True, default="r")  # Red by default
    marker: str = dataframe_field("marker", "Marker", scalar=True, default="o")  # Circle by default
    marker_size: int = dataframe_field("marker_size", "Marker Size", scalar=True, default=10)
    line_thickness: int = dataframe_field("line_thickness", "Line Thickness", scalar=True, default=2)
    visible: bool = dataframe_field("visible", "Visible", scalar=True, default=True)
    complete: bool = dataframe_field("complete", "Complete", scalar=True, default=False)

    labels: list[set[str]] = dataframe_field("labels", "Labels", default_factory=list)
    label_times: list[Optional[datetime.datetime]] = dataframe_field(
        "label_times",
        "Label Time",
        default_factory=list,
    )
    labelers: list[Optional[str]] = dataframe_field("labelers", "Labeler", default_factory=list)

    # Performance optimization: cached data structures
    _frame_index: Optional[dict] = field(default=None, init=False, repr=False)  # Frame number -> detection indices
    _cached_pen: object = field(default=None, init=False, repr=False)  # Cached PyQtGraph pen
    _pen_params: Optional[tuple] = field(default=None, init=False, repr=False)  # Parameters used for cached pen
    _cached_lons: Optional[NDArray[np.float64]] = field(default=None, init=False, repr=False)  # Cached longitude coords
    _cached_lats: Optional[NDArray[np.float64]] = field(default=None, init=False, repr=False)  # Cached latitude coords
    uuid: Optional[uuid_module.UUID] = field(init=False, default=None)

    @pydantic_field_validator("labels", mode="before")
    @classmethod
    def _parse_labels(cls, value, info):
        point_count = len(info.data.get("frames", []))
        if isinstance(value, set):
            values = [value.copy() for _ in range(point_count)]
        elif isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            values = [value] * point_count
        else:
            values = list(value)

        labels = []
        for item in values:
            if isinstance(item, set):
                labels.append(item.copy())
            elif item is None or (not isinstance(item, (list, tuple, set, dict)) and pd.isna(item)):
                labels.append(set())
            elif isinstance(item, str):
                labels.append({label.strip() for label in item.split(",") if label.strip()})
            else:
                labels.append(set(item))
        return labels

    @field_serializer("labels")
    def _serialize_labels(self, value):
        return [", ".join(sorted(labels)) for labels in value]

    @field_serializer("label_times")
    def _serialize_label_times(self, value):
        return [label_time.isoformat() if label_time is not None else "" for label_time in value]

    @field_serializer("labelers")
    def _serialize_labelers(self, value):
        return [labeler or "" for labeler in value]

    def __post_init__(self):
        point_count = len(self.frames)
        if not self.labels:
            self.labels = [set() for _ in range(point_count)]
        if not self.label_times:
            self.label_times = [None] * point_count
        if not self.labelers:
            self.labelers = [None] * point_count

        for field_name in ("labels", "label_times", "labelers"):
            if len(getattr(self, field_name)) != point_count:
                raise ValueError(f"{field_name} must contain one value per point")

        self.uuid = uuid_module.uuid4()

    # TODO: Test that splitting a track into detections or creating a track from detections does not reuse UUIDs.
    def __eq__(self, other):
        if not isinstance(other, Detector):
            return False
        return self.uuid == other.uuid

    def _build_frame_index(self):
        """Build index mapping frame numbers to detection indices for O(1) lookup."""
        if self._frame_index is None:
            self._frame_index = {}
            for i, frame in enumerate(self.frames):
                if frame not in self._frame_index:
                    self._frame_index[frame] = []
                self._frame_index[frame].append(i)

    def get_detections_at_frame(self, frame_num):
        """
        Get detection coordinates at a specific frame using O(1) cached lookup.

        Parameters
        ----------
        frame_num : int
            Frame number to query

        Returns
        -------
        rows : NDArray
            Row coordinates of detections at this frame
        cols : NDArray
            Column coordinates of detections at this frame
        """
        self._build_frame_index()
        indices = self._frame_index.get(frame_num, [])
        if len(indices) > 0:
            return self.rows[indices], self.columns[indices]
        return np.array([]), np.array([])

    def get_geodetic_coords(self) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Get geodetic coordinates for all detection points, computing and caching if needed.

        Projects each detection point using its own frame's sensor geometry, so the
        result represents the true geographic location. The result is cached so
        subsequent calls return immediately.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]] or None
            (longitudes, latitudes) in degrees, or None if the sensor cannot geolocate.
        """
        if self._cached_lons is not None and self._cached_lats is not None:
            return self._cached_lons, self._cached_lats

        if not self.sensor or not self.sensor.can_geolocate():
            return None

        # Single vectorized call — sensor handles frame grouping internally
        locations = self.sensor.pixel_to_geodetic(self.frames, self.rows, self.columns)
        lons = np.asarray(locations.lon.deg, dtype=np.float64)
        lats = np.asarray(locations.lat.deg, dtype=np.float64)

        # Set invalid locations to NaN
        invalid = (locations.y.value == 0) & (locations.z.value == 0)
        lons[invalid] = np.nan
        lats[invalid] = np.nan

        self._cached_lons = lons
        self._cached_lats = lats
        return self._cached_lons, self._cached_lats

    def get_times(self) -> Optional[NDArray[np.datetime64]]:
        """Return imagery timestamps corresponding to each frame, or ``None`` if unavailable."""
        sensor_imagery_frames, sensor_imagery_times = self.sensor.get_imagery_frames_and_times()
        if len(sensor_imagery_times) < 1:
            return None

        indices = np.searchsorted(sensor_imagery_frames, self.frames)
        times = np.full(len(self.frames), np.datetime64("NaT"), dtype="datetime64[ns]")

        in_bounds = indices < len(sensor_imagery_frames)
        clipped = np.minimum(indices, len(sensor_imagery_frames) - 1)
        valid_mask = in_bounds & (sensor_imagery_frames[clipped] == self.frames)
        times[valid_mask] = sensor_imagery_times[indices[valid_mask]]
        return times

    def invalidate_caches(self):
        """Invalidate cached data structures when detector data changes."""
        self._frame_index = None
        self._cached_pen = None
        self._pen_params = None
        self._cached_lons = None
        self._cached_lats = None

    def get_pen(self, width=None, **kwargs):
        """
        Get cached PyQtGraph pen object, creating only if parameters changed.

        Parameters
        ----------
        width : int, optional
            Line width override, uses self.line_thickness if None

        Returns
        -------
        pg.mkPen
            PyQtGraph pen object
        """

        actual_width = width if width is not None else self.line_thickness
        params = (self.color, actual_width)

        if self._pen_params != params:
            self._cached_pen = pg.mkPen(color=self.color, width=actual_width)
            self._pen_params = params

        return self._cached_pen

    def __getitem__(self, s):
        if isinstance(s, slice) or isinstance(s, np.ndarray):
            # Handle slice objects
            detector_slice = self.copy()
            detector_slice.frames = detector_slice.frames[s]
            detector_slice.rows = detector_slice.rows[s]
            detector_slice.columns = detector_slice.columns[s]
            # Subset labels if they exist
            if len(detector_slice.labels) > 0:
                if isinstance(s, slice):
                    detector_slice.labels = detector_slice.labels[s]
                else:  # numpy array boolean mask or indices
                    detector_slice.labels = (
                        [
                            detector_slice.labels[i]
                            for i in np.where(s)[0]
                            if isinstance(s, np.ndarray) and s.dtype == bool
                        ]
                        if isinstance(s, np.ndarray) and s.dtype == bool
                        else [detector_slice.labels[i] for i in s]
                    )
            # Subset per-detection label metadata (same indexing rules as labels)
            if len(detector_slice.label_times) > 0:
                if isinstance(s, slice):
                    detector_slice.label_times = detector_slice.label_times[s]
                else:
                    detector_slice.label_times = (
                        [
                            detector_slice.label_times[i]
                            for i in np.where(s)[0]
                            if isinstance(s, np.ndarray) and s.dtype == bool
                        ]
                        if isinstance(s, np.ndarray) and s.dtype == bool
                        else [detector_slice.label_times[i] for i in s]
                    )
            if len(detector_slice.labelers) > 0:
                if isinstance(s, slice):
                    detector_slice.labelers = detector_slice.labelers[s]
                else:
                    detector_slice.labelers = (
                        [
                            detector_slice.labelers[i]
                            for i in np.where(s)[0]
                            if isinstance(s, np.ndarray) and s.dtype == bool
                        ]
                        if isinstance(s, np.ndarray) and s.dtype == bool
                        else [detector_slice.labelers[i] for i in s]
                    )
            # Slice cached geodetic coords if present
            if detector_slice._cached_lons is not None:
                detector_slice._cached_lons = detector_slice._cached_lons[s]
            if detector_slice._cached_lats is not None:
                detector_slice._cached_lats = detector_slice._cached_lats[s]
            return detector_slice
        else:
            raise TypeError("Invalid index or slice type.")

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = f"{self.__class__.__name__}({self.name})"
        s += "\n" + len(s) * "-" + "\n"
        s += str(self.to_dataframe())
        return s

    @classmethod
    def normalize_dataframe(cls, df: pd.DataFrame, sensor: Sensor, name: str) -> pd.DataFrame:
        """Return a copy with times mapped to frames and blank label metadata normalized.

        Rows outside the sensor's time bounds are dropped; spatial coordinates are unchanged.
        """

        df = df.copy()

        # Determine frames - priority: Frames column > time-to-frame mapping
        if "Frames" in df.columns:
            frames = df["Frames"].to_numpy()
        elif "Times" in df.columns:
            times = pd.to_datetime(df["Times"]).to_numpy()
            sensor_imagery_frames, sensor_imagery_times = sensor.get_imagery_frames_and_times()
            if len(sensor_imagery_times) == 0:
                raise ValueError(
                    f"{cls.__name__} '{name}' has times but no frames. "
                    "Sensor imagery times are required for time-to-frame mapping."
                )

            # Eliminate detections outside the time bounds of the selected sensor
            df = df[(times >= sensor_imagery_times[0]) & (times <= sensor_imagery_times[-1])]
            if len(df) == 0:
                raise ValueError(f"{cls.__name__} '{name}' times are not within the bounds of the selected imagery.")

            times = pd.to_datetime(df["Times"]).to_numpy()
            frames = map_times_to_frames(times, sensor_imagery_times, sensor_imagery_frames)
        else:
            raise ValueError(f"{cls.__name__} '{name}' must have either 'Frames' or 'Times' column")

        df["Frames"] = frames
        for column in ("Label Time", "Labeler"):
            if column in df.columns:
                values = df[column].astype(object)
                df[column] = values.where(values.notna() & values.ne(""), None)
        return df

    @classmethod
    def _from_normalized_dataframe(
        cls,
        df: pd.DataFrame,
        sensor,
        name: str,
        **additional_kwargs,
    ) -> Self:
        """Construct an instance from normalized frame and pixel coordinates."""
        required_columns = {"Frames", "Rows", "Columns"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{cls.__name__} '{name}' is missing required columns: {missing}")

        data = {"name": name, "sensor": sensor, **additional_kwargs}
        for field_name, field_info in cls.__pydantic_fields__.items():
            column = field_info.serialization_alias
            if field_name not in data and column is not None and column in df.columns:
                data[column] = df[column].to_numpy()
        detector = TypeAdapter(cls).validate_python(data)

        # Pre-populate geodetic cache if coords were available in the dataframe
        if "Latitude (deg)" in df.columns and "Longitude (deg)" in df.columns:
            detector._cached_lons = df["Longitude (deg)"].to_numpy(dtype=np.float64)
            detector._cached_lats = df["Latitude (deg)"].to_numpy(dtype=np.float64)

        return detector

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, sensor, name: str | None = None):
        """
        Create Detector from pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing detection data with required columns:
            "Detector", "Rows", "Columns", and either "Frames" or "Times"
        sensor : Sensor
            Sensor object for these detections
        name : str, optional
            Detector name, by default taken from df["Detector"]

        Returns
        -------
        Detector
            New Detector object

        Notes
        -----
        Optional styling columns: "Color", "Marker", "Marker Size",
        "Line Thickness", "Visible", "Labels"

        Labels should be comma-separated strings in the "Labels" column.
        """
        if name is None:
            name = df["Detector"][0]

        df = cls.normalize_dataframe(df, sensor, name)
        return cls._from_normalized_dataframe(df, sensor, name)

    def copy(self) -> Self:
        """Return an independent copy that retains the same sensor."""
        detector_copy = type(self)(
            name=self.name,
            frames=self.frames.copy(),
            rows=self.rows.copy(),
            columns=self.columns.copy(),
            sensor=self.sensor,
            description=self.description,
            color=self.color,
            marker=self.marker,
            marker_size=self.marker_size,
            line_thickness=self.line_thickness,
            visible=self.visible,
            complete=self.complete,
            labels=[label_set.copy() for label_set in self.labels],
            label_times=self.label_times.copy(),
            labelers=self.labelers.copy(),
        )

        # Preserve cached geodetic coords
        if self._cached_lons is not None:
            detector_copy._cached_lons = self._cached_lons.copy()
        if self._cached_lats is not None:
            detector_copy._cached_lats = self._cached_lats.copy()
        return detector_copy

    def to_csv(self, file: Union[str, pathlib.Path]):
        self.to_dataframe().to_csv(file, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        generated_fields = {data_field.name for data_field in fields(self) if not data_field.init}
        data = TypeAdapter(type(self)).dump_python(self, by_alias=True, exclude=generated_fields)
        df = pd.DataFrame(data)

        times = self.get_times()
        if times is not None:
            df["Times"] = pd.to_datetime(times).strftime("%Y-%m-%dT%H:%M:%S.%f")

        return df

    def get_unique_labels(self) -> set[str]:
        """
        Get all unique labels across all detections in this detector.

        Returns
        -------
        set[str]
        """
        unique_labels = set()
        for label_set in self.labels:
            unique_labels.update(label_set)
        return unique_labels
