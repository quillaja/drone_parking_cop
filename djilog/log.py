from enum import StrEnum

import geopandas as gp
import shapely

from djilog.column_names import Cols

from .augment import drone_targets, ground_elevation


class DronePosition(StrEnum):
    position = "geometry"
    target = "target"


class FlightLog:
    GROUND_COL = "ground"
    TARGET_COL = "target"

    def __init__(self, log_segment: gp.GeoDataFrame, ground_dem_filename: str) -> None:
        """
        Prepare a pre-chosen subset of a complete DJI drone log to be queried
        for drone information at each video frame.
        """
        self.df = log_segment
        # add ground elevation under drone and camera target position
        self.df[FlightLog.GROUND_COL] = ground_elevation(self.df.geometry, ground_dem_filename)
        self.df[FlightLog.TARGET_COL] = drone_targets(self.df, FlightLog.GROUND_COL)
        # shift all times to a 0-indexed start
        self._start_ms = self.df[Cols.TIME_MS].iloc[0]
        self.df[Cols.TIME_MS] = self.df[Cols.TIME_MS] - self._start_ms
        self.df = self.df.set_index(Cols.TIME_MS)

    def drone_info(self, time_ms: float, info: DronePosition) -> shapely.Point:
        """
        Given a time in milliseconds, get position information about the drone. If 
        the time falls between log entries, the target point is linearly 
        interpolated. Thus, this is a valid operation only for 
        projected coordinate systems.
        """
        # dji logs are recorded at 200ms intervals. They don't always start at
        # 0ms, but since I shifted them in the constructor, they do.
        lower = int(time_ms/200) * 200
        upper = int(time_ms/200 + 1) * 200
        t = (time_ms - lower)/(upper-lower)
        pt1: shapely.Point = self.df.loc[lower, info]
        pt2: shapely.Point = self.df.loc[upper, info]
        x = pt1.x*(1-t) + pt2.x*t
        y = pt1.y*(1-t) + pt2.y*t
        return shapely.Point(x, y)
