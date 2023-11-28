import math
from typing import NamedTuple

import geopandas as gp
import shapely

from djilog.column_names import Cols

from .augment import drone_targets, ground_elevation


class DroneInfo(NamedTuple):
    drone_alt_ft: float
    ground_alt_ft: float
    heading: float
    gimbal_pitch: float
    drone_position: shapely.Point
    target_position: shapely.Point


class FlightLog:
    GROUND_COL = "ground"
    TARGET_COL = "target"
    DRONE_LOC_COL = "geometry"

    def __init__(self, log_segment: gp.GeoDataFrame, ground_dem_filename: str,
                 ground_fudge_factor: float = 2.5) -> None:
        """
        Prepare a pre-chosen subset of a complete DJI drone log to be queried
        for drone information at each video frame.
        """
        self.df = log_segment
        # add ground elevation under drone and camera target position
        self.df[FlightLog.GROUND_COL] = ground_elevation(
            self.df.geometry, ground_dem_filename)+ground_fudge_factor
        self.df[FlightLog.TARGET_COL] = drone_targets(self.df, FlightLog.GROUND_COL)
        # shift all times to a 0-indexed start
        self._start_ms = self.df[Cols.TIME_MS].iloc[0]
        self.df[Cols.TIME_MS] = self.df[Cols.TIME_MS] - self._start_ms
        self.df = self.df.set_index(Cols.TIME_MS)

    def drone_info(self, time_ms: float) -> DroneInfo | None:
        """
        Given a time in milliseconds, get information about the drone. If 
        the time falls between log entries, the target point is linearly 
        interpolated. Thus, this is a valid operation only for 
        projected coordinate systems.
        """
        # dji logs are recorded at 200ms intervals. They don't always start at
        # 0ms, but since I shifted them in the constructor, they do.
        INTERVAL = 200
        lower = int(time_ms/INTERVAL) * INTERVAL
        upper = int(time_ms/INTERVAL + 1) * INTERVAL
        t = (time_ms - lower)/(upper-lower)

        try:
            # it seems that some log entries weren't written, so occasionally
            # a particular query fails
            infos = [
                DroneInfo(
                    drone_alt_ft=self.df.loc[ms, Cols.ALTITUDE],
                    ground_alt_ft=self.df.loc[ms, FlightLog.GROUND_COL],
                    heading=self.df.loc[ms, Cols.HEADING],
                    gimbal_pitch=self.df.loc[ms, Cols.PITCH],
                    drone_position=self.df.loc[ms, FlightLog.DRONE_LOC_COL],
                    target_position=self.df.loc[ms, FlightLog.TARGET_COL])
                for ms in [lower, upper]]
        except KeyError:
            return None

        # wow, this is getting a bit obtuse...
        # this spreads the 2 DroneInfos, then zips their members,
        # then lerps the members. then the new items from the generator
        # are spread and passed to DroneInfo constructor
        return DroneInfo(*(lerp(l, u, t) for l, u in zip(*infos)))


def lerp(a: float | shapely.Point, b: float | shapely.Point, t: float) -> float:
    if isinstance(a, float) and isinstance(b, float):
        return a*(1-t) + b*t
    elif isinstance(a, shapely.Point) and isinstance(b, shapely.Point):
        x = a.x*(1-t) + b.x*t
        y = a.y*(1-t) + b.y*t
        return shapely.Point(x, y)
    else:
        raise ValueError("both args must be float or shapely.Point")
