from datetime import datetime

import geopandas as gp
import pandas as pd
from dateutil import tz

from .column_names import Cols

# 7911 is "3d" version of 8999; 8999 also works (and is the itrf08 version used in arcgis)
ITRF2008 = "EPSG:8999"
"""
International Terrestrial Reference Frame 2008.
It seems this is the default CRS used in DJI drone logs as of Nov 2023.
EPSG:7911 is the 3D version, with ellipsoidal height in meters. GeoPandas apparently
doesn't do 3D operations, so we'll stick with the 2D version.
https://epsg.io/8999
https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame#Versions
"""


def read_log(log_filename: str, output_crs: str, input_crs: str = ITRF2008) -> gp.GeoDataFrame:
    """
    Open the DJI log in CSV format found at `log_filename` and convert it to a
    GeoPandas GeoDataframe. `input_crs` sets the coordinate reference system of the log
    file (usually EPSG:8999) and `output_crs` is a reference system to project
    the log into, specified as an EPSG string such as `EPSG:4329`.
    """
    # geopandas is making everything an object type instead of int, float, etc.
    # not sure why, but opening the file in pandas first, then converting to geodataframe
    # fixes it, and gives and chance to set geospatial stuff.
    raw_data = pd.read_csv(log_filename)
    # raw_data = raw_data.drop(labels=["height_above_ground_at_drone_location(feet)",
    #                                  "ground_elevation_at_drone_location(feet)"], axis="columns")
    logdf = gp.GeoDataFrame(data=raw_data,
                            geometry=gp.points_from_xy(
                                x=raw_data[Cols.LONGITUDE],
                                y=raw_data[Cols.LATITUDE]),
                            crs=input_crs).to_crs(output_crs)

    logdf[Cols.TIME_UTC] = pd.to_datetime(logdf[Cols.TIME_UTC], utc=True)

    return logdf


def list_video_segments(df: gp.GeoDataFrame) -> tuple[int, int]:
    """
    Get list of start and end times (milliseconds, from the "time(milliseconds)" columns) 
    for segments when a video was being recorded.
    """
    times = []
    video = False
    segment = (0, 0)
    prev_ms = 0
    for row in df[[Cols.TIME_MS, Cols.IS_VIDEO]].itertuples(index=False):
        ms, v = row[0], row[1]
        if v == 1 and not video:
            # video started
            segment = (ms, 0)
            video = True
        elif v == 0 and video:
            # video ended
            segment = (segment[0], prev_ms)
            video = False
            times.append(segment)
        prev_ms = ms
    return times


def segment(df: gp.GeoDataFrame) -> list[gp.GeoDataFrame]:
    """
    Split a log into individual GeoDataframes for each video segment.
    """
    frames = []
    segments = list_video_segments(df)
    df = df.set_index(Cols.TIME_MS)
    for s, e in segments:
        seg = df.loc[s:e].copy().reset_index()
        frames.append(seg)
    return frames


def human_readable(df: gp.GeoDataFrame) -> tuple[datetime, datetime, float]:
    """
    Return useful info about the dataframe as a tuple of the datetimes for
    the first and last row of the log, and of the log duration in fractional
    seconds.
    """
    first = df.iloc[0]
    last = df.iloc[-1]
    duration = (last[Cols.TIME_MS]-first[Cols.TIME_MS])/1000
    local = tz.tzlocal()
    return (first[Cols.TIME_UTC].dt.astimezone(local),
            last[Cols.TIME_UTC].dt.astimezone(local),
            duration)
