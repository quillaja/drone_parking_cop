from math import cos, radians, sin, tan
from typing import Iterable

import geopandas as gp
import pandas as pd
import rasterio as rio
import shapely
from shapely.affinity import translate

from .column_names import Cols


def sample_raster(raster: rio.DatasetReader, coords: Iterable[tuple[float, float]]) -> list[float]:
    """
    Get elevations from raster at the coordinates.
    """
    # this obtuse mess will give the list of coordinates to sample, which returns
    # a generator of iterables over the bands in the raster. Since this is only
    # for DEM rasters, we take the first (and only) band.
    samples = [float(x[0]) for x in raster.sample(xy=coords)]
    return samples


def ground_elevation(geom: gp.GeoSeries, ground_dem_path: str) -> pd.Series:
    """
    Create a new Series containing the elevation from the DEM at each
    point in the geom GeoSeries.

    example:
    ```
    df["ground"] = ground_elevation(df.geometry, "path/to/dem.tif")
    ```
    """
    ground: rio.DatasetReader = rio.open(ground_dem_path)
    geom = geom.to_crs(ground.crs)

    return pd.Series(sample_raster(
        raster=ground,
        coords=((pt.x, pt.y) for pt in geom.tolist())))


def ground_distance(height: float, gimbal: float) -> float:
    """
    Calculate the horizontal distance of the point at which the drone is looking,
    using the height above the surface and the gimbal's pitch angle (degrees from
    horizontal).
    """
    # tan undefined at 90 deg
    if gimbal == 0:
        return 0
    # the gimbal pitch is negative degrees from horizontal, since we want
    # degrees from vertical (to put d opposite the angle), we add 90 deg
    return height * tan(radians(90+gimbal))


def ground_offsets(height: float, gimbal: float, compass: float):
    """
    Calculate horizontal position offsets/deltas of the point at which the drone
    is looking, relative to the drone's current position. Returns (d, dx, dy)
    d is the distance from the drone to the look point and dx and dy are the offsets.

    Gimbal angle is degrees from horizontal and compass angle is degrees from
    true north.
    """
    d = ground_distance(height, gimbal)
    dx = d * cos(radians(90-compass))  # 0⁰N is 90⁰ in mathland
    dy = d * sin(radians(90-compass))
    return (d, dx, dy)


def calc_target(geom: shapely.Geometry, height: float, gimbal: float, compass: float) -> shapely.Geometry:
    """
    Calculate the position of the location on the ground where the drone's camera is
    pointing. `geom` is the drone's position, `height` is the drone's height above the ground,
    `gimbal` is the angle in degrees from horizontal, and `compass` is the angle in degrees from
    true north.
    """
    d, dx, dy = ground_offsets(height, gimbal, compass)
    return translate(geom, xoff=dx, yoff=dy)


def drone_targets(log: gp.GeoDataFrame, ground_col: str = "ground") -> gp.GeoSeries:
    """
    Get a GeoSeries of the points where the drone is looking. `log` must
    have the ground elevation in "ground_col".
    """
    return log.apply(lambda row:
                     calc_target(
                         geom=row.geometry,
                         height=row[Cols.ALTITUDE] - row[ground_col],
                         gimbal=row[Cols.PITCH],
                         compass=row[Cols.HEADING]),
                     axis="columns")
