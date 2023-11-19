from dataclasses import dataclass
from typing import Protocol

import geopandas as gp
import shapely

from .permit import Permits


@dataclass
class SpaceInfo:
    id: int
    lot: int
    required_permit: Permits
    # geom: shapely.Polygon
    # query_point: shapely.Point


class ParkingSpaceDB(Protocol):
    def find_space_by_location(self, loc: shapely.Point) -> SpaceInfo | None:
        ...


class GeoPackageSpaceDB:
    ID_COL = "space_id"
    LOT_COL = "lot"
    TYPE_COL = "type"

    def __init__(self, gpkg_filename: str, table: str = "spaces") -> None:
        self.spaces = gp.read_file(gpkg_filename, layer=table)

    def find_space_by_location(self, loc: shapely.Point) -> SpaceInfo | None:
        # the loc needs to be in a geodataframe to work with gp.overlay()
        # also assume the given point is in the same CRS as spaces
        pt = gp.GeoDataFrame(geometry=[loc], crs=self.spaces.crs)
        #  order of the args matters
        result = gp.overlay(pt, self.spaces, how="intersection")
        if len(result) == 0:
            return None

        return SpaceInfo(
            id=0,  # result[GeoPackageSpaceDB.ID_COL],
            lot=result[GeoPackageSpaceDB.TYPE_COL],
            kind=result[GeoPackageSpaceDB.TYPE_COL],
        )
