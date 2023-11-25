from dataclasses import dataclass
from typing import Protocol

import geopandas as gp
import shapely

from .permit import Permits


@dataclass
class SpaceInfo:
    """Parking space information."""
    id: int
    lot: int
    required_permit: Permits
    # geom: shapely.Polygon
    # query_point: shapely.Point


class ParkingSpaceDB(Protocol):
    """
    A data storage for parking spaces that allows a space's information
    to be retrieved by a geographic location.
    """

    def find_space_by_location(self, loc: shapely.Point) -> SpaceInfo | None:
        """
        Get space info by geographic location. Returns None when
        no space is at that location.
        """
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
            id=result[GeoPackageSpaceDB.ID_COL],
            lot=result[GeoPackageSpaceDB.LOT_COL],
            required_permit=result[GeoPackageSpaceDB.TYPE_COL])
