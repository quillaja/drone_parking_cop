import csv
from dataclasses import asdict, dataclass
from typing import Iterable, Protocol

from .permit import Permits

VehicleJSON = dict[str, str | int]

# this got way too complex for such a simple thing


@dataclass
class VehicleInfo:
    plate: str
    state: str
    permit_number: int
    permit_kind: Permits
    make: str
    color: str
    is_ev: str  # nerfed thanks to python's stupid bools
    note: str

    def to_dict(self) -> VehicleJSON:
        return asdict(self)

    @staticmethod
    def from_dict(d: VehicleJSON) -> 'VehicleInfo':
        return VehicleInfo(
            plate=str(d["plate"]),
            state=str(d["state"]),
            permit_number=int(d["permit_number"]),
            permit_kind=Permits(d["permit_kind"]),
            make=str(d["make"]),
            color=str(d["color"]),
            is_ev=str(d["is_ev"]),
            note=str(d["note"]))


VehicleDict = dict[str, VehicleInfo]


class VehicleDB(Protocol):
    def find_vehicle_by_plate(self, license_plate: str) -> VehicleInfo | None:
        ...


class JSONVehicleDB:

    @staticmethod
    def from_json(objects: list[VehicleJSON] | dict[str, VehicleJSON]) -> VehicleDict:
        """
        Convert a list of dicts or a dict of dicts to a dict of VehicleInfo.
        """
        if isinstance(objects, list):
            return {v.plate: v for v in map(VehicleInfo.from_dict, objects)}
        elif isinstance(objects, dict):
            return {k: VehicleInfo.from_dict(v) for k, v in objects.items()}
        else:
            raise ValueError("objects is not a list or dict")

    def __init__(self, data: VehicleDict) -> None:
        self.data: VehicleDict = data

    def find_vehicle_by_plate(self, license_plate: str) -> VehicleInfo | None:
        return self.data.get(license_plate, None)

# csv is way easier


class CSVVehicleDB:
    def __init__(self, csv_content: Iterable[str]) -> None:
        """
        Create a VehicleDB from CSV contents as a list of strings.

        example:
        ```
        with open("vehicles.csv") as file:
            db = CSVVehicleDB(file.readlines())
        ```
        """
        self.data: VehicleDict = {}
        r = csv.DictReader(csv_content)
        for row in r:
            v = VehicleInfo.from_dict(row)
            self.data[v.plate] = v

    def find_vehicle_by_plate(self, license_plate: str) -> VehicleInfo | None:
        return self.data.get(license_plate, None)
