import db.vehicle_registry as vr

test_vehicle_dict = {
    "plate": "YOLO8",
    "state": "OR",
    "permit_number": 1234,
    "permit_kind": "general",
    "make": "Ford",
    "color": "silver",
    "is_ev": "false",
    "note": ""}

test_vehicle_obj = vr.VehicleInfo(
    plate="YOLO8",
    state="OR",
    permit_number=1234,
    permit_kind=vr.Permits.general,
    make="Ford",
    color="silver",
    is_ev="false",
    note="")


def test_VehicleInfo_from_dict():
    d = test_vehicle_dict
    got = vr.VehicleInfo.from_dict(d)
    assert isinstance(got, vr.VehicleInfo)
    assert got.plate == d["plate"]
    assert got.state == d["state"]
    assert got.permit_number == d["permit_number"]
    assert got.permit_kind == vr.Permits.general
    assert got.make == d["make"]
    assert got.color == d["color"]
    assert got.is_ev == d["is_ev"]


def test_VehicleInfo_to_dict():
    v = test_vehicle_obj
    d = v.to_dict()
    assert d == test_vehicle_dict


def test_JSONVehicleDB_from_json_list():
    list_data = [test_vehicle_dict]
    db = vr.JSONVehicleDB.from_json(list_data)
    assert "YOLO8" in db
    assert db["YOLO8"] == test_vehicle_obj


def test_JSONVehicleDB_from_json_dict():
    dict_data = {"YOLO8": test_vehicle_dict}
    db = vr.JSONVehicleDB.from_json(dict_data)
    assert "YOLO8" in db
    assert db["YOLO8"] == test_vehicle_obj


def test_CSVVehicleDB():
    csv = [
        "plate,state,permit_number,permit_kind,make,color,is_ev,note",
        "YOLO8,OR,1234,general,Ford,silver,false,"
    ]
    db = vr.CSVVehicleDB(csv)
    assert "YOLO8" in db.data
    assert db.data["YOLO8"] == test_vehicle_obj
