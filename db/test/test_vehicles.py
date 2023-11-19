import db.vehicle_registry as vr


def test_VehicleInfo_from_dict():
    d = {
        "plate": "YOLO8",
        "permit_number": 1234,
        "permit_kind": "general",
        "make": "Ford",
        "model": "Focus",
        "color": "silver"}
    got = vr.VehicleInfo.from_dict(d)
    assert isinstance(got, vr.VehicleInfo)
    assert got.plate == d["plate"]
    assert got.permit_number == d["permit_number"]
    assert got.permit_kind == vr.Permits.general
    assert got.make == d["make"]
    assert got.model == d["model"]
    assert got.color == d["color"]


def test_VehicleInfo_to_dict():
    v = vr.VehicleInfo(
        plate="YOLO8",
        permit_number=1234,
        permit_kind=vr.Permits.general,
        make="Ford",
        model="Focus",
        color="silver")
    d = v.to_dict()
    assert d == {
        "plate": "YOLO8",
        "permit_number": 1234,
        "permit_kind": "general",
        "make": "Ford",
        "model": "Focus",
        "color": "silver"}


def test_JSONVehicleDB_from_json_list():
    list_data = [{
        "plate": "YOLO8",
        "permit_number": 1234,
        "permit_kind": "general",
        "make": "Ford",
        "model": "Focus",
        "color": "silver"}]
    db = vr.JSONVehicleDB.from_json(list_data)
    assert "YOLO8" in db
    assert db["YOLO8"] == vr.VehicleInfo(
        plate="YOLO8",
        permit_number=1234,
        permit_kind=vr.Permits.general,
        make="Ford",
        model="Focus",
        color="silver")


def test_JSONVehicleDB_from_json_dict():
    dict_data = {"YOLO8": {
        "plate": "YOLO8",
        "permit_number": 1234,
        "permit_kind": "general",
        "make": "Ford",
        "model": "Focus",
        "color": "silver"}}
    db = vr.JSONVehicleDB.from_json(dict_data)
    assert "YOLO8" in db
    assert db["YOLO8"] == vr.VehicleInfo(
        plate="YOLO8",
        permit_number=1234,
        permit_kind=vr.Permits.general,
        make="Ford",
        model="Focus",
        color="silver")


def test_CSVVehicleDB():
    csv = [
        "plate,permit_number,permit_kind,make,model,color",
        "YOLO8,1234,general,Ford,Focus,silver"
    ]
    db = vr.CSVVehicleDB(csv)
    assert "YOLO8" in db.data
    assert db.data["YOLO8"] == vr.VehicleInfo(
        plate="YOLO8",
        permit_number=1234,
        permit_kind=vr.Permits.general,
        make="Ford",
        model="Focus",
        color="silver")
