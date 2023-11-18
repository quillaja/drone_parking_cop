from enum import StrEnum


class Cols(StrEnum):
    TIME_MS = "time(millisecond)"  # need
    TIME_UTC = "datetime(utc)"
    LATITUDE = "latitude"  # need
    LONGITUDE = "longitude"  # need
    # "height_above_takeoff(feet)"
    # "height_above_ground_at_drone_location(feet)"
    # "ground_elevation_at_drone_location(feet)"
    ALTITUDE = "altitude_above_seaLevel(feet)"  # need
    # "height_sonar(feet)"
    # "speed(mph)"
    # "distance(feet)"
    # "mileage(feet)"
    # "satellites"
    # "gpslevel"
    # "voltage(v)"
    # "max_altitude(feet)"
    # "max_ascent(feet)"
    # "max_speed(mph)"
    # "max_distance(feet)"
    # "xSpeed(mph)"
    # "ySpeed(mph)"
    # "zSpeed(mph)"
    HEADING = "compass_heading(degrees)"  # need
    # "pitch(degrees)"
    # "roll(degrees)"
    # "isPhoto"
    IS_VIDEO = "isVideo"  # need
    # "rc_elevator"
    # "rc_aileron"
    # "rc_throttle"
    # "rc_rudder"
    # "rc_elevator(percent)"
    # "rc_aileron(percent)"
    # "rc_throttle(percent)"
    # "rc_rudder(percent)"
    # "gimbal_heading(degrees)"
    PITCH = "gimbal_pitch(degrees)"  # need
    # "gimbal_roll(degrees)"
    # "battery_percent"
    # "voltageCell1"
    # "voltageCell2"
    # "voltageCell3"
    # "voltageCell4"
    # "voltageCell5"
    # "voltageCell6"
    # "current(A)"
    # "battery_temperature(f)"
    # "altitude(feet)"  # equal to altitude_above_seaLevel(feet)?
    # "ascent(feet)"
    # "flycStateRaw"
    # "flycState"
    # "message"


ImportantColumns = [Cols.TIME_MS, Cols.TIME_UTC, Cols.LONGITUDE,
                    Cols.LATITUDE, Cols.ALTITUDE, Cols.HEADING, Cols.PITCH, Cols.IS_VIDEO]
