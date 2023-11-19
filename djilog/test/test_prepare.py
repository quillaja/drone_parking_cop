
from datetime import datetime
from pathlib import Path

import geopandas as gp
import pyproj
import pytest
from dateutil import tz

import djilog.prepare as prepare

TEST_LOG = str(Path(__file__).with_name("test_log.csv").absolute())


def test_read_log():
    df = prepare.read_log(TEST_LOG, input_crs=prepare.ITRF2008, output_crs="EPSG:4326")
    assert isinstance(df, gp.GeoDataFrame)
    assert df.crs == pyproj.CRS.from_string("EPSG:4326")
    assert len(df) == 2553


def test_list_video_segments():
    expected = [(83300, 88500),
                (192900, 283300),
                (315500, 323500),
                (348700, 388900),
                (453100, 493300)]
    df = prepare.read_log(TEST_LOG)
    got = prepare.list_video_segments(df)
    assert got == expected


def test_segment():
    df = prepare.read_log(TEST_LOG)
    segments = prepare.segment(df)
    assert len(segments) == 5
    assert all([isinstance(seg, gp.GeoDataFrame) for seg in segments])
    assert all([seg.loc[0, prepare.Cols.IS_VIDEO] == 1 for seg in segments])


def test_human_readable():
    df = prepare.read_log(TEST_LOG)
    got = prepare.human_readable(df)
    s, e, d = got
    assert isinstance(s, datetime)
    assert isinstance(e, datetime)
    assert isinstance(d, float)
    assert s.tzinfo == tz.tzlocal()
    assert e.tzinfo == tz.tzlocal()
