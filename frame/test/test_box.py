import pytest

from frame.box import Box

square = Box([0, 0], [1, 1])
hrect = Box([0, 0], [2, 1])
vrect = Box([0, 0], [1, 2])


def test_from_point_size():
    assert square == Box.from_point_size([0, 0], 1, 1)
    assert hrect == Box.from_point_size([0, 0], 2, 1)
    assert vrect == Box.from_point_size([0, 0], 1, 2)


def test_topright():
    assert square.topright == [1, 0]
    assert hrect.topright == [2, 0]
    assert vrect.topright == [1, 0]


def test_bottomleft():
    assert square.bottomleft == [0, 1]
    assert hrect.bottomleft == [0, 1]
    assert vrect.bottomleft == [0, 2]


def test_width():
    assert square.width == 1
    assert hrect.width == 2
    assert vrect.width == 1


def test_height():
    assert square.height == 1
    assert hrect.height == 1
    assert vrect.height == 2


def test_aspect():
    assert square.aspect == 1
    assert hrect.aspect == 2/1
    assert vrect.aspect == 1/2


def test_sides():
    assert square.sides == [0, 1, 0, 1]
    assert hrect.sides == [0, 2, 0, 1]
    assert vrect.sides == [0, 1, 0, 2]


def test_center():
    assert square.center == [0.5, 0.5]
    assert hrect.center == [1, 0.5]
    assert vrect.center == [0.5, 1]


def test_translate():
    assert square.translate(1, 1) == Box([1, 1], [2, 2])
    assert hrect.translate(1, 1) == Box([1, 1], [3, 2])
    assert vrect.translate(1, 1) == Box([1, 1], [2, 3])


def test_scale():
    assert square.scale(2) == Box([0, 0], [2, 2])
    assert square.scale(2, 2) == Box([0, 0], [2, 2])
    assert square.scale(2, 2, 2) == Box([0, 0], [2, 2])
    assert square.scale(2, 1) == hrect
    assert square.scale(1, 2) == vrect
