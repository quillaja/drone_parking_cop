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
