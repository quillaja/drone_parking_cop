from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    topleft: list[float]
    """Top left pixel of the box."""
    bottomright: list[float]
    """Bottom left pixel of the box."""

    @staticmethod
    def from_xyxy(s: list[float]) -> 'Box':
        """Make a Box from a list of numbers as [xmin, ymin, xmax, ymax]"""
        return Box([s[0], s[1]], [s[2], s[3]])

    @staticmethod
    def from_point_size(topleft: list[float], dx: float, dy: float) -> 'Box':
        """Make a Detection using one corner of a box and the box's width and height."""
        return Box(topleft, [topleft[0]+dx, topleft[1]+dy])

    @property
    def sides(self) -> list[float]:
        """Get sides of box as [xmin, xmax, ymin, ymax]. """
        return [self.topleft[0], self.bottomright[0], self.topleft[1], self.bottomright[1]]

    @property
    def topright(self) -> list[float]:
        """top right corner"""
        return [self.bottomright[0], self.topleft[1]]

    @property
    def bottomleft(self) -> list[float]:
        """bottom left corner"""
        return [self.topleft[0], self.bottomright[1]]

    @property
    def width(self) -> float:
        """width"""
        return self.bottomright[0]-self.topleft[0]

    @property
    def height(self) -> float:
        """height"""
        return self.bottomright[1]-self.topleft[1]

    @property
    def aspect(self) -> float:
        """width/height"""
        return self.width/self.height

    @property
    def center(self) -> list[float]:
        """center coord of box"""
        xmin, xmax, ymin, ymax = self.sides
        return [(xmin+xmax)/2, (ymin+ymax)/2]

    def translate(self, dx: float, dy: float) -> 'Box':
        """translate. Returns new object."""
        tl = [self.topleft[0]+dx, self.topleft[1]+dy]
        br = [self.bottomright[0]+dx, self.bottomright[1]+dy]
        return Box(tl, br)

    def scale(self, *s: float) -> 'Box':
        """scale by s. Returns new object."""
        if len(s) == 1:
            return Box([self.topleft[0]*s[0], self.topleft[1]*s[0]],
                       [self.bottomright[0]*s[0], self.bottomright[1]*s[0]])
        elif len(s) >= 2:
            return Box([self.topleft[0]*s[0], self.topleft[1]*s[1]],
                       [self.bottomright[0]*s[0], self.bottomright[1]*s[1]])
        else:
            raise ValueError("s must be of length 1 or 2")

    def scale_center(self, *s: float) -> 'Box':
        """scale by s from box's center"""
        cx, cy = self.center
        return self.translate(-cx, -cy).scale(*s).translate(cx, cy)

    def as_int(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """get top left and bottom right corners as integers"""
        xmin, xmax, ymin, ymax = self.sides
        return ((int(xmin), int(ymin)), (int(xmax), int(ymax)))

    @property
    def is_empty(self) -> bool:
        """True if wither width or height is <= 0"""
        w = int(self.width)
        h = int(self.height)
        return w <= 0 or h <= 0

    def contains(self, x: float, y: float,
                 x_margin: float = 0, y_margin: float = 0) -> bool:
        """
        True if box contains (x,y). The optional margins adjust the box
        sides relative to the box center.
        """
        xmin, xmax, ymin, ymax = self.sides
        return ((xmin-x_margin <= x and x <= xmax+x_margin)
                and (ymin-y_margin <= y and y <= ymax+y_margin))
