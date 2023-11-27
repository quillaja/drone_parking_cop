from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class Box:
    topleft: list[float]
    """Top left pixel of the box."""
    bottomright: list[float]
    """Bottom left pixel of the box."""

    # @staticmethod
    # def from_two_points(top_left: list[float], bottom_right: list[float]) -> 'Detection':
    #     """Make a Detection using two corners of a box."""
    #     return Detection(top_left, bottom_right)

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

    def translate(self, dx: int, dy: int) -> 'Box':
        """translate. Returns new object."""
        tl = [self.topleft[0]+dx, self.topleft[1]+dy]
        br = [self.bottomright[0]+dx, self.bottomright[1]+dy]
        return Box(tl, br)

    def scale(self, *s: float):
        """scale by s. Returns new object."""
        if len(s) == 1:
            return Box([self.topleft[0]*s[0], self.topleft[1]*s[0]],
                       [self.bottomright[0]*s[0], self.bottomright[1]*s[0]])
        elif len(s) >= 2:
            return Box([self.topleft[0]*s[0], self.topleft[1]*s[1]],
                       [self.bottomright[0]*s[0], self.bottomright[1]*s[1]])

    def as_int(self) -> tuple[tuple[int], tuple[int]]:
        xmin, xmax, ymin, ymax = self.sides
        return ((int(xmin), int(ymin)), (int(xmax), int(ymax)))

    @property
    def is_empty(self) -> bool:
        w = int(self.width)
        h = int(self.height)
        return w <= 0 or h <= 0


def draw(img: cv2.Mat, box: Box,
         upper_text: str = "", lower_text: str = "",
         box_color: tuple[int, int, int] = (0, 255, 0)):
    """draw box overlayed on image. this modifies the image."""
    BOX_COLOR = box_color
    TEXT_COLOR = (0, 0, 0)
    TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX

    pt_min, pt_max = box.as_int()
    # draw box
    img = cv2.rectangle(img=img, pt1=pt_min, pt2=pt_max,
                        color=BOX_COLOR, thickness=2)

    # draw upper text
    if upper_text:
        tsize, _ = cv2.getTextSize(text=upper_text, fontFace=TEXT_FONT,
                                   fontScale=1, thickness=1)
        img = cv2.rectangle(img=img, pt1=pt_min,
                            pt2=(pt_min[0]+tsize[0], pt_min[1]-tsize[1]),
                            color=BOX_COLOR, thickness=-1)
        img = cv2.putText(img=img, text=upper_text,
                          org=pt_min,
                          fontFace=TEXT_FONT,
                          fontScale=1, color=TEXT_COLOR, thickness=1)

    # # draw lower text
    if lower_text:
        tsize, _ = cv2.getTextSize(text=lower_text, fontFace=TEXT_FONT,
                                   fontScale=1, thickness=1)
        img = cv2.rectangle(img=img, pt1=(pt_min[0], pt_max[1]),
                            pt2=(pt_min[0]+tsize[0], pt_max[1]+tsize[1]),
                            color=BOX_COLOR, thickness=-1)
        img = cv2.putText(img=img, text=lower_text,
                          org=(pt_min[0], pt_max[1]+tsize[1]),
                          fontFace=TEXT_FONT,
                          fontScale=1, color=TEXT_COLOR, thickness=1)
