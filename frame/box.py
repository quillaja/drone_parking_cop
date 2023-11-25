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

    def draw(self, img: cv2.Mat, box_color: tuple[int, int, int] = (0, 255, 0)):
        """draw result overlayed on image. this modifies the image."""
        BOX_COLOR = box_color
        TEXT_COLOR = (0, 0, 0)
        TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX

        # draw box
        img = cv2.rectangle(img=img, pt1=self.topleft, pt2=self.bottomright,
                            color=BOX_COLOR, thickness=1)
        # # draw ocr text
        # tsize, _ = cv2.getTextSize(text=self.text, fontFace=TEXT_FONT,
        #                            fontScale=1, thickness=1)
        # img = cv2.rectangle(img=img, pt1=self.box[0],
        #                     pt2=(self.box[0][0]+tsize[0], self.box[0][1]-tsize[1]),
        #                     color=BOX_COLOR, thickness=-1)
        # img = cv2.putText(img=img, text=self.text,
        #                   org=self.box[0],
        #                   fontFace=TEXT_FONT,
        #                   fontScale=1, color=TEXT_COLOR, thickness=1)
        # # draw confidence level
        # confidence = f"{self.confidence:0.2f}"
        # tsize, _ = cv2.getTextSize(text=confidence, fontFace=TEXT_FONT,
        #                            fontScale=1, thickness=1)
        # img = cv2.rectangle(img=img, pt1=self.box[3],
        #                     pt2=(self.box[3][0]+tsize[0], self.box[3][1]+tsize[1]),
        #                     color=BOX_COLOR, thickness=-1)
        # img = cv2.putText(img=img, text=confidence,
        #                   org=(self.box[3][0], self.box[3][1]+tsize[1]),
        #                   fontFace=TEXT_FONT,
        #                   fontScale=1, color=TEXT_COLOR, thickness=1)
