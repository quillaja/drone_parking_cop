from typing import NamedTuple

import cv2
from cv2.typing import MatLike

import frame as fr


class Color(NamedTuple):
    b: int
    g: int
    r: int


BLUE = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
RED = Color(0, 0, 255)
WHITE = Color(255, 255, 255)
BLACK = Color(0, 0, 0)
GRAY50 = Color(127, 127, 127)
CYAN = Color(255, 255, 0)
MAGENTA = Color(255, 0, 255)
YELLOW = Color(0, 255, 255)


def center_lines(frame: MatLike, box: fr.Box, color: Color = BLUE):
    """
    Draw vertical and horizontal lines through box's center.
    """
    cx, cy = int(box.center[0]), int(box.center[1])
    w2, h2 = int(box.width/2), int(box.height/2)
    # vertical line
    cv2.line(frame, pt1=(cx, cy-h2), pt2=(cx, cy+h2),
             color=color, thickness=1)
    # horizontal line
    cv2.line(frame, pt1=(cx-w2, cy), pt2=(cx+w2, cy),
             color=color, thickness=1)


def textline(frame: MatLike, text: str, line: int):
    """
    Draw text in lines at upper left corner.
    """
    TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX
    TEXT_COLOR = WHITE
    TEXT_HALO = BLACK
    # h, w = frame.shape
    tsize, _ = cv2.getTextSize(text=text, fontFace=TEXT_FONT,
                               fontScale=1.5, thickness=2)
    frame = cv2.putText(img=frame, text=text,
                        org=(5, (line+1)*(tsize[1]+8)),
                        fontFace=TEXT_FONT,
                        fontScale=1.5, color=TEXT_HALO, thickness=6)
    frame = cv2.putText(img=frame, text=text,
                        org=(5, (line+1)*(tsize[1]+8)),
                        fontFace=TEXT_FONT,
                        fontScale=1.5, color=TEXT_COLOR, thickness=2)


def box(img: MatLike, box: fr.Box,
        upper_text: str = "", lower_text: str = "",
        color: Color = WHITE):
    """draw box overlayed on image. this modifies the image."""
    BOX_COLOR = color
    BOX_THICKNESS = 2
    TEXT_COLOR = BLACK
    TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SIZE = 1.25
    TEXT_THICKNESS = 2

    pt_min, pt_max = box.as_int()
    # draw box
    img = cv2.rectangle(img=img, pt1=pt_min, pt2=pt_max,
                        color=BOX_COLOR, thickness=BOX_THICKNESS)
    # draw upper text
    if upper_text:
        tsize, _ = cv2.getTextSize(text=upper_text, fontFace=TEXT_FONT,
                                   fontScale=TEXT_SIZE, thickness=TEXT_THICKNESS)
        img = cv2.rectangle(img=img, pt1=pt_min,
                            pt2=(pt_min[0]+tsize[0], pt_min[1]-tsize[1]),
                            color=BOX_COLOR, thickness=4)
        img = cv2.rectangle(img=img, pt1=pt_min,
                            pt2=(pt_min[0]+tsize[0], pt_min[1]-tsize[1]),
                            color=BOX_COLOR, thickness=-1)
        img = cv2.putText(img=img, text=upper_text,
                          org=pt_min,
                          fontFace=TEXT_FONT,
                          fontScale=TEXT_SIZE, color=TEXT_COLOR, thickness=TEXT_THICKNESS)
    # # draw lower text
    if lower_text:
        tsize, _ = cv2.getTextSize(text=lower_text, fontFace=TEXT_FONT,
                                   fontScale=TEXT_SIZE, thickness=TEXT_THICKNESS)
        img = cv2.rectangle(img=img, pt1=(pt_min[0], pt_max[1]),
                            pt2=(pt_min[0]+tsize[0], pt_max[1]+tsize[1]),
                            color=BOX_COLOR, thickness=4)
        img = cv2.rectangle(img=img, pt1=(pt_min[0], pt_max[1]),
                            pt2=(pt_min[0]+tsize[0], pt_max[1]+tsize[1]),
                            color=BOX_COLOR, thickness=-1)
        img = cv2.putText(img=img, text=lower_text,
                          org=(pt_min[0], pt_max[1]+tsize[1]),
                          fontFace=TEXT_FONT,
                          fontScale=TEXT_SIZE, color=TEXT_COLOR, thickness=TEXT_THICKNESS)
