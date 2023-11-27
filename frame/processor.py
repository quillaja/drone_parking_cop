import re
from typing import Callable, NamedTuple, Protocol

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from frame.box import Box
from sort.sort import Sort


class ObjectDetection(NamedTuple):
    track_id: int
    box: Box


class TextRecognition(NamedTuple):
    track_id: int
    confidence: float
    text: str


class Detector(Protocol):
    """A Detector finds objects within an image or video frame."""

    def find(self, frame: cv2.Mat) -> list[ObjectDetection]:
        """Find objects in frame."""
        ...


class YoloSortDetector:
    """Use YOLO for detection and https://github.com/abewley/sort for tracking."""

    def __init__(self, detector: YOLO, tracker: Sort) -> None:
        self.detector = detector
        self.tracker = tracker

    def find(self, frame: cv2.Mat) -> list[ObjectDetection]:
        # YOLO model returns a list because it can take a list as the source
        # param. Thus [0], to get the first result of a len-1 list.
        # https://docs.ultralytics.com/modes/predict/#working-with-results
        objects = self.detector(frame, verbose=False)[0]
        # convert yolo output to tracker format by slicing off the last
        # item (the class id) of each detected object
        objects = objects.boxes.data.cpu().numpy()[:, :-1]
        # the Sort tracker takes a numpy array with (n,5) shape where the len-5
        # dimension is [xmin,ymin,xmax,ymax,confidence_score]. returns a similar
        # ndarray with confidence_score replaced with tracking id. tracker doesn't
        # keep items in the same order as input (appears to be new-to-old order),
        # and the boxes are often different
        if len(objects) == 0:
            objects = np.empty((0, 5))
        track_ids = self.tracker.update(objects)
        # convert tracking results
        detected_plates: list[ObjectDetection] = []
        for obj in track_ids:
            bb = Box([obj[0], obj[1]], [obj[2], obj[3]])
            id = int(obj[4])
            detected_plates.append(ObjectDetection(track_id=id, box=bb))

        return detected_plates


class YoloOnlyDetector:
    """Use YOLO for both tracking and detection."""

    def __init__(self, detector: YOLO) -> None:
        self.detector = detector

    def find(self, frame: cv2.Mat) -> list[ObjectDetection]:
        # YOLO model returns a list because it can take a list as the source
        # param. Thus [0], to get the first result of a len-1 list.
        # https://docs.ultralytics.com/modes/predict/#working-with-results
        # for tracking, see
        # https://docs.ultralytics.com/modes/track/#persisting-tracks-loop
        objects = self.detector.track(frame, persist=True, verbose=False)[0]
        # convert yolo output to something i can use. this will be a list of:
        # [xmin, ymin, xmax, ymax, track_id, confidence, class_id]
        objects = objects.boxes.data.tolist()
        # convert results
        detected_plates: list[ObjectDetection] = []
        for obj in objects:
            bb = Box([obj[0], obj[1]], [obj[2], obj[3]])
            id = int(obj[4])
            detected_plates.append(ObjectDetection(track_id=id, box=bb))

        return detected_plates


def crop(frame: cv2.Mat, box: Box) -> cv2.Mat:
    """Crop frame to the box size."""
    xmin, xmax, ymin, ymax = box.sides
    return frame[int(ymin):int(ymax), int(xmin):int(xmax)]


class Reader(Protocol):
    """
    A Reader extracts text strings from an image or video frame
    based on the detected objects.
    """

    def read(self, frame: cv2.Mat, detections: list[ObjectDetection]) -> list[TextRecognition]:
        """Read text from frame found within the list of object detections."""
        ...


class EasyOCRReader:
    ALPHABET = "- 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, reader: easyocr.Reader) -> None:
        self.ocr = reader
        self.filter = filter if filter is not None else lambda x: True

    def read(self, frame: cv2.Mat, detections: list[ObjectDetection]) -> list[TextRecognition]:
        plate_texts: list[TextRecognition] = []
        for d in detections:
            # get a crop of the frame containing just the plate
            plate_crop = crop(frame, d.box)
            # do OCR
            # i'm not sure how much width_ths and batch_size improve
            # results or performance
            result = self.ocr.readtext(image=plate_crop,
                                       allowlist=EasyOCRReader.ALPHABET,
                                       width_ths=1.5,
                                       # workers=2, # doesn't like workers
                                       batch_size=16)
            # convert results
            for box, text, conf in result:
                # remove spaces and hypens which are allowed in ALPHABET
                text: str = text.replace(" ", "").replace("-", "")
                plate_texts.append(TextRecognition(
                    track_id=d.track_id,
                    confidence=conf,
                    text=text))

        return plate_texts


class Plate(NamedTuple):
    """
    A license plate found and read from a video frame.
    """
    track_id: int
    frame: int
    text: str
    confidence: float
    box: Box


PlateValidator = Callable[[Plate, Plate], bool]
"""
A function that determines whether or not too keep a new Plate vs and old one.
PlateValidator(old, new) -> bool
"""

PLATE_NUMBER = re.compile(r"[A-Z0-9]{4,7}")
"""regexp to id a variety of plates"""


def default_validator(old: Plate, new: Plate) -> bool:
    """Keep new if text is 4-7 alphanumeric and new confidence > old confidence."""
    if not PLATE_NUMBER.match(new.text):
        return False
    return new.confidence > old.confidence


class Processor:
    """
    A Processor extracts and manages data from sequential video frames.
    """

    def __init__(self, detector: Detector, reader: Reader, validator: PlateValidator = default_validator) -> None:
        self.detector = detector
        self.reader = reader
        self.results_by_frame: list[list[Plate]] = []
        self.max_confidence: dict[int, Plate] = {}
        self.validator: PlateValidator = validator

    def process(self, frame: cv2.Mat):
        """called once per frame"""
        frame_number = len(self.results_by_frame)

        detections = self.detector.find(frame)
        plate_texts = self.reader.read(frame, detections)

        plates = [
            Plate(
                track_id=p.track_id,
                frame=frame_number,
                text=p.text,
                confidence=p.confidence,
                box=d.box)
            for d, p in zip(detections, plate_texts)]

        self.results_by_frame.append(plates)
        # self.update_max(frame_number)

    def update_max(self, frame_number: int):
        """
        Updates the max_confidence dict using plates in `frame_number`.
        """
        for p in self.results_by_frame[frame_number]:
            id = p.track_id
            prev_plate = self.max_confidence.setdefault(id, p)
            if self.validator(prev_plate, p):
                self.max_confidence[id] = p
