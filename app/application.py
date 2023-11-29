from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import easyocr
from cv2.typing import MatLike
from ultralytics import YOLO

import db
import djilog
import frame as fr
from app import draw
from sort.sort import Sort


class VideoMetrics(NamedTuple):
    width: int
    height: int
    fps: float
    frames: int
    length_sec: float


def video_size(video: cv2.VideoCapture) -> tuple[int, int]:
    w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return (int(w), int(h))


def video_fps(video: cv2.VideoCapture) -> float:
    return video.get(cv2.CAP_PROP_FPS)


def video_frame_count(video: cv2.VideoCapture) -> int:
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


def video_length(video: cv2.VideoCapture) -> float:
    """Calculate the video length in seconds."""
    return video_frame_count(video)/video_fps(video)


def frame_to_ms(frame: int, fps: float) -> float:
    return (frame/fps)*1000


ROOT = Path(__file__).resolve().parent.parent
YOLO_DETECT_DATA = ROOT/"assets/yolov8n.pt"
YOLO_SEGMENT_DATA = ROOT/"assets/yolov8n-seg.pt"
YOLO_PLATE_DATA = ROOT/"assets/license_plate_detector.pt"
LICENSE_PLATE_DATA = ROOT/"assets/plate_database.csv"
PARKING_SPACE_DATA = ROOT/"assets/plates.gpkg"
PCC_DEM = ROOT/"assets/pcc_sylvania_dtm_2014_crs6559.tif"


class Application:
    def __init__(self, video: str, log: str) -> None:
        self.video = cv2.VideoCapture(video)
        self.video_segments = djilog.segment(djilog.read_log(
            log,
            # input_crs="EPSG:4326",
            output_crs="EPSG:6559"))
        self.log: djilog.FlightLog = None

        self.parking = db.GeoPackageSpaceDB(PARKING_SPACE_DATA)
        with Path(LICENSE_PLATE_DATA).open() as file:
            self.vehicles = db.CSVVehicleDB(file.readlines(), 0.5)

        self.processor = fr.Processor(
            # detector=frame.YoloSortDetector(
            #     detector=YOLO(YOLO_DETECT_DATA),
            #     tracker=Sort(min_hits=6)),
            # detector=fr.YoloOnlyDetector(YOLO(YOLO_DETECT_DATA)),
            detector=fr.TwoLevelDetector(
                vehicle_model=YOLO(YOLO_SEGMENT_DATA),
                plate_model=YOLO(YOLO_PLATE_DATA)),
            reader=fr.EasyOCRReader(reader=easyocr.Reader(["en"])))

    def get_video_metrics(self) -> VideoMetrics:
        """Video information."""
        w, h = video_size(self.video)
        return VideoMetrics(
            width=w, height=h,
            fps=video_fps(self.video),
            frames=video_frame_count(self.video),
            length_sec=video_length(self.video))

    def get_segment_selection(self) -> list[tuple[int, datetime, datetime, float]]:
        """
        Get a list of information about the log video (isVideo) segments, in
        the form (index, start, end, duration in seconds).
        """
        return [(i, s, e, d) for i, (s, e, d,) in enumerate(
            djilog.human_readable(log) for log in self.video_segments)]

    def select_active_log(self, index: int):
        """
        Set the active log segment based on the index returned from
        get_segment_selection().
        """
        self.log = djilog.FlightLog(log_segment=self.video_segments[index],
                                    ground_dem_filename=PCC_DEM)

    def _draw_frame_marks(self, frame: MatLike, cropbox: fr.Box, vidbox: fr.Box):
        """draw representations of cropbox and vidbox."""
        draw.center_lines(frame, vidbox)
        draw.box(frame, cropbox, color=draw.GRAY50)

    def _draw_vehicle_box(self, frame: MatLike, center: list[float],
                          track_id: int, bb: fr.Box, space_info: db.SpaceInfo):
        """Draw box and annotation for a particular vehicle"""
        mc = self.processor.max_confidence[track_id]
        if bb.contains(*center):
            reg = self.vehicles.find_vehicle_by_plate(mc.text)
            t = f"{mc.track_id}:{mc.text} {mc.confidence:0.2f}"
            b = ""
            color = draw.BLUE
            if space_info:
                t += f" |{space_info.id}:{space_info.required_permit}"
            if reg:
                b = f"{reg.color} {reg.make}:{reg.permit_kind}"
            if reg and space_info:
                if reg.permit_kind == space_info.required_permit:
                    color = draw.GREEN
                else:
                    color = draw.RED
            draw.box(frame, bb, t, b, color)
        else:
            draw.box(frame, bb,
                     f"{mc.track_id}:{mc.text} {mc.confidence:0.2f}")

    def _draw_detections(self, frame: MatLike, frame_number: int,
                         cropbox: fr.Box, vidbox: fr.Box, space_info: db.SpaceInfo):
        """
        draw boxes/annotations for all vehicles found in the frame with number `frame_number`.
        """
        # sort detected objs by bb ymin so they are drawn "back to front"
        zsorted = sorted(self.processor.results_by_frame[frame_number],
                         key=lambda p: p.boxes[0].topleft[1])
        for p in zsorted:
            # move p.box from position in cropped frame to full frame
            bbs = [b.translate(*cropbox.topleft) for b in p.boxes]
            for bb in bbs:
                draw.box(frame, bb, color=draw.CYAN)
            outer_bb = bbs[0]
            self._draw_vehicle_box(frame, vidbox.center, p.track_id, outer_bb, space_info)

    def _draw_info(self, frame: MatLike, frame_number: int, frame_time_ms: float, num_objects: int,
                   drone_info: djilog.DroneInfo, space_info: db.SpaceInfo):
        """draw text info overlay in top left corner of frame"""
        draw.textline(frame, f"Frame: {frame_number}", 0)
        draw.textline(frame, f"Time: {frame_time_ms:0.2f}", 1)
        draw.textline(frame, f"Objects: {num_objects}", 2)
        if drone_info:
            dp = drone_info.drone_position
            tp = drone_info.target_position
            agl = drone_info.drone_alt_ft - drone_info.ground_alt_ft
            draw.textline(frame, "Drone", 4)
            draw.textline(frame, f" Location: {dp.x:0.2f}, {dp.y:0.2f}", 5)
            draw.textline(frame, f" Target:   {tp.x:0.2f}, {tp.y:0.2f}", 6)
            draw.textline(frame, f" DroneMSL: {drone_info.drone_alt_ft:0.2f}", 7)
            draw.textline(frame, f" GrndMSL:  {drone_info.ground_alt_ft:0.2f}", 8)
            draw.textline(frame, f" DroneAGL: {agl:0.2f}AGL", 9)
            draw.textline(frame, f" Compass: {drone_info.heading:0.2f}", 10)
            draw.textline(frame, f" Gimbal:  {drone_info.gimbal_pitch:0.2f}", 11)
        if space_info:
            draw.textline(
                frame, f"Parking space: {space_info.id} {space_info.required_permit}", 13)

    def run(self):
        """
        """
        vm = self.get_video_metrics()
        cropbox = fr.Box([vm.width*0.1, vm.height*0.15],
                         [vm.width*(1-0.1), vm.height*(1-0.05)])
        # box representing the frame
        vidbox = fr.Box([0, 0], [vm.width, vm.height])

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('output.avi', fourcc, 30.0, (vm.width, vm.height))

        frame_number = 0
        while self.video.isOpened():
            has_frame, frame = self.video.read()
            # if frame is read correctly has_frame is True
            if not has_frame:
                print(" No more frames. Exiting ...")
                break

            # get drone info from log and parking space info from parking db
            frame_time_ms = frame_to_ms(frame_number, vm.fps)
            drone_info = self.log.drone_info(frame_time_ms)
            if drone_info:
                space_info = self.parking.find_space_by_location(drone_info.target_position)

            frame_cropped = fr.crop(frame, cropbox)  # use cropped frame for actual detections
            self.processor.process(frame_cropped)
            self.processor.update_max(frame_number)
            num_objects = len(self.processor.max_confidence)

            self._draw_frame_marks(frame, cropbox, vidbox)
            self._draw_detections(frame, frame_number, cropbox, vidbox, space_info)
            self._draw_info(frame, frame_number, frame_time_ms, num_objects, drone_info, space_info)

            showframe = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)  # resize huge 4K video
            cv2.imshow('frame', showframe)

            out.write(frame)

            key = cv2.pollKey()
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)

            frame_number += 1

        out.release()
        self.video.release()
        cv2.destroyAllWindows()
