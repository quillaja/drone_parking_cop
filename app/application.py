from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import easyocr
from ultralytics import YOLO

import db
import djilog
import frame as fr
from sort.sort import Sort


class VideoMetrics(NamedTuple):
    width: int
    height: int
    fps: float
    frames: int
    length_sec: float


def video_size(video: cv2.VideoCapture) -> (int, int):
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


def draw_center_lines(frame: cv2.Mat, box: fr.Box):
    frame_h_center = int(box.center[0])
    frame_v_center = int(box.center[1])
    w, h = box.width, box.height
    cv2.line(frame,
             pt1=(frame_h_center, 0),
             pt2=(frame_h_center, int(box.bottomright[1])),
             color=(255, 0, 0), thickness=1)
    cv2.line(frame,
             pt1=(0, frame_v_center),
             pt2=(int(box.bottomright[0]), frame_v_center),
             color=(255, 0, 0), thickness=1)


def draw_text(frame: cv2.Mat, text: str, line: int):
    TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX
    TEXT_COLOR = (255, 255, 255)
    # h, w = frame.shape
    tsize, _ = cv2.getTextSize(text=text, fontFace=TEXT_FONT,
                               fontScale=1, thickness=2)
    frame = cv2.putText(img=frame, text=text,
                        org=(5, (line+1)*(tsize[1]+6)),
                        fontFace=TEXT_FONT,
                        fontScale=1, color=TEXT_COLOR, thickness=2)


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

    def run(self):
        """
        """
        vm = self.get_video_metrics()
        cropbox = fr.Box([vm.width*0.1, vm.height*0.1],
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
                print(" Can't receive frame (stream end?). Exiting ...")
                break

            # get drone info from log and parking space info from parking db
            frame_time_ms = frame_to_ms(frame_number, 30)
            drone_info = self.log.drone_info(frame_time_ms)
            space_info = self.parking.find_space_by_location(drone_info.target_position)

            frame_cropped = fr.crop(frame, cropbox)  # use cropped frame for actual detections
            self.processor.process(frame_cropped)
            self.processor.update_max(frame_number)

            draw_center_lines(frame, vidbox)
            fr.draw(frame, cropbox, color=(128, 128, 128))

            draw_text(frame, f"Frame: {frame_number}", 0)
            draw_text(frame, f"Time: {frame_time_ms:0.2f}", 1)
            draw_text(frame, f"Objects: {len(self.processor.max_confidence)}", 2)
            if drone_info:
                dp = drone_info.drone_position
                tp = drone_info.target_position
                agl = drone_info.drone_alt_ft - drone_info.ground_alt_ft
                draw_text(frame, "Drone", 4)
                draw_text(frame, f" Location: {dp.x:0.2f}, {dp.y:0.2f}, {agl:+0.2f}AGL", 5)
                draw_text(frame, f" Target:   {tp.x:0.2f}, {tp.y:0.2f}", 6)
                draw_text(frame, f" DroneAlt: {drone_info.drone_alt_ft:0.2f}", 7)
                draw_text(frame, f" GrndAlt:  {drone_info.ground_alt_ft:0.2f}", 8)
                draw_text(frame, f" Compass: {drone_info.heading:0.2f}", 9)
                draw_text(frame, f" Gimbal:  {drone_info.gimbal_pitch:0.2f}", 10)
            if space_info:
                draw_text(frame, f"Parking space: {space_info.id} {space_info.required_permit}", 12)

            for p in self.processor.results_by_frame[frame_number]:
                # move p.box from position in cropped frame to full frame
                bbs = [b.translate(*cropbox.topleft) for b in p.boxes]
                outer_bb = bbs[0]
                mc = self.processor.max_confidence[p.track_id]
                if outer_bb.contains(*vidbox.center):
                    reg = self.vehicles.find_vehicle_by_plate(mc.text)
                    t = "None"
                    b = "None"
                    color = (255, 0, 0)  # blue
                    if space_info:
                        t = f"{mc.text}:{space_info.id}:{space_info.required_permit}"
                    if reg:
                        b = f"{reg.color} {reg.make}:{reg.permit_kind}"
                    if reg and space_info:
                        if reg.permit_kind == space_info.required_permit:
                            color = (0, 255, 0)  # green
                        else:
                            color = (0, 0, 255)  # red
                    fr.draw(frame, outer_bb, t, b, color)
                else:
                    fr.draw(frame, outer_bb,
                            f"{mc.track_id}:{mc.text}",
                            f"{mc.confidence:0.2f}",
                            (255, 255, 255))

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


def test() -> Application:
    a = Application(video="../data/DJI_0665.MP4",
                    log="../data/logs/Oct-17th-2023-02-53PM-Flight-Airdata.csv")
    a.select_active_log(1)
    return a
