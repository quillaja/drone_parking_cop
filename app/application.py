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


def draw_center_lines(frame: cv2.Mat, vidbox: fr.Box):
    frame_h_center = int(vidbox.center[0])
    frame_v_center = int(vidbox.center[1])
    cv2.line(frame,
             pt1=(frame_h_center, 0),
             pt2=(frame_h_center, int(vidbox.bottomright[1])),
             color=(255, 0, 0), thickness=1)
    cv2.line(frame,
             pt1=(0, frame_v_center),
             pt2=(int(vidbox.bottomright[0]), frame_v_center),
             color=(255, 0, 0), thickness=1)


def draw_text(frame: cv2.Mat, text: str, line: int):
    TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX
    TEXT_COLOR = (255, 255, 255)
    # h, w = frame.shape
    tsize, _ = cv2.getTextSize(text=text, fontFace=TEXT_FONT,
                               fontScale=1, thickness=1)
    frame = cv2.putText(img=frame, text=text,
                        org=(0, (line+1)*(tsize[1]+4)),
                        fontFace=TEXT_FONT,
                        fontScale=1, color=TEXT_COLOR, thickness=1)


ROOT = Path(__file__).resolve().parent.parent
YOLO_DATA = ROOT/"assets/yolov8n.pt"  # yolov8n.pt for general objects
LICENSE_PLATE_DATA = ROOT/"assets/plate_database.csv"
PARKING_SPACE_DATA = ROOT/"assets/plates.gpkg"
PCC_DEM = ROOT/"assets/pcc_sylvania_dtm_2014_crs6559.tif"


class Application:
    def __init__(self, video: str, log: str) -> None:
        self.video = cv2.VideoCapture(video)
        self.video_segments = djilog.segment(djilog.read_log(
            log,
            input_crs="EPSG:4326",
            output_crs="EPSG:6559"))
        self.log: djilog.FlightLog = None

        self.parking = db.GeoPackageSpaceDB(PARKING_SPACE_DATA)
        with Path(LICENSE_PLATE_DATA).open() as file:
            self.vehicles = db.CSVVehicleDB(file.readlines())

        self.processor = fr.Processor(
            # detector=frame.YoloSortDetector(
            #     detector=YOLO(YOLO_DATA),
            #     tracker=Sort(min_hits=6)),
            detector=fr.YoloOnlyDetector(YOLO(YOLO_DATA)),
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
        SCALE = 1
        vm = self.get_video_metrics()
        frame_number = 0

        cropbox = fr.Box([vm.width*0.0, vm.height*0.0],
                         [vm.width*(1-0.0), vm.height*(1-0.0)])
        # box representing cropped frame
        vidbox = cropbox.translate(-cropbox.topleft[0], -cropbox.topleft[1]).scale(SCALE)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(vidbox.width), int(vidbox.height)))
        print(vidbox)

        while self.video.isOpened():
            has_frame, frame = self.video.read()
            # if frame is read correctly has_frame is True
            if not has_frame:
                print(" Can't receive frame (stream end?). Exiting ...")
                break

            frame_time_ms = frame_to_ms(frame_number, vm.fps)
            drone_pos = self.log.drone_info(frame_time_ms, djilog.DronePosition.position)
            drone_target = self.log.drone_info(frame_time_ms, djilog.DronePosition.target)
            space_info = self.parking.find_space_by_location(drone_target)

            # frame = fr.crop(frame, cropbox)
            frame = cv2.resize(frame, dsize=(0, 0), fx=SCALE, fy=SCALE)  # resize huge 4K video

            self.processor.process(frame)
            self.processor.update_max(frame_number)

            # print(f"=== frame {frame_number} of {vm.frames} = {frame_time_ms}ms ===")
            # print(f"pos {drone_pos}\ttar {drone_target}")
            # print(f"space: {space_info}")
            # print()
            # # print([(p.text, p.confidence) for p in self.processor.max_confidence.values()])
            # # print()

            draw_center_lines(frame, vidbox)

            if drone_pos:
                draw_text(frame, f"D: {drone_pos.x:0.2f}, {drone_pos.y:0.2f}", 0)
            if drone_target:
                draw_text(frame, f"T: {drone_target.x:0.2f}, {drone_target.y:0.2f}", 1)
            if space_info:
                draw_text(frame, f"S: {space_info.id} {space_info.required_permit}", 2)
            draw_text(frame, f"MC: {len(self.processor.max_confidence)}", 3)

            for p in self.processor.results_by_frame[frame_number]:
                checkbox = p.box  # .scale_center(1.5, 5)
                # fr.draw(frame, checkbox, box_color=(255, 0, 0))
                mc = self.processor.max_confidence[p.track_id]
                if checkbox.contains(*vidbox.center):
                    reg = self.vehicles.find_vehicle_by_plate(mc.text)
                    # print(p)
                    # print(space_info)
                    # print(reg)
                    # print()
                    t = "None"
                    b = "None"
                    if space_info:
                        t = f"{mc.text}:{space_info.id}:{space_info.required_permit}"
                    if reg:
                        b = f"{reg.color} {reg.make}:{reg.permit_kind}"
                    fr.draw(frame, p.box, t, b, (0, 0, 255))
                else:
                    fr.draw(frame, p.box,
                            f"{mc.track_id}:{mc.text}",
                            f"{mc.confidence:0.2f}")

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
