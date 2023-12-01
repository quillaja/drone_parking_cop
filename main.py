import argparse
from pathlib import Path

from app.application import Application


def main(args: argparse.Namespace):
    args.video = Path(args.video).resolve()
    if not args.video.exists():
        print(f"video file '{str(args.video)}' does not exist.")
        return
    args.log = Path(args.log).resolve()
    if not args.log.exists():
        print(f"log file '{args.log}' does not exist.")
        return
    if args.log.suffix.lower() != ".csv":
        print(f"log file must be a csv.")

    app = Application(
        video=str(args.video),
        log=str(args.log))

    if args.segment is None:
        print("No log segment selected.")
        print(f"Log segments found in '{args.log}' are:")
        for i, s, e, d in app.get_segment_selection():
            print(f"\t{i}\t{s:%c}\t{e:%c}\t{d:0.1f}s")
        print(f"The video is approximately {app.get_video_metrics().length_sec:0.1f}s long.")
        return

    num_segs = len(app.get_segment_selection())
    if args.segment < 0 or args.segment >= num_segs:
        print(f"segment selection {args.segment} is not in the acceptable range of 0-{num_segs-1}.")
        return

    print(f"Processing video...")
    app.select_active_log(args.segment)
    app.output_video = "output.avi"
    app.run()
    print(f"Finished processing.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="drone_traffic_cop",
        description="Your new robot overlord.")
    parser.add_argument("-v", "--video", dest="video", type=str, required=True,
                        help="The video footage to process.")
    parser.add_argument("-l", "--log", dest="log", type=str, required=True,
                        help="The log recorded during the video's flight.")
    parser.add_argument("-s", "--segment", dest="segment", type=int,
                        help="An number indicating which log segment coinsides with the video, when a log file has more than one video segment.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # log 2-53PM
    # 665 = 1
    # 666 = 2
    # 667 = 3
    # 668 = 4
    # a = Application(video="../data/DJI_0667.MP4",
    #                 log="../data/logs/Oct-17th-2023-02-53PM-Flight-Airdata.csv")
    # a.select_active_log(3)
    # a.run()
