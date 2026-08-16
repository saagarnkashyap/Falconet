import argparse
import sys
import os

import cv2
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.styling import draw_glow_box, draw_stats_panel, color_for_index

CATEGORY_MAP = {
    9: "bikes", 2: "bikes", 6: "bikes", 7: "bikes",
    3: "cars",
    5: "trucks",
    4: "others", 8: "others",
}
CATEGORY_ORDER = ["bikes", "cars", "trucks", "others"]
CATEGORY_BOX_COLOR = {
    "bikes": (255, 140, 0),
    "cars": (0, 255, 180),
    "trucks": (0, 210, 255),
    "others": (255, 0, 200),
}


def run(video_path, model_path, out_path, classes=None, show_preview=False):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open {video_path}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = YOLO(model_path)

    seen_ids = {cat: set() for cat in CATEGORY_ORDER}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.track(frame, persist=True, classes=classes, conf=0.25, verbose=False)[0]
        styled = frame.copy()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)

            for box, cls_id, track_id in zip(boxes, clss, ids):
                category = CATEGORY_MAP.get(cls_id, "others")
                seen_ids[category].add(track_id)
                color = CATEGORY_BOX_COLOR[category]
                styled = draw_glow_box(styled, box, color, glow_strength=10, thickness=1)

        counts = {cat.upper(): len(seen_ids[cat]) for cat in CATEGORY_ORDER}
        styled = draw_stats_panel(styled, counts, anchor_bottom_right=(w - 24, h - 24))

        writer.write(styled)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames... {counts}")

        if show_preview:
            cv2.imshow("BirdEye", styled)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    final_counts = {cat.upper(): len(seen_ids[cat]) for cat in CATEGORY_ORDER}
    print(f"\nDone. Final unique counts: {final_counts}")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help="Path to trained best.pt")
    ap.add_argument("--out", default="out/birdeye_output.mp4")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                     help="Class ids to detect, e.g. --classes 2 3 4 5 6 7 8 9 for all VisDrone vehicle types")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    run(args.video, args.model, args.out, args.classes, args.show)





# """
# Step 5 (v2) - Track every vehicle, bucket it into a category (bikes, cars,
# trucks, others), and show a running unique-count scoreboard panel in the
# bottom-right corner. No zones/lines - just detection + tracking + a clean
# stats readout.
#
# Counts are UNIQUE vehicles seen so far (tracked by ID), not a per-frame
# snapshot - so the numbers climb steadily through the video rather than
# jumping around, which is what you want for a "total counted" readout.
#
# VisDrone class ids -> category buckets:
#   bikes  = motor(9), bicycle(2), tricycle(6), awning-tricycle(7)
#   cars   = car(3)
#   trucks = truck(5)
#   others = van(4), bus(8)
#
# Usage:
#     python 05_run_counting.py --video ../data/input_video.mp4 \
#         --model runs/birdeye/yolo26n_visdrone/weights/best.pt \
#         --out ../out/birdeye_output.mp4
# """
# import argparse
# import sys
# import os
#
# import cv2
# from ultralytics import YOLO
#
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from utils.styling import draw_glow_box, draw_stats_panel, color_for_index
#
# CATEGORY_MAP = {
#     9: "bikes", 2: "bikes", 6: "bikes", 7: "bikes",
#     3: "cars",
#     5: "trucks",
#     4: "others", 8: "others",
# }
# CATEGORY_ORDER = ["bikes", "cars", "trucks", "others"]
# CATEGORY_BOX_COLOR = {
#     "bikes": (255, 140, 0),
#     "cars": (0, 255, 180),
#     "trucks": (0, 210, 255),
#     "others": (255, 0, 200),
# }
#
#
# def run(video_path, model_path, out_path, classes=None, show_preview=False):
#     cap = cv2.VideoCapture(video_path)
#     assert cap.isOpened(), f"Could not open {video_path}"
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25
#
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#
#     detector = YOLO(model_path)
#
#     seen_ids = {cat: set() for cat in CATEGORY_ORDER}
#     frame_idx = 0
#
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#
#         result = detector.track(frame, persist=True, classes=classes, verbose=False)[0]
#         styled = frame.copy()
#
#         if result.boxes is not None and result.boxes.id is not None:
#             boxes = result.boxes.xyxy.cpu().numpy()
#             clss = result.boxes.cls.cpu().numpy().astype(int)
#             ids = result.boxes.id.cpu().numpy().astype(int)
#
#             for box, cls_id, track_id in zip(boxes, clss, ids):
#                 category = CATEGORY_MAP.get(cls_id, "others")
#                 seen_ids[category].add(track_id)
#                 color = CATEGORY_BOX_COLOR[category]
#                 styled = draw_glow_box(styled, box, color, glow_strength=10, thickness=1)
#
#         counts = {cat.upper(): len(seen_ids[cat]) for cat in CATEGORY_ORDER}
#         styled = draw_stats_panel(styled, counts, anchor_bottom_right=(w - 24, h - 24))
#
#         writer.write(styled)
#         frame_idx += 1
#         if frame_idx % 30 == 0:
#             print(f"Processed {frame_idx} frames... {counts}")
#
#         if show_preview:
#             cv2.imshow("BirdEye", styled)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#
#     cap.release()
#     writer.release()
#     if show_preview:
#         cv2.destroyAllWindows()
#
#     final_counts = {cat.upper(): len(seen_ids[cat]) for cat in CATEGORY_ORDER}
#     print(f"\nDone. Final unique counts: {final_counts}")
#     print(f"Output saved to: {out_path}")
#
#
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--video", required=True)
#     ap.add_argument("--model", required=True, help="Path to trained best.pt")
#     ap.add_argument("--out", default="out/birdeye_output.mp4")
#     ap.add_argument("--classes", type=int, nargs="*", default=None,
#                      help="Class ids to detect, e.g. --classes 2 3 4 5 6 7 8 9 for all VisDrone vehicle types")
#     ap.add_argument("--show", action="store_true")
#     args = ap.parse_args()
#     run(args.video, args.model, args.out, args.classes, args.show)
#
#
#
#
# # """
# # Step 5 — The finale. Runs your trained YOLO26n model on the full video with
# # BOTH Ultralytics solutions combined:
# #   - RegionCounter: live occupancy per named zone (the "28", "24", "43" style
# #     numbers in the reference video)
# #   - ObjectCounter: cumulative crossings per line (the red directional lines)
# #
# # Detection/tracking for the visual boxes is run directly via model.track()
# # so we have full control over rendering (rather than relying on internal
# # attributes of the solutions' return objects, which vary by version) —
# # counts still come from the two solutions above.
# #
# # Usage:
# #     python 05_run_counting.py --video path/to/bengaluru_drone.mp4 \
# #         --model ../runs/birdeye/yolo26n_aerial_cars/weights/best.pt \
# #         --out out/bengaluru_birdeye.mp4
# # """
# # import argparse
# # import sys
# # import os
# #
# # import cv2
# # from ultralytics import YOLO, solutions
# #
# # sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# # from utils.styling import draw_glow_box, draw_translucent_zone, draw_count_label, color_for_index
# # from configs.regions import REGIONS, LINES
# #
# #
# # def run(video_path, model_path, out_path, classes=None, show_preview=False):
# #     cap = cv2.VideoCapture(video_path)
# #     assert cap.isOpened(), f"Could not open {video_path}"
# #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #     fps = cap.get(cv2.CAP_PROP_FPS) or 25
# #
# #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
# #     writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
# #
# #     # Model used directly for drawing (full control over the visual boxes)
# #     detector = YOLO(model_path)
# #
# #     # Solutions used purely for their COUNTING logic — we pass them a
# #     # throwaway copy of each frame so their own default rendering never
# #     # touches the image we actually output.
# #     region_counter = solutions.RegionCounter(region=REGIONS, model=model_path, classes=classes, show=False)
# #     line_counters = {
# #         name: solutions.ObjectCounter(region=pts, model=model_path, classes=classes, show=False)
# #         for name, pts in LINES.items()
# #     }
# #
# #     zone_names = list(REGIONS.keys())
# #     frame_idx = 0
# #
# #     while True:
# #         ok, frame = cap.read()
# #         if not ok:
# #             break
# #
# #         # --- Counts (solutions mutate whatever image they're given, so
# #         # always feed them a disposable copy) ---
# #         region_result = region_counter(frame.copy())
# #         region_counts = region_result.region_counts
# #
# #         line_counts = {}
# #         for name, counter in line_counters.items():
# #             res = counter(frame.copy())
# #             line_counts[name] = res.in_count + res.out_count
# #
# #         # --- Boxes for drawing (run directly, fully under our control) ---
# #         track_result = detector.track(frame, persist=True, classes=classes, verbose=False)[0]
# #         boxes_xyxy = track_result.boxes.xyxy.cpu().numpy() if track_result.boxes is not None else []
# #
# #         # --- Render from the ORIGINAL untouched frame, fully our style ---
# #         styled = frame.copy()
# #
# #         # Zones: translucent fill + border + count label
# #         for i, name in enumerate(zone_names):
# #             color = color_for_index(i)
# #             pts = REGIONS[name]
# #             styled = draw_translucent_zone(styled, pts, color, alpha=0.22)
# #             cx = int(sum(p[0] for p in pts) / len(pts))
# #             cy = int(sum(p[1] for p in pts) / len(pts))
# #             styled = draw_count_label(styled, str(region_counts.get(name, 0)), (cx, cy), color=color)
# #
# #         # Crossing lines: draw as a highlighted line + running total
# #         for i, (name, pts) in enumerate(LINES.items()):
# #             color = color_for_index(i + len(zone_names))
# #             (x1, y1), (x2, y2) = pts
# #             cv2.line(styled, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
# #             styled = draw_count_label(styled, f"{name}: {line_counts[name]}", (x1, max(20, y1 - 10)), color=color)
# #
# #         # Tracked vehicle boxes, our own clean glow style (no clutter text)
# #         for box in boxes_xyxy:
# #             styled = draw_glow_box(styled, box, (0, 255, 180), glow_strength=10, thickness=1)
# #
# #         writer.write(styled)
# #         frame_idx += 1
# #         if frame_idx % 30 == 0:
# #             print(f"Processed {frame_idx} frames...")
# #
# #         if show_preview:
# #             cv2.imshow("BirdEye", styled)
# #             if cv2.waitKey(1) & 0xFF == ord("q"):
# #                 break
# #
# #     cap.release()
# #     writer.release()
# #     if show_preview:
# #         cv2.destroyAllWindows()
# #     print(f"\nDone. Output saved to: {out_path}")
# #
# #
# # if __name__ == "__main__":
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--video", required=True)
# #     ap.add_argument("--model", required=True, help="Path to trained best.pt")
# #     ap.add_argument("--out", default="out/birdeye_output.mp4")
# #     ap.add_argument("--classes", type=int, nargs="*", default=None,
# #                      help="Class ids to count, e.g. --classes 3 for VisDrone 'car' only")
# #     ap.add_argument("--show", action="store_true")
# #     args = ap.parse_args()
# #     run(args.video, args.model, args.out, args.classes, args.show)
