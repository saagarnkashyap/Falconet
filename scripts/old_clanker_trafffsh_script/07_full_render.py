import argparse
import sys
import os

import cv2
from ultralytics import YOLO, solutions

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.styling import (
    draw_glow_box, draw_translucent_zone, draw_count_label, draw_name_tag,
    draw_stats_panel, apply_cinematic_look, draw_hud_text, draw_scan_line,
    color_for_index,
)
from configs.regions import REGIONS, LINES

# VisDrone class ids -> readable label + category bucket
CLASS_LABELS = {
    2: "bicycle", 3: "car", 4: "van", 5: "truck",
    6: "tricycle", 7: "tricycle", 8: "bus", 9: "bike",
}
CATEGORY_MAP = {2: "bikes", 6: "bikes", 7: "bikes", 9: "bikes", 3: "cars", 5: "trucks", 4: "others", 8: "others"}
CATEGORY_ORDER = ["bikes", "cars", "trucks", "others"]
CATEGORY_COLOR = {"bikes": (0, 140, 255), "cars": (170, 255, 60), "trucks": (255, 210, 0), "others": (200, 0, 255)}

HUD_MESSAGES = [
    "AERIAL FEED - INITIALIZING",
    "CALIBRATING TRACKING SYSTEM",
    "LOCATING INTERSECTION...",
]


def run(video_path, model_path, out_path, cutover_sec, classes=None):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open {video_path}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = YOLO(model_path)
    region_counter = solutions.RegionCounter(region=REGIONS, model=model_path, classes=classes, show=False)
    zone_names = list(REGIONS.keys())

    seen_ids = {cat: set() for cat in CATEGORY_ORDER}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        if t < cutover_sec:
            # --- Cinematic-only pass, no detection ---
            styled = apply_cinematic_look(frame)
            progress = (t % 3.0) / 3.0  # scan line loops every 3s
            styled = draw_scan_line(styled, progress)
            msg = HUD_MESSAGES[int(t) % len(HUD_MESSAGES)]
            styled = draw_hud_text(styled, msg)

        else:
            # --- Full pipeline: zones + tracked/labeled boxes + scoreboard ---
            region_result = region_counter(frame.copy())
            region_counts = region_result.region_counts

            track_result = detector.track(frame, persist=True, classes=classes, conf=0.25, verbose=False)[0]
            styled = frame.copy()

            for i, name in enumerate(zone_names):
                color = color_for_index(i)
                pts = REGIONS[name]
                styled = draw_translucent_zone(styled, pts, color, alpha=0.20)
                cx = int(sum(p[0] for p in pts) / len(pts))
                cy = int(sum(p[1] for p in pts) / len(pts))
                styled = draw_count_label(styled, str(region_counts.get(name, 0)), (cx, cy), color=color)

            if track_result.boxes is not None and track_result.boxes.id is not None:
                boxes = track_result.boxes.xyxy.cpu().numpy()
                clss = track_result.boxes.cls.cpu().numpy().astype(int)
                ids = track_result.boxes.id.cpu().numpy().astype(int)

                for box, cls_id, track_id in zip(boxes, clss, ids):
                    category = CATEGORY_MAP.get(cls_id, "others")
                    seen_ids[category].add(track_id)
                    color = CATEGORY_COLOR[category]
                    styled = draw_glow_box(styled, box, color, glow_strength=10, thickness=1)
                    label = CLASS_LABELS.get(cls_id, "vehicle")
                    styled = draw_name_tag(styled, box, label, color)

            counts = {cat.upper(): len(seen_ids[cat]) for cat in CATEGORY_ORDER}
            styled = draw_stats_panel(styled, counts, anchor_bottom_right=(w - 24, h - 24))

        writer.write(styled)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames (t={t:.1f}s)...")

    cap.release()
    writer.release()
    print(f"\nDone. Output saved to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="../out/birdeye_full.mp4")
    ap.add_argument("--cutover", type=float, default=16.5, help="Seconds - detection pipeline starts here")
    ap.add_argument("--classes", type=int, nargs="*", default=[2, 3, 4, 5, 6, 7, 8, 9])
    args = ap.parse_args()
    run(args.video, args.model, args.out, args.cutover, args.classes)
