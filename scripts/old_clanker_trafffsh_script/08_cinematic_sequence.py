import argparse
import sys
import os

import cv2
import numpy as np
from ultralytics import YOLO, solutions

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.styling import (
    draw_glow_box, draw_translucent_zone, draw_count_label, draw_name_tag,
    draw_stats_panel, apply_cinematic_look, draw_hud_text, color_for_index,
)
from utils.cinematic import (
    chromatic_aberration, add_bloom, MotionTrailBuffer, unfold_warp,
    glowing_edges, extract_light_particles, render_particle_field,
    draw_point_marker,
)
from utils.hud import (
    draw_corner_bracket_box, draw_id_tag, TrajectoryTrails,
    draw_viewfinder_frame, draw_title_block, draw_legend_panel, draw_hud_stats_panel,
)
from configs.regions import REGIONS, LINES

CLASS_LABELS = {2: "bicycle", 3: "car", 4: "van", 5: "truck", 6: "tricycle", 7: "tricycle", 8: "bus", 9: "bike"}
CATEGORY_MAP = {2: "bikes", 6: "bikes", 7: "bikes", 9: "bikes", 3: "cars", 5: "trucks", 4: "others", 8: "others"}
CATEGORY_ORDER = ["bikes", "cars", "trucks", "others"]
CATEGORY_COLOR = {"bikes": (0, 140, 255), "cars": (170, 255, 60), "trucks": (255, 210, 0), "others": (200, 0, 255)}

# Phase boundaries in seconds - EDIT THESE if your clip's scene cuts differ
PHASE_BOUNDS = {
    "cinematic": (0.0, 1.0),
    "reality_break": (1.0, 2.8),
    "unfold": (2.8, 9.6),
    "perception": (9.6, 12.4),
    "particles": (12.4, 16.4),
    "data": (16.4, 999.0),
}


def phase_for(t):
    for name, (lo, hi) in PHASE_BOUNDS.items():
        if lo <= t < hi:
            progress = (t - lo) / max(hi - lo, 0.01)
            return name, min(progress, 1.0)
    return "data", 1.0


def run(video_path, model_path, out_path, classes=None):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open {video_path}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = YOLO(model_path)

    trail_buffer = MotionTrailBuffer((h, w, 3), decay=0.82)
    trajectories = TrajectoryTrails(max_len=20)
    seen_ids = {cat: set() for cat in CATEGORY_ORDER}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        phase, progress = phase_for(t)

        if phase == "cinematic":
            styled = apply_cinematic_look(frame)

        elif phase == "reality_break":
            shift = int(2 + progress * 8)
            styled = chromatic_aberration(frame, shift)
            styled = trail_buffer.update(styled)
            styled = add_bloom(styled, intensity=0.5)

        elif phase == "unfold":
            styled = unfold_warp(frame, progress, strength=1.1)
            styled = glowing_edges(styled, color=(0, 255, 180), dilate=1)

        elif phase == "perception":
            result = detector.track(frame, persist=True, classes=classes, conf=0.2, verbose=False)[0]
            styled = apply_cinematic_look(frame, vignette_strength=0.5)
            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().numpy():
                    styled = draw_point_marker(styled, box, color=(0, 255, 180))
            styled = draw_hud_text(styled, "TRACKING ACQUIRED")

        elif phase == "particles":
            particles = extract_light_particles(frame, threshold=185, min_area=2)
            styled = render_particle_field(frame.shape, particles)

        else:  # "data" - the tracking-system HUD look
            track_result = detector.track(frame, persist=True, classes=classes, conf=0.25, verbose=False)[0]
            styled = frame.copy()
            styled = draw_viewfinder_frame(styled)
            styled = draw_title_block(styled, "BIRDEYE VISION SYSTEM", "REAL-TIME TRAFFIC INTELLIGENCE")

            active_ids = set()
            if track_result.boxes is not None and track_result.boxes.id is not None:
                boxes = track_result.boxes.xyxy.cpu().numpy()
                clss = track_result.boxes.cls.cpu().numpy().astype(int)
                ids = track_result.boxes.id.cpu().numpy().astype(int)
                for box, cls_id, track_id in zip(boxes, clss, ids):
                    active_ids.add(track_id)
                    category = CATEGORY_MAP.get(cls_id, "others")
                    seen_ids[category].add(track_id)
                    color = CATEGORY_COLOR[category]
                    trajectories.update(track_id, box)
                    styled = trajectories.draw(styled, track_id, color)
                    styled = draw_corner_bracket_box(styled, box, color)
                    styled = draw_id_tag(styled, box, track_id, color)
            trajectories.prune(active_ids)

            legend = [(cat.upper(), CATEGORY_COLOR[cat]) for cat in CATEGORY_ORDER]
            styled = draw_legend_panel(styled, legend)

            total = sum(len(seen_ids[c]) for c in CATEGORY_ORDER)
            stat_rows = [(cat.upper(), len(seen_ids[cat]), CATEGORY_COLOR[cat]) for cat in CATEGORY_ORDER]
            styled = draw_hud_stats_panel(styled, total, stat_rows)

        writer.write(styled)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} (t={t:.1f}s, phase={phase})...")

    cap.release()
    writer.release()
    print(f"\nDone. Output saved to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="../out/birdeye_cinematic.mp4")
    ap.add_argument("--classes", type=int, nargs="*", default=[2, 3, 4, 5, 6, 7, 8, 9])
    args = ap.parse_args()
    run(args.video, args.model, args.out, args.classes)
