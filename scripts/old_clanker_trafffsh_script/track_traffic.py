import argparse
import collections
import time

import cv2
import numpy as np
from ultralytics import YOLO


VISDRONE_CLASSES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}

COLOR_CAR = (255, 220, 90)      # cyan
COLOR_MOTO = (203, 61, 255)     # magenta
COLOR_PERSON = (32, 176, 255)   # amber

CLASS_COLOR = {
    "car": COLOR_CAR, "van": COLOR_CAR, "truck": COLOR_CAR, "bus": COLOR_CAR,
    "bicycle": COLOR_MOTO, "motor": COLOR_MOTO, "tricycle": COLOR_MOTO,
    "awning-tricycle": COLOR_MOTO,
    "pedestrian": COLOR_PERSON, "people": COLOR_PERSON,
}

# Grouping for the "Detection Status" panel (Style C)
GROUP_CARS = {"car", "van"}
GROUP_TRUCKS = {"truck", "bus"}
GROUP_MOTOS = {"motor", "bicycle", "tricycle", "awning-tricycle"}
GROUP_PEDS = {"pedestrian", "people"}

TRAIL_LENGTH = 30
CORNER_LEN_RATIO = 0.22
CORNER_THICKNESS = 2
BOX_MIN_ARM = 8

FLOW_MONITOR_WINDOW = (2.8, 6.0)
SMART_ANALYTICS_WINDOW = (6.0, 10.0)
DENSITY_MAP_WINDOW = (11.9, 15.2)

SPEED_SCALE_KMH = 0.9          # px/frame * fps -> rough km/h
FLOW_WINDOW_SEC = 8.0          # rolling window for veh/min estimates
OCCUPANCY_CAPACITY = 45        # active tracks that reads as 100% occupancy
DENSITY_GRID = (18, 32)        # rows, cols for the heatmap accumulator
DENSITY_DECAY = 0.90



def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def class_color(class_name: str) -> tuple:
    return CLASS_COLOR.get(class_name, (255, 255, 255))


def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=CORNER_THICKNESS):
    w, h = x2 - x1, y2 - y1
    arm_x = max(BOX_MIN_ARM, int(w * CORNER_LEN_RATIO))
    arm_y = max(BOX_MIN_ARM, int(h * CORNER_LEN_RATIO))
    corners = [
        [(x1, y1), (x1 + arm_x, y1)], [(x1, y1), (x1, y1 + arm_y)],
        [(x2, y1), (x2 - arm_x, y1)], [(x2, y1), (x2, y1 + arm_y)],
        [(x1, y2), (x1 + arm_x, y2)], [(x1, y2), (x1, y2 - arm_y)],
        [(x2, y2), (x2 - arm_x, y2)], [(x2, y2), (x2, y2 - arm_y)],
    ]
    for (p1, p2) in corners:
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_id_tag(frame, x1, y1, track_id, color):
    label = f"ID:{int(track_id):02d}"
    org = (x1, max(12, y1 - 6))
    cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_trail(frame, points: collections.deque, color):
    n = len(points)
    if n < 2:
        return
    overlay = frame.copy()
    for i in range(1, n):
        alpha = i / n
        thickness = 1 if i < n * 0.6 else 2
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(overlay, points[i - 1], points[i], faded, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)


def draw_timestamp(frame, frame_idx: int, fps: float):
    """Default minimal timestamp — used outside all overlay windows."""
    seconds = frame_idx / fps if fps > 0 else 0
    hh, mm, ss = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    text = f"{hh:02d}:{mm:02d}:{ss:02d}"
    cv2.putText(frame, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)


def update_trails(trails: dict, track_id: int, centroid: tuple):
    if track_id not in trails:
        trails[track_id] = collections.deque(maxlen=TRAIL_LENGTH)
    trails[track_id].append(centroid)


def prune_trails(trails: dict, active_ids: set):
    for tid in [t for t in trails if t not in active_ids]:
        del trails[tid]



def draw_panel_bg(frame, x, y, w, h, alpha=0.55, color=(15, 15, 18), border=(70, 70, 76)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)


def draw_text(frame, text, x, y, size=0.5, color=(230, 230, 230), weight=1, halo=True):
    if halo:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), weight + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight, cv2.LINE_AA)


def draw_sparkline(frame, values, x, y, w, h, color=(255, 220, 90)):
    if len(values) < 2:
        return
    vmin, vmax = min(values), max(values)
    span = max(vmax - vmin, 1e-6)
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        px = x + int(i / (n - 1) * w)
        py = y + h - int((v - vmin) / span * h)
        pts.append((px, py))
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)


class FrameStats:
    """Rolling metrics computed once per frame and reused by all overlay styles."""

    def __init__(self, fps):
        self.fps = fps
        self.speed_history = collections.deque(maxlen=60)
        self.flow_history = collections.deque(maxlen=60)
        self.entry_events = collections.deque()   # (t_sec, direction)
        self.logged_ids = set()
        self.density_grid = np.zeros(DENSITY_GRID, dtype=np.float32)

    def update(self, t_sec, tracks, trails, frame_w, frame_h):
        """tracks: list of (track_id, cls_name, cx, cy)"""
        speeds = []
        for track_id, cls_name, cx, cy in tracks:
            pts = trails.get(track_id)
            if pts and len(pts) >= 2:
                (px, py) = pts[-2]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                speeds.append(dist * self.fps * SPEED_SCALE_KMH * 0.1)
        avg_speed = float(np.median(speeds)) if speeds else 0.0
        self.speed_history.append(avg_speed)

        for track_id, cls_name, cx, cy in tracks:
            if track_id in self.logged_ids:
                continue
            pts = trails.get(track_id)
            if pts and len(pts) >= 3:
                dy = pts[-1][1] - pts[0][1]
                direction = "north" if dy < 0 else "south"
                self.entry_events.append((t_sec, direction))
                self.logged_ids.add(track_id)

        while self.entry_events and t_sec - self.entry_events[0][0] > FLOW_WINDOW_SEC:
            self.entry_events.popleft()

        self.flow_history.append(len(tracks))

        self.density_grid *= DENSITY_DECAY
        rows, cols = DENSITY_GRID
        for _, _, cx, cy in tracks:
            r = min(rows - 1, max(0, int(cy / frame_h * rows)))
            c = min(cols - 1, max(0, int(cx / frame_w * cols)))
            self.density_grid[r, c] += 1.0

    def northbound_per_min(self):
        n = sum(1 for _, d in self.entry_events if d == "north")
        return int(n * 60 / FLOW_WINDOW_SEC)

    def southbound_per_min(self):
        n = sum(1 for _, d in self.entry_events if d == "south")
        return int(n * 60 / FLOW_WINDOW_SEC)

    def avg_speed(self):
        return float(np.mean(self.speed_history)) if self.speed_history else 0.0

    def occupancy_pct(self, active_count):
        return int(min(100, active_count / OCCUPANCY_CAPACITY * 100))

    def density_pct(self):
        return int(min(100, float(self.density_grid.mean()) * 12))

def draw_style_flow_monitoring(frame, stats: FrameStats, now_str):
    h, w = frame.shape[:2]

    draw_text(frame, "FLOW MONITORING", 20, 34, 0.7, (235, 235, 235), 2)
    draw_text(frame, "REAL-TIME TRACKING", 20, 56, 0.42, (150, 150, 150), 1)

    draw_panel_bg(frame, 20, 80, 190, 150)
    draw_text(frame, "DIRECTION FLOW", 34, 104, 0.38, (150, 150, 150))
    draw_text(frame, "NORTHBOUND", 34, 126, 0.36, (150, 150, 150))
    draw_text(frame, f"{stats.northbound_per_min()}", 34, 156, 0.75, (255, 220, 90), 2)
    draw_text(frame, "VEH/MIN", 34, 172, 0.32, (140, 140, 140))
    draw_text(frame, "SOUTHBOUND", 34, 196, 0.36, (150, 150, 150))
    draw_text(frame, f"{stats.southbound_per_min()}", 34, 216, 0.6, (203, 61, 255), 2)

    # draw_panel_bg(frame, 20, 240, 190, 60)
    # draw_text(frame, "TIME", 34, 262, 0.36, (150, 150, 150))
    # draw_text(frame, now_str, 34, 282, 0.5, (230, 230, 230), 1)

    rx = w - 190
    # draw_panel_bg(frame, rx, 80, 170, 80)
    # draw_text(frame, "AVG SPEED", rx + 14, 102, 0.36, (150, 150, 150))
    # draw_text(frame, f"{stats.avg_speed():.0f}", rx + 14, 132, 0.7, (255, 220, 90), 2)
    # draw_text(frame, "KM/H", rx + 14, 148, 0.3, (140, 140, 140))

    draw_panel_bg(frame, rx, 170, 170, 80)
    draw_text(frame, "TRAFFIC FLOW", rx + 14, 192, 0.36, (150, 150, 150))
    draw_sparkline(frame, list(stats.flow_history), rx + 14, 200, 142, 40)

    draw_panel_bg(frame, rx, 260, 170, 50)
    draw_text(frame, "ROAD STATUS", rx + 14, 280, 0.36, (150, 150, 150))
    draw_text(frame, "NORMAL", rx + 14, 300, 0.5, (110, 220, 110), 1)


def draw_style_smart_analytics(frame, stats: FrameStats, class_counts, now_str):
    h, w = frame.shape[:2]

    draw_text(frame, "SMART TRAFFIC", 20, 32, 0.65, (235, 235, 235), 2)
    draw_text(frame, "ANALYTICS SYSTEM", 20, 54, 0.65, (235, 235, 235), 2)

    draw_panel_bg(frame, 20, 96, 190, 120)
    rows = [
        ("CARS", class_counts.get("cars", 0), COLOR_CAR),
        ("MOTORCYCLES", class_counts.get("motos", 0), COLOR_MOTO),
        ("TRUCKS", class_counts.get("trucks", 0), COLOR_CAR),
        ("PEDESTRIANS", class_counts.get("peds", 0), COLOR_PERSON),
    ]
    draw_text(frame, "DETECTION STATUS", 34, 112, 0.36, (150, 150, 150))
    yy = 138
    for label, count, color in rows:
        cv2.circle(frame, (38, yy - 4), 4, color, -1)
        draw_text(frame, label, 50, yy, 0.36, (200, 200, 200))
        draw_text(frame, str(count), 175, yy, 0.4, (230, 230, 230), 1)
        yy += 26

    draw_panel_bg(frame, 20, 226, 190, 96)
    draw_text(frame, "TRAJECTORY PATHS", 34, 248, 0.36, (150, 150, 150))
    legend = [("CAR", COLOR_CAR), ("MOTORCYCLE", COLOR_MOTO), ("PEDESTRIAN", COLOR_PERSON)]
    yy = 268
    for label, color in legend:
        cv2.line(frame, (34, yy - 4), (54, yy - 4), color, 2, cv2.LINE_AA)
        draw_text(frame, label, 60, yy, 0.34, (190, 190, 190))
        yy += 18

    rx = w - 170
    draw_panel_bg(frame, rx, 96, 150, 80)
    draw_text(frame, "TRAFFIC FLOW", rx + 12, 118, 0.34, (150, 150, 150))
    draw_sparkline(frame, list(stats.flow_history), rx + 12, 126, 126, 40, COLOR_CAR)

    draw_panel_bg(frame, rx, 186, 150, 60)
    draw_text(frame, "OCCUPANCY", rx + 12, 206, 0.34, (150, 150, 150))
    occ = stats.occupancy_pct(sum(class_counts.values()))
    draw_text(frame, f"{occ}%", rx + 12, 232, 0.6, (255, 220, 90), 2)

    # draw_panel_bg(frame, rx, 256, 150, 60)
    # draw_text(frame, "AVG SPEED", rx + 12, 276, 0.34, (150, 150, 150))
    # draw_text(frame, f"{stats.avg_speed():.0f} KM/H", rx + 12, 300, 0.5, (230, 230, 230), 1)


def draw_style_density_map(frame, stats: FrameStats):
    h, w = frame.shape[:2]

    grid_u8 = np.clip(stats.density_grid * 40, 0, 255).astype(np.uint8)
    heat = cv2.resize(grid_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=25)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    mask = (heat > 15).astype(np.float32)[..., None] * 0.45
    frame[:] = (frame * (1 - mask) + heat_color * mask).astype(np.uint8)

    draw_text(frame, "DENSITY MAP", 20, 32, 0.65, (235, 235, 235), 2)
    draw_text(frame, "LIVE ANALYSIS", 20, 52, 0.4, (150, 150, 150), 1)

    draw_panel_bg(frame, 20, 76, 130, 110)
    draw_text(frame, "DENSITY LEVEL", 32, 96, 0.32, (150, 150, 150))
    grad = np.linspace(255, 0, 60).astype(np.uint8).reshape(-1, 1)
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    frame[104:164, 32:44] = grad_color
    draw_text(frame, "HIGH", 50, 112, 0.32, (230, 90, 90))
    draw_text(frame, "LOW", 50, 160, 0.32, (90, 140, 230))

    pct = stats.density_pct()
    status = "HEAVY CONGESTION" if pct > 70 else ("MODERATE" if pct > 40 else "LIGHT")
    status_color = (60, 60, 230) if pct > 70 else ((60, 170, 230) if pct > 40 else (110, 220, 110))
    draw_panel_bg(frame, 20, 196, 190, 76)
    draw_text(frame, "TRAFFIC DENSITY", 34, 218, 0.34, (150, 150, 150))
    draw_text(frame, f"{pct}%", 34, 248, 0.7, status_color, 2)
    draw_text(frame, status, 34, 264, 0.34, status_color, 1)


def get_overlay_style(t_sec):
    if FLOW_MONITOR_WINDOW[0] <= t_sec < FLOW_MONITOR_WINDOW[1]:
        return "flow"
    if SMART_ANALYTICS_WINDOW[0] <= t_sec < SMART_ANALYTICS_WINDOW[1]:
        return "analytics"
    if DENSITY_MAP_WINDOW[0] <= t_sec < DENSITY_MAP_WINDOW[1]:
        return "density"
    return "default"


def process_video(source: str, weights_path: str, out_path: str,
                   conf: float = 0.35, classes: list = None, tracker_cfg: str = "bytetrack.yaml"):
    model = load_model(weights_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    trails = {}
    stats = FrameStats(fps)
    frame_idx = 0
    t0 = time.time()

    results_stream = model.track(
        source=source, conf=conf, classes=classes, tracker=tracker_cfg,
        persist=True, stream=True, verbose=False,
    )

    for result in results_stream:
        frame = result.orig_img.copy()
        t_sec = frame_idx / fps
        active_ids = set()
        tracks = []
        counts = {"cars": 0, "motos": 0, "trucks": 0, "peds": 0}

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), track_id, cls_id in zip(boxes, ids, cls_ids):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cls_name = VISDRONE_CLASSES.get(cls_id, "unknown")
                color = class_color(cls_name)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                update_trails(trails, track_id, (cx, cy))
                active_ids.add(track_id)
                tracks.append((track_id, cls_name, cx, cy))

                if cls_name in GROUP_CARS:
                    counts["cars"] += 1
                elif cls_name in GROUP_TRUCKS:
                    counts["trucks"] += 1
                elif cls_name in GROUP_MOTOS:
                    counts["motos"] += 1
                elif cls_name in GROUP_PEDS:
                    counts["peds"] += 1

                draw_trail(frame, trails[track_id], color)
                draw_corner_box(frame, x1, y1, x2, y2, color)
                draw_id_tag(frame, x1, y1, track_id, color)

        prune_trails(trails, active_ids)
        stats.update(t_sec, tracks, trails, width, height)

        style = get_overlay_style(t_sec)
        hh, mm, ss = int(t_sec // 3600), int((t_sec % 3600) // 60), int(t_sec % 60)
        now_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        if style == "flow":
            draw_style_flow_monitoring(frame, stats, now_str)
        elif style == "analytics":
            draw_style_smart_analytics(frame, stats, counts, now_str)
        elif style == "density":
            draw_style_density_map(frame, stats)
        else:
            draw_timestamp(frame, frame_idx, fps)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    elapsed = time.time() - t0
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cinematic YOLO traffic tracking renderer")
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="output_tracked.mp4")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    args = parser.parse_args()

    process_video(
        source=args.source, weights_path=args.weights, out_path=args.out,
        conf=args.conf, classes=args.classes, tracker_cfg=args.tracker,
    )


if __name__ == "__main__":
    main()
