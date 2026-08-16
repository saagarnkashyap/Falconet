import cv2
import numpy as np
from collections import deque, defaultdict


def draw_corner_bracket_box(img, box, color, corner_ratio=0.28, thickness=2):
    """L-shaped corner brackets instead of a full rectangle - the
    recognizable 'targeting/tracking' look."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    cl = max(4, int(min(w, h) * corner_ratio))

    corners = [
        ((x1, y1), (x1 + cl, y1), (x1, y1 + cl)),
        ((x2, y1), (x2 - cl, y1), (x2, y1 + cl)),
        ((x1, y2), (x1 + cl, y2), (x1, y2 - cl)),
        ((x2, y2), (x2 - cl, y2), (x2, y2 - cl)),
    ]
    for corner, h_end, v_end in corners:
        cv2.line(img, corner, h_end, color, thickness, cv2.LINE_AA)
        cv2.line(img, corner, v_end, color, thickness, cv2.LINE_AA)
    return img


def draw_id_tag(img, box, track_id, color):
    """Small 'ID: NN' label above a box, matching reference style."""
    x1, y1, x2, y2 = [int(v) for v in box]
    text = f"ID:{track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.4, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx, ty = x1, y1 - 6
    overlay = img.copy()
    cv2.rectangle(overlay, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)
    return img


class TrajectoryTrails:
    """Per-track fading trail of recent center points, drawn as a dashed,
    alpha-fading line - the glowing 'where it's been' path from the
    reference image."""

    def __init__(self, max_len=20):
        self.max_len = max_len
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, track_id, box):
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        self.history[track_id].append((cx, cy))

    def draw(self, img, track_id, color, dashed=True):
        pts = list(self.history.get(track_id, []))
        if len(pts) < 2:
            return img
        overlay = img.copy()
        n = len(pts)
        for i in range(1, n):
            if dashed and i % 2 == 0:
                continue
            alpha = i / n
            thickness = 1 if alpha < 0.5 else 2
            cv2.line(overlay, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        return img

    def prune(self, active_ids):
        stale = [tid for tid in self.history if tid not in active_ids]
        for tid in stale:
            del self.history[tid]


def draw_viewfinder_frame(img, color=(0, 220, 200), margin=14, length=40, thickness=2):
    """Four corner brackets framing the whole video - subtle 'system
    online' framing device, matches reference image border style."""
    h, w = img.shape[:2]
    pts = [
        ((margin, margin), (margin + length, margin), (margin, margin + length)),
        ((w - margin, margin), (w - margin - length, margin), (w - margin, margin + length)),
        ((margin, h - margin), (margin + length, h - margin), (margin, h - margin - length)),
        ((w - margin, h - margin), (w - margin - length, h - margin), (w - margin, h - margin - length)),
    ]
    for corner, h_end, v_end in pts:
        cv2.line(img, corner, h_end, color, thickness, cv2.LINE_AA)
        cv2.line(img, corner, v_end, color, thickness, cv2.LINE_AA)
    return img


def draw_title_block(img, title, subtitle=None, pos=(28, 40)):
    x, y = pos
    cv2.putText(img, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(img, subtitle, (x, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 1, cv2.LINE_AA)
    return img


def draw_legend_panel(img, entries, anchor_bottom_left=(24, None)):
    """entries: list of (label, color) tuples - dashed-line swatch legend,
    matching the reference 'TRAJECTORY PATHS' panel."""
    h = img.shape[0]
    x0, y0 = anchor_bottom_left[0], (anchor_bottom_left[1] or h - 24)
    line_gap = 24
    panel_h = 20 + line_gap * len(entries)
    y0 = y0 - panel_h
    panel_w = 170

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), 1)

    cv2.putText(img, "TRAJECTORY PATHS", (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (label, color) in enumerate(entries):
        ty = y0 + 18 + line_gap * (i + 1)
        cv2.line(img, (x0 + 12, ty - 5), (x0 + 32, ty - 5), color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 40, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_hud_stats_panel(img, total, counts, anchor_top_right=None):
    """Clean minimal stats readout - big TOTAL number + per-category rows,
    matching the reference 'DETECTION STATUS' + 'TOTAL COUNT' panels but
    without fabricated fields (no fake avg speed/occupancy)."""
    h, w = img.shape[:2]
    x2 = w - 24
    panel_w = 190
    line_gap = 28
    panel_h = 70 + line_gap * len(counts)
    x1 = x2 - panel_w
    y1 = 24
    y2 = y1 + panel_h

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.putText(img, "TOTAL VEHICLES", (x1 + 14, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, str(total), (x1 + 14, y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    ty = y1 + 78
    for label, count, color in counts:
        cv2.putText(img, label, (x1 + 14, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(img, str(count), (x2 - 34, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        ty += line_gap
    return img
