import cv2
import numpy as np

# A punchy palette so each region gets a distinct, readable color
PALETTE = [
    (0, 210, 255),   # amber/yellow
    (255, 0, 200),   # magenta
    (80, 220, 80),   # green
    (255, 140, 0),   # orange
    (0, 180, 255),   # cyan-blue
    (200, 80, 255),  # purple
]


def color_for_index(i: int):
    return PALETTE[i % len(PALETTE)]


def draw_glow_box(img, box, color, glow_strength=15, thickness=2):
    """Draw a bounding box with a soft outer glow, additively blended."""
    x1, y1, x2, y2 = [int(v) for v in box]
    glow_layer = np.zeros_like(img, dtype=np.uint8)
    cv2.rectangle(glow_layer, (x1, y1), (x2, y2), color, thickness + 4)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=glow_strength / 3)
    cv2.add(img, glow_layer, img)  # additive blend = glow
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_translucent_zone(img, polygon_pts, color, alpha=0.25, border_thickness=2):
    """Fill a polygonal zone with a translucent color + solid border."""
    overlay = img.copy()
    pts = np.array(polygon_pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=border_thickness)
    return img


def draw_stats_panel(img, counts: dict, anchor_bottom_right, padding=16, line_gap=34):
    """Draw a semi-transparent rounded-corner stats panel with one line per
    category, anchored to its bottom-right corner (e.g. near the video's
    bottom-right edge)."""
    x2, y2 = anchor_bottom_right
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "LIVE COUNT"
    title_scale, line_scale = 0.6, 0.65
    thick = 2

    # Work out panel width from the longest line
    lines = [f"{name}: {count}" for name, count in counts.items()]
    max_w = max(
        [cv2.getTextSize(title, font, title_scale, thick)[0][0]]
        + [cv2.getTextSize(l, font, line_scale, thick)[0][0] for l in lines]
    )
    panel_w = max_w + padding * 2
    panel_h = padding * 2 + line_gap * (len(lines) + 1)
    x1, y1 = x2 - panel_w, y2 - panel_h

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    ty = y1 + padding + 18
    cv2.putText(img, title, (x1 + padding, ty), font, title_scale, (255, 255, 255), thick, cv2.LINE_AA)
    for i, (name, count) in enumerate(counts.items()):
        ty += line_gap
        color = color_for_index(i)
        cv2.putText(img, f"{name}: {count}", (x1 + padding, ty), font, line_scale, color, thick, cv2.LINE_AA)

    return img