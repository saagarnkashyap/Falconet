import cv2
import numpy as np


def chromatic_aberration(img, shift_px):
    """Split and offset color channels for a glitchy 'reality breaking' look."""
    b, g, r = cv2.split(img)
    h, w = img.shape[:2]
    M_r = np.float32([[1, 0, shift_px], [0, 1, 0]])
    M_b = np.float32([[1, 0, -shift_px], [0, 1, 0]])
    r = cv2.warpAffine(r, M_r, (w, h))
    b = cv2.warpAffine(b, M_b, (w, h))
    return cv2.merge([b, g, r])


def add_bloom(img, threshold=200, blur_size=25, intensity=0.5):
    """Glowing highlight bloom - makes headlights/taillights radiate."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    bright = cv2.bitwise_and(img, mask_3ch)
    bloom = cv2.GaussianBlur(bright, (0, 0), sigmaX=blur_size)
    return cv2.addWeighted(img, 1.0, bloom, intensity, 0)


class MotionTrailBuffer:
    """Accumulates a fading trail of bright motion (headlights/taillights)
    across frames - call update() every frame, get the trail-augmented
    frame back."""

    def __init__(self, shape, decay=0.85):
        self.buffer = np.zeros(shape, dtype=np.float32)
        self.decay = decay

    def update(self, frame, threshold=190):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
        bright = (frame.astype(np.float32) * (mask_3ch / 255.0))

        self.buffer = self.buffer * self.decay + bright * (1 - self.decay) * 3
        self.buffer = np.clip(self.buffer, 0, 255)

        out = cv2.addWeighted(frame.astype(np.float32), 1.0, self.buffer, 0.9, 0)
        return np.clip(out, 0, 255).astype(np.uint8)


def unfold_warp(img, progress, strength=1.3):
    """Animated perspective warp - interpolates from identity (progress=0)
    to a 'far edge pulled toward viewer' bird's-eye-ish transform
    (progress=1). This is a stylistic 2D fake, not true 3D reprojection -
    but reads convincingly as 'the road unfolding' when animated across
    frames. Includes a compensating zoom so content doesn't just shrink
    into a corner."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    top_push = strength * w * 0.5
    bottom_pull = strength * w * 0.15
    dst_target = np.float32([
        [-top_push, 0], [w + top_push, 0],
        [w - bottom_pull, h], [bottom_pull, h],
    ])
    dst = src * (1 - progress) + dst_target * progress

    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(5, 5, 5))

    # Compensating zoom: as the warp stretches content outward/off-frame,
    # scale back in around center so the road stays visible and framed.
    zoom = 1.0 + progress * (strength * 0.35)
    if zoom > 1.0:
        zh, zw = int(h / zoom), int(w / zoom)
        y0, x0 = (h - zh) // 2, (w - zw) // 2
        cropped = warped[y0:y0 + zh, x0:x0 + zw]
        warped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return warped


def glowing_edges(img, color=(0, 255, 180), low=60, high=150, dilate=1):
    """Neon-line overlay tracing road/lane edges via Canny."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    if dilate:
        edges = cv2.dilate(edges, np.ones((dilate, dilate), np.uint8))
    edges_3ch = np.zeros_like(img)
    edges_3ch[edges > 0] = color
    edges_glow = cv2.GaussianBlur(edges_3ch, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1.0, edges_glow, 0.8, 0)


def extract_light_particles(img, threshold=195, min_area=2):
    """Find bright blobs (headlights/taillights) - returns list of
    (x, y, brightness_color) for particle rendering."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    particles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        b, g, r = img[cy, cx].tolist()
        particles.append((cx, cy, (int(b), int(g), int(r)), min(area, 40)))
    return particles


def render_particle_field(shape, particles, glow=True):
    """Black canvas with glowing colored dots at each particle location -
    the 'traffic becomes particles' abstract look."""
    canvas = np.zeros(shape, dtype=np.uint8)
    for x, y, color, size in particles:
        radius = max(2, int(np.sqrt(size)))
        cv2.circle(canvas, (x, y), radius, color, -1)
    if glow:
        blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=6)
        canvas = cv2.addWeighted(canvas, 1.0, blurred, 1.2, 0)
    return canvas


def draw_point_marker(img, box, color=(0, 255, 180), radius=4):
    """Small glowing dot at a box's center - for the 'machine perception,
    not yet boxes' phase."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius * 3, color, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=4)
    img = cv2.addWeighted(img, 1.0, overlay, 0.6, 0)
    cv2.circle(img, (cx, cy), radius, color, -1)
    return img
