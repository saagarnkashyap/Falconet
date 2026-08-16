import argparse
import sys
import os

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.regions import REGIONS, LINES

PALETTE = [(0, 210, 255), (255, 0, 200), (80, 220, 80), (255, 140, 0), (0, 180, 255)]


def preview(video_path, at_sec, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_sec * fps))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {at_sec}s — check the video length.")

    for i, (name, pts) in enumerate(REGIONS.items()):
        color = PALETTE[i % len(PALETTE)]
        overlay = img.copy()
        p = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [p], color)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.polylines(img, [p], True, color, 2)
        cx, cy = int(np.mean([x for x, y in pts])), int(np.mean([y for x, y in pts]))
        cv2.putText(img, name, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for name, (p1, p2) in LINES.items():
        cv2.line(img, p1, p2, (0, 0, 255), 3)
        cv2.putText(img, name, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(out_path, img)
    print(f"Saved preview to {out_path} — open it and check alignment before running 05_run_counting.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--at_sec", type=float, default=0, help="Timestamp to grab the check frame from")
    ap.add_argument("--out", default="../regions_preview.jpg")
    args = ap.parse_args()
    preview(args.video, args.at_sec, args.out)
