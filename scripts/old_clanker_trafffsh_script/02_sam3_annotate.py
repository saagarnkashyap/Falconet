import argparse
import os
import cv2
import numpy as np
from ultralytics import SAM


def mask_to_boxes(mask: np.ndarray, min_area: int):
    """Turn a binary mask (possibly multiple blobs) into xyxy boxes."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    return boxes


def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def annotate(frames_dir, labels_dir, previews_dir, prompt, min_box_area, class_id=0):
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(previews_dir, exist_ok=True)

    model = SAM("sam3.pt")  # auto-downloads on first use

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png")))
    print(f"Found {len(frame_files)} frames to annotate with prompt='{prompt}'")

    for fname in frame_files:
        fpath = os.path.join(frames_dir, fname)
        img = cv2.imread(fpath)
        h, w = img.shape[:2]

        # Text-concept prompt -> segments every matching instance in the frame
        results = model(fpath, prompts=[prompt])
        r = results[0]

        boxes_yolo = []
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                mask_resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                for box in mask_to_boxes(mask_resized, min_box_area):
                    boxes_yolo.append(xyxy_to_yolo(box, w, h))

        # Write YOLO label file
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        with open(label_path, "w") as f:
            for xc, yc, bw, bh in boxes_yolo:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Save a preview so you can eyeball quality before training
        preview = img.copy()
        for xc, yc, bw, bh in boxes_yolo:
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(previews_dir, fname), preview)

        print(f"{fname}: {len(boxes_yolo)} boxes")

    print("\nDone. REVIEW dataset/previews/ before training — fix or drop bad frames.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="dataset/raw_frames")
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--previews", default="dataset/previews")
    ap.add_argument("--prompt", default="car", help="Text concept to segment, e.g. 'car'")
    ap.add_argument("--min_box_area", type=int, default=20, help="Filter out tiny noise blobs (pixels^2)")
    ap.add_argument("--class_id", type=int, default=0)
    args = ap.parse_args()
    annotate(args.frames, args.labels, args.previews, args.prompt, args.min_box_area, args.class_id)
