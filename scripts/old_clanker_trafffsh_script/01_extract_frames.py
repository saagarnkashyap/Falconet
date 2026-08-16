import argparse
import os
import cv2


def extract_frames(video_path: str, out_dir: str, num_frames: int):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Video reports 0 frames — check the file/codec.")

    # Evenly spaced frame indices across the whole clip
    step = max(total // num_frames, 1)
    indices = list(range(0, total, step))[:num_frames]

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        fname = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(fname, frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to source drone video")
    ap.add_argument("--out", default="dataset/raw_frames", help="Output frame folder")
    ap.add_argument("--num_frames", type=int, default=30)
    args = ap.parse_args()
    extract_frames(args.video, args.out, args.num_frames)
