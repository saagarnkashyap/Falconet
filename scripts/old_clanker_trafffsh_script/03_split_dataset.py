import argparse
import os
import random
import shutil


def split(frames_dir, labels_dir, out_dir, val_ratio, seed=42):
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png")))
    random.seed(seed)
    random.shuffle(frame_files)

    n_val = max(1, int(len(frame_files) * val_ratio))
    val_files = set(frame_files[:n_val])

    for split_name in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split_name), exist_ok=True)

    for fname in frame_files:
        stem = os.path.splitext(fname)[0]
        split_name = "val" if fname in val_files else "train"

        src_img = os.path.join(frames_dir, fname)
        dst_img = os.path.join(out_dir, "images", split_name, fname)
        shutil.copy2(src_img, dst_img)

        src_lbl = os.path.join(labels_dir, f"{stem}.txt")
        dst_lbl = os.path.join(out_dir, "labels", split_name, f"{stem}.txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()  # empty label = frame with no cars

    print(f"Train: {len(frame_files) - n_val} | Val: {n_val}")
    print(f"Dataset ready at: {out_dir}/images and {out_dir}/labels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="dataset/raw_frames")
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    args = ap.parse_args()
    split(args.frames, args.labels, args.out, args.val_ratio)
