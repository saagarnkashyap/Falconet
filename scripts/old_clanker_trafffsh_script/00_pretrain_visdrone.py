import argparse
from ultralytics import YOLO


def train(epochs, imgsz, batch, device, workers, cache):
    model = YOLO("yolo26n.pt")

    model.train(
        data="VisDrone.yaml",   # auto-downloads on first run, no setup needed
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,        # dataloader worker processes — lower this if your machine chokes
        cache=cache,            # False = read from disk each time (safer on limited RAM)
        patience=30,
        mosaic=1.0,             # crucial for VisDrone's tiny/dense objects
        close_mosaic=10,
        cos_lr=True,
        project="runs/birdeye",
        name="yolo26n_visdrone",
    )

    metrics = model.val()
    print(metrics)
    print("\nWeights saved to runs/birdeye/yolo26n_visdrone/weights/best.pt")
    print("Use this directly with 05_run_counting.py --classes 3 (car only),")
    print("or --classes 3 4 5 8 9 to count car/van/truck/bus/motorbike together.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    ap.add_argument("--workers", type=int, default=4, help="Dataloader workers — lower (e.g. 2) if your machine freezes")
    ap.add_argument("--cache", default=False, action="store_true", help="Cache images in RAM — only use if you have plenty of RAM")
    args = ap.parse_args()
    train(args.epochs, args.imgsz, args.batch, args.device, args.workers, args.cache)