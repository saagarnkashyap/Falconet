import argparse
from ultralytics import YOLO


def train(data_yaml, epochs, imgsz, batch, device):
    # Start from COCO-pretrained yolo26n weights (good general-purpose
    # feature extractor) then fully fine-tune on the aerial car dataset.
    model = YOLO("yolo26n.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=30,          # early stop if val mAP stalls
        mosaic=1.0,           # helps a lot with small/dense objects
        close_mosaic=10,      # disable mosaic for the last N epochs to stabilize
        degrees=0.0,          # skip rotation aug — camera angle is fixed/top-down
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,           # top-down view has no "up", flip vertically too
        cos_lr=True,
        project="runs/birdeye",
        name="yolo26n_aerial_cars",
    )

    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../configs/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    args = ap.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch, args.device)
