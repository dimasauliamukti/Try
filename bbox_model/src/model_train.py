from ultralytics import YOLO
from config_bbox import MODEL_DIR, DATASET_DIR


def main():
    """Train a YOLO model and export the best and last checkpoints to ONNX."""

    # Load pretrained model
    model = YOLO(MODEL_DIR / "yolo11s.pt")

    results = model.train(
        data=DATASET_DIR / "bbb" / "data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="0",
        name="good_augwed0",       # Save folder name
        project=str(MODEL_DIR),   # Root directory for saving results
    )

    # Export best and last checkpoints to ONNX
    weights_dir = results.save_dir / "weights"
    for weight_name in ["best", "last"]:
        pt_model = YOLO(weights_dir / f"{weight_name}.pt")
        pt_model.export(format="onnx")
        print(f"ONNX saved to: {weights_dir / f'{weight_name}.onnx'}")


if __name__ == "__main__":
    main()
    print("Training Finished")