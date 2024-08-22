from ultralytics import YOLO
if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from YAML

    # Train the model with 2 GPUs
    results = model.train(data="tip_wire_yolov8_data/data.yaml", epochs=800, imgsz=320, val=False, workers=5, batch=0.90)