from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(data="data.yaml", epochs=80, imgsz=640)

def main():
    results