import cv2
from ultralytics import YOLO
model = YOLO("./best.pt")
def main():
    patch = "example_images/test13.jpg"
    results = model.predict(source=patch, conf=0.2, save=True)
main()