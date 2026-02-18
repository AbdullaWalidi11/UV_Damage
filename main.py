from ultralytics import YOLO

model = YOLO("models/best.pt")
model.predict("test.jpg")
