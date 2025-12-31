from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("yolov8n.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "School Bus Detector API is running. Use /detect endpoint to send images."}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if "bus" in label.lower():
                return {"bus": True}
    
    return {"bus": False}