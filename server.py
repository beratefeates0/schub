from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

# FastAPI uygulaması
app = FastAPI()

# YOLO modelini yükle
model = YOLO("yolov8n.pt")

# CORS middleware ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tüm domainlerden isteğe izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Dosyayı oku ve numpy array'e dönüştür
    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Model ile tahmin yap
    results = model(img)
    
    # Otobüs var mı kontrol et
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if "bus" in label.lower():  # "bus" içeriyorsa
                return {"bus": True}
    
    return {"bus": False}
