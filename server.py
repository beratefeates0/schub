from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI()

# Load YOLOv8 Nano - en hafif model
model = YOLO("yolov8n.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("scbus.html"):
        return FileResponse("scbus.html")
    return "HTML file not found"

@app.get("/beep.m4a")
async def get_audio():
    if os.path.exists("beep.m4a"):
        return FileResponse("beep.m4a")
    return {"error": "Audio not found"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Hızlı okuma
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"bus": False}

        # Performans için: imgsz=320 ve conf=0.4 ayarlarıyla hızı maksimize et
        results = model.predict(img, imgsz=320, conf=0.4, verbose=False)
        
        for r in results:
            # Sadece bus (otobüs) sınıfı için kontrol (YOLO sınıf ID 5)
            if 5 in r.boxes.cls.tolist():
                return {"bus": True}
                    
        return {"bus": False}
    except:
        return {"bus": False}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)