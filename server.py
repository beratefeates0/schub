from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI()

# 1. Load YOLO Model
# Ensure the model file is in the root directory
model_path = "yolov8n.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    print(f"WARNING: {model_path} not found!")

# 2. CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Main Page (Serve HTML)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = "scbus.html"
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return """
    <html>
        <body style="background: black; color: white; text-align: center; padding: 50px; font-family: sans-serif;">
            <h1>❌ scbus.html file not found!</h1>
            <p>Please ensure the file is in the root directory (same level as server.py).</p>
        </body>
    </html>
    """

# 4. Serve Audio File
@app.get("/beep.m4a")
async def get_audio():
    audio_path = "beep.m4a"
    if os.path.exists(audio_path):
        return FileResponse(audio_path)
    return {"error": "Audio file not found"}

# 5. Detection Algorithm (Detect Endpoint)
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Image could not be decoded", "bus": False}

        results = model(img)
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                # Detecting 'bus' class
                if "bus" in label.lower():
                    return {"bus": True}
                    
        return {"bus": False}
    except Exception as e:
        return {"error": str(e), "bus": False}

# Port configuration for Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)