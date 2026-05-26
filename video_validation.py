from fastapi import FastAPI, File, UploadFile, HTTPException
from nudenet import NudeDetector
from ultralytics import YOLO
import logging
import os
import cv2
import asyncio
import numpy as np
from typing import Dict, Any, List
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Content Moderation API",
    description="AI-based Image and Video Moderation",
    version="3.5.1"
)

try:
    logger.info("Loading AI models...")
    nude_detector = NudeDetector()
    yolo_model = YOLO("yolov8l.pt")
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise

def extract_frames(video_path: str) -> List[str]:
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frame_interval = max(1, fps // 2)
        frame_paths = []
        os.makedirs("frames", exist_ok=True)
        frame_count = 0

        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            success, frame = cap.read()
            if not success:
                break

            frame_path = f"frames/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += frame_interval

        cap.release()
        return frame_paths
    except Exception as e:
        logger.error(f"Frame extraction failed: {str(e)}")
        return []

async def analyze_image_content(image_path: str) -> float:
    try:
        results = await asyncio.to_thread(yolo_model, image_path, conf=0.3, device="cpu")
        detected_objects = [float(box.conf[0]) for result in results for box in result.boxes]
        
        if not detected_objects:
            return 0.2
        
        return min(1.0, max(detected_objects) * 0.6 + (sum(detected_objects) / len(detected_objects)) * 0.4)
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        return 0.2

async def analyze_video(video_data: bytes, filename: str) -> Dict[str, Any]:
    try:
        video_path = f"temp_{filename}"
        with open(video_path, "wb") as f:
            f.write(video_data)

        frame_paths = extract_frames(video_path)
        os.remove(video_path)

        if not frame_paths:
            return {"result": "error", "risk_score": 0.0, "message": "No frames extracted", "confidence": "low"}

        risk_scores = await asyncio.gather(*[analyze_image_content(frame) for frame in frame_paths])

        for frame in frame_paths:
            os.remove(frame)

        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.2
        
        confidence = "high" if avg_risk_score > 0.5 else "medium" if avg_risk_score > 0.3 else "low"
        result = "rejected" if avg_risk_score > 0.5 else "warning" if avg_risk_score > 0.3 else "allowed"
        message = "Content flagged" if result == "rejected" else "Needs review" if result == "warning" else "Safe"

        return {"result": result, "risk_score": round(avg_risk_score, 4), "message": message, "confidence": confidence}
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        return {"result": "error", "risk_score": 0.0, "message": "Analysis failed", "confidence": "low"}

@app.post("/analyze")
async def analyze_content(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()
    content = await file.read()

    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    if file_extension in ["jpg", "jpeg", "png"]:
        risk_score = await analyze_image_content(file_path)
        os.remove(file_path)
        return {"result": "processed", "risk_score": risk_score}

    elif file_extension in ["mp4", "avi", "mov", "mkv"]:
        return await analyze_video(content, file.filename)

    raise HTTPException(status_code=400, detail="Unsupported file format")

if __name__ == "__main__":
    import os
    script_name = os.path.splitext(os.path.basename(__file__))[0]  # Get script filename without .py
    uvicorn.run(f"{script_name}:app", host="0.0.0.0", port=8000, reload=True)

 


