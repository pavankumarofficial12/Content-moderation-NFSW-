
Image Validation code: 

from fastapi import FastAPI, File, UploadFile, HTTPException
from nudenet import NudeDetector
from ultralytics import YOLO
import logging
import os
from typing import Dict, Any, List
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Content Moderation API",
    description="AI-based Image Moderation",
    version="3.1.0"
)

# Define constants for result types
RESULT_ALLOWED = "Allowed"
RESULT_WARNING = "Warning"
RESULT_REJECTED = "Rejected"
RESULT_ERROR = "Error"

# Define constants for confidence levels
CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"

# Initialize models globally
try:
    logger.info("Loading NudeNet detector...")
    detector = NudeDetector()
    
    logger.info("Loading YOLOv8 model...")
    model = YOLO("yolov8x.pt")  
    
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise

def analyze_image_content(image_path: str) -> float:
    """Analyze image content using AI models."""
    try:
        results = model(image_path, conf=0.3)

        detected_objects = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                detection = {
                    "class_id": class_id,
                    "name": result.names[class_id],
                    "confidence": confidence,
                }
                detected_objects.append(detection)

        risk_score = calculate_risk_score(detected_objects)
        return risk_score
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        return 0.0

def calculate_risk_score(detections: List[Dict[str, Any]]) -> float:
    """Calculate risk score."""
    try:
        if not detections:
            return 0.0

        confidence_scores = [d['confidence'] for d in detections]
        max_conf = max(confidence_scores)
        avg_conf = sum(confidence_scores) / len(confidence_scores)

        risk_score = (max_conf * 0.5 + avg_conf * 0.5)
        return min(1.0, risk_score)  
    except Exception as e:
        logger.error(f"Risk score calculation failed: {str(e)}")
        return 0.0

async def analyze_image(image_data: bytes, filename: str) -> Dict[str, Any]:
    """Analyze image and return moderation results."""
    try:
        temp_path = f"temp_{filename}"
        with open(temp_path, "wb") as f:
            f.write(image_data)

        risk_score = analyze_image_content(temp_path)
        os.remove(temp_path)

        confidence = (
            CONFIDENCE_HIGH if risk_score > 0.67 else
            CONFIDENCE_MEDIUM if risk_score > 0.33 else
            CONFIDENCE_LOW
        )

        # Changed threshold from 0.67 to 0.5 for rejection
        if risk_score > 0.5:
            result = RESULT_REJECTED
            message = "Content flagged as inappropriate."
        elif risk_score > 0.33:
            result = RESULT_WARNING
            message = "Content may require moderation."
        else:
            result = RESULT_ALLOWED
            message = "Content appears safe."

        return {
            "result": result,
            "risk_score": round(risk_score, 4),
            "message": message,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return {
            "result": RESULT_ERROR,
            "risk_score": 0.0,
            "message": f"Analysis failed: {str(e)}",
            "confidence": CONFIDENCE_LOW
        }

@app.post("/analyze")
async def analyze_content(file: UploadFile = File(...)):
    """API endpoint to analyze uploaded images."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        content = await file.read()
        result = await analyze_image(content, file.filename)
        return result
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(__name__ + ":app", host="0.0.0.0", port=8000, reload=True)

