import os
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add root folder to sys.path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from pydantic import BaseModel
from typing import Optional
from app.inference import MangoPredictor, decode_image, DEVICE
from app.packing import CONTAINER_PRESETS, estimate_count

class CountRequest(BaseModel):
    mango_h: float
    mango_w: float
    mango_t: float
    container_id: str
    custom_l: Optional[float] = None
    custom_w: Optional[float] = None
    custom_h: Optional[float] = None


# Instantiate the predictor wrapper
predictor = MangoPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models into memory on application startup
    predictor.load_models()
    yield
    # Shutdown logic (optional)
    print("Shutting down FastAPI server, releasing resources.")

app = FastAPI(
    title="Mango Morphology & Ripeness System",
    description="FastAPI application estimating morphological and biological properties of mangoes using Monocular Depth + YOLOv11.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local testing/development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """
    Serves the dashboard user interface.
    """
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(
            status_code=404, 
            detail="Dashboard frontend template not found at templates/index.html"
        )
    
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

@app.get("/containers")
async def get_containers():
    """
    Returns list of preset containers.
    """
    return CONTAINER_PRESETS

@app.post("/count-in-container")
async def count_in_container(req: CountRequest):
    """
    Performs packing simulation and returns how many mangoes fit and their coordinates.
    """
    try:
        result = estimate_count(
            mango_h=req.mango_h,
            mango_w=req.mango_w,
            mango_t=req.mango_t,
            container_id=req.container_id,
            custom_l=req.custom_l,
            custom_w=req.custom_w,
            custom_h=req.custom_h
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    fx: float = Form(2123.0),
    fy: float = Form(2123.0),
    cx: float = Form(1500.0),
    cy: float = Form(2000.0)
):
    """
    Accepts an uploaded image file, processes it, and returns
    estimated parameters along with visualization base64 assets.
    """
    # Verify file content type is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="File provided is not a valid image type."
        )
    
    contents = await file.read()
    img_rgb = decode_image(contents)
    if img_rgb is None:
        raise HTTPException(
            status_code=400, 
            detail="Failed to decode image content. Please upload a valid JPEG or PNG."
        )

    # Execute predictions using the loaded pipeline
    intrinsics = (fx, fy, cx, cy)
    results = predictor.predict(img_rgb, intrinsics=intrinsics)
    
    if not results.get("success"):
        raise HTTPException(
            status_code=422, 
            detail=results.get("error", "Prediction failed.")
        )
        
    return results

@app.post("/detect-aruco")
async def detect_aruco_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns whether an ArUco marker is detected
    along with its recommended size.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="File provided is not a valid image type."
        )
    
    contents = await file.read()
    img_rgb = decode_image(contents)
    if img_rgb is None:
        raise HTTPException(
            status_code=400, 
            detail="Failed to decode image content."
        )

    from app.inference import detect_aruco
    found, _, _ = detect_aruco(img_rgb)
    return {
        "aruco_detected": bool(found),
        "marker_size_cm": 5.0
    }

@app.get("/health")
async def health_check():
    """
    Standard health check endpoint.
    """
    return {
        "status": "healthy",
        "device": str(DEVICE if predictor.depth_model else "not_loaded")
    }

