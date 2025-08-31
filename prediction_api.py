from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import logging
import os
from typing import List, Optional
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================
# FASTAPI APP SETUP
# ======================
app = FastAPI(
    title="AI Image Classifier API",
    description="FastAPI server for image classification using TensorFlow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = "trained_model.keras"
IMG_SIZE = (128, 128)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size

# ======================
# GLOBAL VARIABLES
# ======================
model = None
class_names = []

# ======================
# PYDANTIC MODELS
# ======================
class ImageBase64(BaseModel):
    image: str
    
    @validator('image')
    def validate_base64_image(cls, v):
        if not v:
            raise ValueError("Image data cannot be empty")
        
        # Check if it's a valid data URL or raw base64
        if not (v.startswith('data:image/') or len(v) > 100):
            raise ValueError("Invalid image format")
        
        return v

class PredictionResponse(BaseModel):
    success: bool
    predicted_class: str
    confidence: float
    processing_time_ms: float
    all_predictions: List[dict]
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_input_shape: Optional[str]
    classes: List[str]
    num_classes: int
    server_info: dict

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None

# ======================
# MODEL LOADING
# ======================
def load_model_and_classes():
    """Load the trained model and class names with better error handling"""
    global model, class_names
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found: {MODEL_PATH}")
            return False
        
        # Load TensorFlow model
        logger.info(f"🔄 Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Model input shape: {model.input_shape}")
        
        # Try to load class names from validation directory
        try:
            validation_set = tf.keras.utils.image_dataset_from_directory(
                'valid',
                labels="inferred",
                label_mode="categorical",
                batch_size=32,
                image_size=IMG_SIZE,
                shuffle=False
            )
            class_names = validation_set.class_names
            logger.info(f"✅ Classes loaded from 'valid' directory: {class_names}")
            
        except Exception as e:
            # Fallback to manual class names
            logger.warning(f"⚠️ Could not load classes from 'valid' directory: {e}")
            class_names = ["Class_0", "Class_1", "Class_2"]  # UPDATE THESE!
            logger.info(f"🔄 Using fallback classes: {class_names}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting AI Image Classifier API...")
    success = load_model_and_classes()
    if success:
        logger.info("✅ Server startup completed successfully!")
    else:
        logger.error("❌ Server startup failed - model not loaded!")

# ======================
# IMAGE PROCESSING FUNCTIONS
# ======================
def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image with better error handling"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        logger.info(f"📦 Decoded image: {len(image_bytes)} bytes")
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"🖼️ PIL Image: {pil_image.size}, mode: {pil_image.mode}")
        
        return pil_image
        
    except Exception as e:
        logger.error(f"❌ Base64 decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Enhanced preprocessing with better error handling and logging
    """
    try:
        start_time = time.time()
        
        # Convert to RGB (handle RGBA, grayscale, etc.)
        if pil_image.mode != "RGB":
            logger.info(f"🎨 Converting from {pil_image.mode} to RGB")
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array
        img_array = np.array(pil_image)
        logger.info(f"🔢 Original image shape: {img_array.shape}")
        
        # Resize using TensorFlow (maintains aspect ratio better)
        img_array = tf.image.resize(img_array, IMG_SIZE)
        logger.info(f"📏 Resized to: {IMG_SIZE}")
        
        # Normalize pixel values (0-255 → 0-1)
        img_array = tf.cast(img_array, tf.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        logger.info(f"📦 Final shape: {img_array.shape}")
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"⏱️ Preprocessing took: {processing_time:.2f}ms")
        
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

def make_prediction(processed_image: np.ndarray) -> dict:
    """Make prediction with detailed results"""
    try:
        start_time = time.time()
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get results
        predicted_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = class_names[predicted_idx]
        confidence = float(np.max(predictions))
        
        # Prepare all predictions
        all_predictions = []
        for i, prob in enumerate(predictions[0]):
            all_predictions.append({
                'class': class_names[i],
                'probability': float(prob)
            })
        
        # Sort by probability
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        prediction_time = (time.time() - start_time) * 1000
        logger.info(f"🧠 Prediction: {predicted_class} ({confidence:.4f}) in {prediction_time:.2f}ms")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'processing_time_ms': prediction_time
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ======================
# API ENDPOINTS
# ======================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with detailed server information"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_input_shape=str(model.input_shape) if model else None,
        classes=class_names,
        num_classes=len(class_names),
        server_info={
            "framework": "FastAPI",
            "tensorflow_version": tf.__version__,
            "model_path": MODEL_PATH,
            "image_size": IMG_SIZE,
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_from_base64(data: ImageBase64):
    """
    Predict image class from base64 encoded image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        logger.info("🚀 New base64 prediction request")
        
        # Decode base64 image
        pil_image = decode_base64_image(data.image)
        
        # Preprocess image
        processed_image = preprocess_image(pil_image)
        
        # Make prediction
        result = make_prediction(processed_image)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Total request time: {total_time:.2f}ms")
        
        return PredictionResponse(
            success=True,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            processing_time_ms=total_time,
            all_predictions=result['all_predictions'],
            message="Prediction completed successfully"
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/predict_file", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    """
    Predict image class from uploaded file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        logger.info(f"📁 File upload prediction: {file.filename}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(file_content))
        
        # Preprocess and predict
        processed_image = preprocess_image(pil_image)
        result = make_prediction(processed_image)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"✅ File prediction completed in {total_time:.2f}ms")
        
        return PredictionResponse(
            success=True,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            processing_time_ms=total_time,
            all_predictions=result['all_predictions'],
            message="File prediction completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ File prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

@app.get("/classes")
async def get_classes():
    """Get available classification classes"""
    return {
        "classes": class_names,
        "num_classes": len(class_names)
    }

@app.post("/test")
async def test_prediction():
    """Test endpoint with a random image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create random test image
        test_image = np.random.randint(0, 255, (*IMG_SIZE, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_image)
        
        # Process and predict
        processed = preprocess_image(test_pil)
        result = make_prediction(processed)
        
        return PredictionResponse(
            success=True,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            processing_time_ms=result['processing_time_ms'],
            all_predictions=result['all_predictions'],
            message="Test prediction with random image completed"
        )
        
    except Exception as e:
        logger.error(f"❌ Test prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {e}")

# ======================
# EXCEPTION HANDLERS
# ======================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }

# ======================
# MAIN APPLICATION
# ======================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 AI IMAGE CLASSIFIER - FASTAPI SERVER")
    print("="*60)
    print(f"📁 Model Path: {MODEL_PATH}")
    print(f"📏 Input Size: {IMG_SIZE}")
    print(f"🌐 Server will run at: http://localhost:8000")
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"🔍 ReDoc: http://localhost:8000/redoc")
    print("="*60)
    
    uvicorn.run(
        app, 
        host="10.121.196.41", 
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )