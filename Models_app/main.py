from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from fastapi.responses import StreamingResponse
from pathlib import Path
import io
import os
from .services.image_processor import process_image
from .services.logging_config import LOG_FILE_PATH, app_logger

app = FastAPI()

@app.get("/health")
def health_check():
    """Simple health check endpoint that returns a status OK."""
    return {"status": "ok"}

# Directory to save images
SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/image-info/")
async def extract_image_info(file: UploadFile = File(...), apply_orientation_correction: bool = Form(True)):
    """
    Extracts information from an uploaded image file by processing it through 
    the following steps:
    
    1. Read the uploaded image
    2. Convert to grayscale
    3. Correct image orientation
    4. Save the processed image
    5. Process the image with OCR
    
    Args:
        file (UploadFile): The uploaded image file
        apply_orientation_correction (bool): Whether to apply orientation correction
        
    Returns:
        dict: Dictionary containing OCR results
        
    Raises:
        HTTPException: If image processing fails
    """
    # Read the uploaded image
    image = Image.open(io.BytesIO(await file.read()))
    
    # Get original image size
    original_size = image.size
    
    # Convert the image to grayscale as a sample processing step
    processed_image = image.convert("L")
    
    # Get processed image size
    processed_size = processed_image.size
    
    # Save the processed image
    image_path = os.path.join(SAVE_DIR, file.filename)
    processed_image.save(image_path)
    
    # Call process_image function with saved image path
    ocr_result = process_image(image_path,image_id=0)  # Assuming process_image is defined elsewhere
    
    return {"model_api_response":ocr_result}

@app.get("/download-log")
async def download_log():
    """
    Endpoint to download the application log file.
    
    Returns:
        StreamingResponse: The log file as a downloadable attachment
    """
    log_file = Path(LOG_FILE_PATH)
    
    if not log_file.exists():
        return {"error": "Log file not found"}
    
    def file_stream():
        with open(log_file, "rb") as f:
            yield from f
    
    return StreamingResponse(
        file_stream(), 
        media_type="text/plain", 
        headers={"Content-Disposition": "attachment; filename=app.log"}
    )
