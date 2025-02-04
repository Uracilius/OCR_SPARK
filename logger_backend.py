from fastapi import FastAPI, HTTPException
import logging
from typing import Any
from pydantic import BaseModel
from typing import Any

class OCRResult(BaseModel):
    input: str
    confidence: float
    word_count: int
    status: str
    language: str
    id: str

    class Config:
        # This will allow extra fields in case there are other fields in the input JSON
        extra = "allow"

# Configure logging to a file
logging.basicConfig(
    filename='ocr_backend_receiver.log',  # Log file
    level=logging.INFO,                   # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/receive")
async def receive_ocr_result(result: OCRResult):
    """
    Receive OCR result as raw JSON and log it.
    """
    try:
        # Log the received OCR result
        logger.info(f"Received OCR result: {result}")
        return {"message": "OCR result received successfully", "status": "success"}
    
    except Exception as e:
        # Log any error that occurs
        logger.error(f"Failed to process OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process OCR result")
