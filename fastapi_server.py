import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import os
from spark_manager import SparkManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
spark_manager = SparkManager()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/processImage")
async def process_image(file: UploadFile = File(...), job_id: str = Form(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file locally
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        logger.info(f"Received image {file.filename} for job {job_id}")

        # Submit OCR job to Spark
        spark_manager.process_image(file_path, job_id)

        return {"status": "processing", "id": job_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
