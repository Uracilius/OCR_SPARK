import logging
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi import BackgroundTasks
from spark_manager import SparkManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/processImage")
async def process_image(background_tasks: BackgroundTasks, file: UploadFile = File(...), job_id: str = Form(...)):
    try:
        # Generate a unique filename using UUID
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        # Save the uploaded file locally
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Add background task to process the image without waiting for completion
        background_tasks.add_task(run_spark_processing, file_path, job_id)

        return {"status": "processing", "id": job_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_spark_processing(file_path: str, job_id: str):
    try:
        # Initialize SparkManager to handle distributed processing
        spark_manager = SparkManager()
        logger.info(f"Processing image {file_path} for job {job_id}")

        # Submit OCR job to Spark for image processing
        spark_manager.process_image_on_worker(file_path)

    except Exception as e:
        logger.error(f"Error processing image {file_path} for job {job_id}: {str(e)}")
