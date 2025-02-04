import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import os
from spark_manager import SparkManager
from fastapi import BackgroundTasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/processImage")
async def process_image(background_tasks: BackgroundTasks, file: UploadFile = File(...), job_id: str = Form(...)):
    try:
        # Save the file locally before passing to the background task
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Save file locally on disk
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())  # This reads and saves the file content

        # Add background task to process the image without waiting for completion
        background_tasks.add_task(run_spark_processing, file_path, job_id)

        return {"status": "processing", "id": job_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_spark_processing(file_path: str, job_id: str):
    try:
        spark_manager = SparkManager()

        logger.info(f"Processing image {file_path} for job {job_id}")

        # Submit OCR job to Spark (asynchronously)
        spark_manager.process_image(file_path, job_id)

    except Exception as e:
        logger.error(f"Error processing image {file_path} for job {job_id}: {str(e)}")