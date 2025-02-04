import logging
from pyspark.sql import SparkSession
from pyspark import SparkFiles
import cv2
import numpy as np
from ocr_processor import CheckProcessor
import requests
from config import backend_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkManager:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("OCRProcessing") \
            .master("spark://your-spark-cluster:7077") \
            .getOrCreate()

    def process_image(self, file_path: str, job_id: str):
        """Runs OCR on an image file using Spark and sends the result to backend"""
        try:
            logger.info(f"Processing image {file_path} for job {job_id}")

            # Read image from file
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Failed to load image from {file_path}")

            # Process the check using OCR
            result = CheckProcessor.process_check(img)
            result['id'] = job_id

            # Send result to backend
            self.send_to_backend(result)

        except Exception as e:
            logger.error(f"Spark OCR processing failed: {str(e)}")

    def send_to_backend(self, result: dict):
        """Send OCR result to backend API"""
        try:
            response = requests.post(backend_api, json=result)
            logger.info(f"Sent OCR result to backend: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Failed to send result to backend: {str(e)}")
