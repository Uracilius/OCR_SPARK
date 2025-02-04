
import logging
from pyspark.sql import SparkSession
from pyspark import SparkFiles
import cv2
import numpy as np
from ocr_processor import CheckProcessor
import requests
from config import backend_api
import config 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spark_path = config.spark_master

class SparkManager:
    def __init__(self):
        self.spark = None

    def get_spark_session(self):
        if not self.spark:
            # Initialize SparkSession if not already created
            self.spark = SparkSession.builder \
                .appName("OCRProcessing") \
                .master("spark://10.202.15.94:7077") \
                .config("spark.executor.memory", "4g") \
                .config("spark.executor.cores", "2") \
                .config("spark.driver.memory", "2g") \
                .config("spark.driver.cores", "1")  \
                .getOrCreate()
        return self.spark

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
        finally:
            # Ensure Spark session is stopped after task completion
            if self.spark:
                logger.info("Stopping Spark session.")
                self.spark.stop()
                self.spark = None

    def send_to_backend(self, result: dict):
        """Send OCR result to backend API"""
        try:
            response = requests.post(backend_api, json=result)
            logger.info(f"Sent OCR result to backend: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Failed to send result to backend: {str(e)}")
