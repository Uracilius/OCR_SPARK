# spark_manager.py

import findspark
findspark.init()

import os
import sys
import cv2
import logging
import requests
import config  # Your config module, e.g., with backend_api and spark_master values.
from pyspark.sql import SparkSession
from pyspark import SparkFiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# (Optional) Ensure the local directory for ocr_processor.py is in sys.path for local development.
local_ocr_processor_path = r'C:\OCR\src'
if local_ocr_processor_path not in sys.path:
    sys.path.insert(0, local_ocr_processor_path)

# ---------------------------------------------------------------------------
# Top-Level Worker Function
# ---------------------------------------------------------------------------
def worker_process(image_data):
    """
    This function is run on Spark worker nodes. It processes a piece of image data
    using the OCR processor from the distributed ocr_processor.py file.
    """
    # Add the directory of the distributed ocr_processor.py to sys.path.
    ocr_processor_file = SparkFiles.get("ocr_processor.py")
    ocr_processor_dir = os.path.dirname(ocr_processor_file)
    if ocr_processor_dir not in sys.path:
        sys.path.insert(0, ocr_processor_dir)
    
    # Import CheckProcessor from the now-accessible ocr_processor module.
    from ocr_processor import CheckProcessor

    # Process the image data using the OCR processor.
    result = CheckProcessor.process_check(image_data)
    return result

# ---------------------------------------------------------------------------
# SparkManager Class
# ---------------------------------------------------------------------------
class SparkManager:
    def __init__(self):
        findspark.init()
        self.spark = None

    def get_spark_session(self):
        if not self.spark:
            logger.info("Initializing new Spark session.")
            self.spark = SparkSession.builder \
                .appName("OCRProcessing") \
                .master("spark://192.168.1.3:7077") \
                .config("spark.executor.memory", "8g") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.cores", "4") \
                .config("spark.driver.cores", "2") \
                .config("spark.python.worker.reuse", "false") \
                .config("spark.python.worker.timeout", "600") \
                .config("spark.driver.host", "localhost") \
                .config("spark.driver.bindAddress", "localhost") \
                .config("spark.worker.port", "8888") \
                .getOrCreate()

            # Add the ocr_processor.py file so that it is available on all workers.
            ocr_processor_abs_path = r"C:\OCR\src\ocr_processor.py"  # Use an absolute path.
            logger.info(f"Adding file to Spark context: {ocr_processor_abs_path}")
            self.spark.sparkContext.addFile("file:///" + ocr_processor_abs_path)
        else:
            logger.info("Using existing Spark session.")
        return self.spark

    def send_to_backend(self, results: list):
        """
        Sends OCR results to the backend API.
        """
        try:
            response = requests.post(config.backend_api, json=results)
            logger.info(f"Sent OCR results to backend: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Failed to send results to backend: {str(e)}")

    def process_image_on_worker(self, file_path: str):
        """
        Process an image on a Spark worker. The image is read and preprocessed locally,
        then the processed image is sent to the worker node(s) for OCR processing.
        """
        spark = self.get_spark_session()

        # Read the image from file_path using OpenCV.
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Failed to read image from path: {file_path}")
            return None

        # Preprocess the image locally using the CheckProcessor.
        from ocr_processor import CheckProcessor
        processed_image = CheckProcessor.preprocess_image(image)

        # For demonstration, the entire processed image is wrapped in a list.
        # If needed, split the image into multiple segments for parallel processing.
        rdd = spark.sparkContext.parallelize([processed_image], numSlices=1)

        # Map the top-level worker_process function over the RDD.
        results = rdd.map(worker_process).collect()

        # Optionally, send the OCR results to your backend.
        self.send_to_backend(results)

        return results
