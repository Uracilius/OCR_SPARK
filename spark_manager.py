import logging
from pyspark.sql import SparkSession
import requests
import config
import cv2
from ocr_processor import CheckProcessor  # We'll later ensure this is distributed
import findspark
from pyspark import SparkFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spark_path = config.spark_master

class SparkManager:
    def __init__(self):
        findspark.init()
        import sys
        sys.path.append(r'C:\OCR\src')  # Local path for your development environment

        self.spark = None

    def get_spark_session(self):
        if not self.spark:
            logger.info("Initializing new Spark session.")
            
            # Initialize the Spark session with appropriate configurations
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
            
            # Add the ocr_processor.py file to the Spark workers
            # If using a local file, specify the path. Use an absolute path here.
            self.spark.sparkContext.addFile("file:///C:/OCR/src/ocr_processor.py")
        else:
            logger.info("Using existing Spark session.")
        return self.spark

    def send_to_backend(self, results: list):
        """ Sends OCR results to the backend API """
        try:
            response = requests.post(config.backend_api, json=results)
            logger.info(f"Sent OCR results to backend: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Failed to send results to backend: {str(e)}")

    def process_image_on_worker(self, file_path):
        """ Process image on the worker node. Uses the `ocr_processor` file that was added """
        spark = self.get_spark_session()

        processed_image = CheckProcessor.preprocess_image(cv2.imread(file_path))
        rdd = spark.sparkContext.parallelize(processed_image).repartition(10)  # Repartition to prevent large tasks

        results = rdd.map(self.process_image_on_worker).collect()
        import sys
        sys.path.append(SparkFiles.get("ocr_processor.py"))  # Add the distributed file to sys.path
        from ocr_processor import CheckProcessor  # Now we can import CheckProcessor

        # Now you can run the OCR processing
        result = CheckProcessor.process_check(rdd)
        return result
