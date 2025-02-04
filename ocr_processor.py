import cv2
import pytesseract
import numpy as np
from typing import Optional, Union
import logging
pytesseract.pytesseract.tesseract_cmd = r'C:\OCR\env\tesseract\tesseract.exe'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckProcessor:
    @staticmethod
    def preprocess_image(image: Union[str, np.ndarray]) -> np.ndarray:
        """Preprocess the check image for better OCR results"""
        try:
            # Load image if path is provided
            if isinstance(image, str):
                img = cv2.imread(image)
            else:
                img = image

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding instead of global
            # This often works better for varying lighting conditions and Russian text
            gray = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply slight Gaussian blur to reduce noise while preserving text edges
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            
            # Optionally increase contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
            
            return gray
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    @staticmethod
    def process_check(image: Union[str, np.ndarray]) -> Optional[dict]:
        """Process a check image and return extracted text and metadata"""
        try:
            # Preprocess the image
            processed_img = CheckProcessor.preprocess_image(image)
            
            # Configure Tesseract for Russian
            custom_config = r'--oem 3 --psm 6 -l rus'
            
            # Perform OCR with Russian language
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            # Get confidence scores and other data
            data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'extracted_text': text,
                'confidence': avg_confidence,
                'word_count': len([word for word in data['text'] if word.strip()]),
                'status': 'success',
                'language': 'rus'
            }
            
        except Exception as e:
            logger.error(f"Error processing check: {str(e)}")
            return {
                'error': str(e),
                'status': 'error'
            }