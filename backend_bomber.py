import os
import requests

# Define the URL for processing the image
url = "http://127.0.0.1:8000/processImage"
job_id = 1  # Assuming job_id is 1 for all requests

# Path to the dataset images (TrainSet/X folder)
dataset_images_path = "C:\\Users\\admin\\.cache\\kagglehub\\datasets\\saifkhichi96\\bank-checks-signatures-segmentation-dataset\\versions\\2\\TrainSet\\X"

# List all image files (jpeg format) in the directory
image_files = [f for f in os.listdir(dataset_images_path) if f.lower().endswith(('jpeg'))]

# Loop through the images and send requests
for i, image_file in enumerate(image_files, start=1):
    image_path = os.path.join(dataset_images_path, image_file)
    
    # Open the image file for sending in the request
    with open(image_path, "rb") as f:
        files = {'file': (image_file, f, 'image/jpeg')}
        data = {'job_id': job_id}
        
        # Send the POST request
        response = requests.post(url, files=files, data=data)
        
        # Print the response status for each request
        print(f"Sent request {i}/{len(image_files)} - {image_file} - Response status: {response.status_code}")
