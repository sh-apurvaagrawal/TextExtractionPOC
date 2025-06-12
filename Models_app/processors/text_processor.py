
import json
import re
from PIL import Image
from dotenv import load_dotenv
from .base_processor import BaseProcessor
from ..services.logging_config import app_logger,image_id_var
from ast import literal_eval
import time
import os
import base64
from openai import OpenAI
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv() 

vllm_server_url = os.getenv("VLLM_SERVER_URL")
vllm_model_id = os.getenv("VLLM_MODEL_ID")
openai_api_key = "EMPTY"
openai_api_base = vllm_server_url
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def encode_image(pil_image):
    """
    Encode a PIL Image object to a base64 string.

    Args:
        pil_image (PIL.Image.Image): The PIL Image object to encode.

    Returns:
        str: Base64 encoded string of the image.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can specify the format as needed
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def extract_text_from_image(image,node_id): 
    instruction = """"Extract the following structured information from the given image and return the output in valid JSON format. Ensure high accuracy in text extraction, preserving names, numbers, and medical terms correctly. The required fields are: {\"Name\": \"<Extracted Name>\", \"Age\": \"<Extracted Age>\", \"Date of Birth\": \"<Extracted Date of Birth (DD-MM-YYYY or YYYY-MM-DD format)>\", \"Disease\": \"<List of Extracted Diseases, if mentioned>\"}. Ensure that the output is well-formatted JSON with no missing or incorrect fields. If a field is not present in the image, return an empty string for that field.
    ** PLEASE RETRUN ONLY THE JSON IN THE OUTPUT.

    """
    # Getting the Base64 string
    base64_image = encode_image(image)
    try:
        response = client.chat.completions.create(
            model=vllm_model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{instruction}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=64,
            timeout=2
        )

        model_response = response.choices[0].message.content
        # app_logger.info(f"Ocr model response:{model_response}")
        return model_response
    except Exception as e:
        app_logger.error(f"Error in calling VLLM API for node {node_id}: {str(e)}")
        return ""
        
            

class TextProcessor(BaseProcessor):

    def extract_content(self,response_str):
        # Extract JSON content from LLM response by finding the first { and last }
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not match:
            return {"Name":"", "Disease":"[]","Age":"","Date of Birth":""}
        
        try:
            response = json.loads(match.group(0))  # Convert string to dictionary
        except json.JSONDecodeError:
            return {"Name":"", "Disease":"[]","Age":"","Date of Birth":""}
        
        cleaned_response = {}
        for key, value in response.items():
            # Ensure value is a string
            value = str(value).strip()
            
            # Remove values that are empty or incorrect (like non-date values in "Date of Birth")
            if value:
                cleaned_response[key] = value
            else:
                cleaned_response[key] = ""
        
        return cleaned_response

    def extract_text_from_label(self,node_id, img):
        image_id_var.set(self.tree.image_id)
        app_logger.info(f"extracting text from label for node {node_id}")
        nodes =self.tree.nodes["predictions"]
        node=nodes[node_id]
        model_response = extract_text_from_image(img,node_id)
        ocr_response = self.extract_content(model_response)
        node["display_name"] = ocr_response.get("Name")
        node["age"] = ocr_response.get("age")
        node["dob"] = ocr_response.get("Date of Birth")

    def merge_text_labels(self,text_crops):
        """
        Merge multiple text label crops into a single image (vertically stacked).
        
        Args:
            text_crops (list[PIL.Image]): List of cropped text images.
        
        Returns:
            PIL.Image: Merged image.
        """
        if not text_crops:
            return None  # No image to merge

        # Get the max width and total height needed
        max_width = max(img.width for img in text_crops)
        total_height = sum(img.height for img in text_crops)

        # Create a blank image to paste all crops
        merged_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

        # Paste images one below another
        y_offset = 0
        for img in text_crops:
            merged_image.paste(img, (0, y_offset))
            y_offset += img.height  # Move down for the next image

        return merged_image


    def group_and_merge_labels(self):
        text_data= self.tree.text
        image_path = self.tree.image_path
        img = Image.open(image_path)
        nodes = self.tree.nodes
        """
        Group text labels by their closest node, merge images, and return node_id with merged images.

        Args:
            text_data (dict): Dictionary containing text bounding boxes.
            nodes (list): List of node bounding boxes.
            img (PIL.Image): The original image from which text crops are extracted.

        Returns:
            list of tuples: [(node_id, merged_image)]
        """
        node_text_map = {i: [] for i in range(len(nodes["predictions"]))}  # Mapping: node index -> text crops

        # Assign text labels to the closest node
        for text_label in text_data["predictions"]:
            text_box = self.get_bounding_box(text_label)
            text_crop = img.crop(text_box)

            # Find the closest node
            distances = [self.euclidean_distance(self.calculate_center(text_box), self.calculate_center(self.get_bounding_box(node)))
                        for node in nodes["predictions"]]
            closest_node = distances.index(min(distances))

            # Store the crop in the corresponding node group
            node_text_map[closest_node].append(text_crop)

        # Merge images for each node and return results
        merged_results = [(node_id, self.merge_text_labels(crops)) for node_id, crops in node_text_map.items() if crops]

        return merged_results

    def process_text_data(self):
            start_time = time.perf_counter()
            node_text_map = self.group_and_merge_labels()
            results=[]
            with ThreadPoolExecutor(max_workers=8) as executor:
                    # Submit all image processing tasks to the executor
                    future_to_image = {executor.submit(self.extract_text_from_label,node_id,image): image for node_id,image in node_text_map}
                    for future in as_completed(future_to_image):
                        image = future_to_image[future]
                        data = future.result()
                        results.append(data)
            end_time = time.perf_counter()
            app_logger.info(f"Time taken to process: {end_time - start_time:.2f} seconds")
            return results



            
                
        
