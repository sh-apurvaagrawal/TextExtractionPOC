from..processors.text_processor import TextProcessor
from .pedigree_detector import PedigreeDetector
from .pedigree_tree import PedigreeTree
from .logging_config import app_logger
import os
detector = PedigreeDetector()
from dotenv import load_dotenv
load_dotenv()
USE_REACT_FLOW = os.getenv("USE_REACT_FLOW", "False").lower() == "true"

def process_image(image_path,image_id):
    try:
        json_data =detector.detection_pipeline(image_path)
        tree = PedigreeTree(nodes=json_data[0],text=json_data[1],image_path=image_path,image_id=image_id)
        TextProcessor(tree).process_text_data()
        return tree.nodes
    except Exception as e:
        app_logger.warning(f"Error processing image: {e}",exc_info=True)
        return tree.nodes
