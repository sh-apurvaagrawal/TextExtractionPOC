import os
import cv2
from collections import Counter
from .labels_conversion import detections_to_predictions
from dotenv import load_dotenv
from .logging_config import app_logger
from ultralytics import YOLO
import supervision as sv
from supervision import Detections
import numpy as np

load_dotenv()

class PedigreeDetector:
    def __init__(self):
        load_dotenv()
        
        try:
            # Required models
            required_models = {
                "NODES_MODEL_PATH": None,
                "TEXT_MODEL_PATH": None,
            }
            
            # Fetch paths and validate
            for key in required_models:
                path = os.getenv(key)
                if not path:
                    raise ValueError(f"Missing environment variable: {key}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found at path: {path}. Please check the environment variable {key}.")
                required_models[key] = path
            
            # Fetch and validate confidence thresholds
            def get_confidence(key, default=0.5):
                value = os.getenv(key)
                if value is None:
                    app_logger.warning(f"{key} not set in .env, using default: {default}")
                    return default
                try:
                    conf = float(value)
                    if not (0.0 <= conf <= 1.0):
                        raise ValueError
                    return conf
                except ValueError:
                    raise ValueError(f"Invalid confidence value for {key}: {value}. Must be between 0.0 and 1.0.")
            
            self.nodes_model_conf = get_confidence("NODES_MODEL_CONF")
            self.text_model_conf = get_confidence("TEXT_MODEL_CONF")

            # Fetch save results flag
            self.save_results = os.getenv("SAVE_RESULTS", "False").lower() == "true"

            # Initialize models
            self.nodes_model = YOLO(required_models["NODES_MODEL_PATH"])
            self.text_model = YOLO(required_models["TEXT_MODEL_PATH"])

            app_logger.info("All models initialized successfully.")
        except Exception as e:
            app_logger.error(f"Error initializing PedigreeDetector: {str(e)}")
            raise

    
    def detect(
        self,
        image_path: str,
        model,
        conf: float,
        name: str = 'results'
    ) -> Detections:
        '''
        Generic function to detect labels using a model, apply confidence, and save predictions if required.
        '''

        result = model(image_path, conf=conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        app_logger.info({"Detection results": dict(Counter(detections.data['class_name'])),"model":f"{name}"})
        if self.save_results:
            dir_path = os.path.dirname(image_path)
            file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]
            save_dir = os.path.join(dir_path, file_name_without_extension)
            os.makedirs(save_dir,exist_ok=True)
            result.plot(save=True,filename=f"{save_dir}/{name}.png",labels=False)
        return detections
    

    def detection_pipeline(self, image_path: str):
        app_logger.info(f"Performing detection on {image_path}")
        img = cv2.imread(image_path)
        
        nodes_detections = self.detect(image_path, model=self.nodes_model, conf=self.nodes_model_conf,name ="nodes")
        text_detections = self.detect(image_path, model=self.text_model, conf=self.text_model_conf, name='text')

        
        # filtering detections
        nodes_detections= nodes_detections.with_nmm(threshold=0.1, class_agnostic=True) # handling overlapping nodes
        text_detections=text_detections.with_nmm(threshold=0.01)

        detections_list = [nodes_detections,text_detections,]
        json_data = tuple(map(lambda detections: detections_to_predictions(detections, img), detections_list))

        return json_data

if __name__ == "__main__":
    pass
