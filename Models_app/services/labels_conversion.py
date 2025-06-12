import cv2
import supervision as sv

def convert_label_to_prediction(img_path, label_path, class_names):
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    predictions = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.split()))
            class_id = int(parts[0])
            center_x_norm, center_y_norm, width_norm, height_norm = parts[1:]

            # Denormalize the coordinates
            x = center_x_norm * w
            y = center_y_norm * h
            width = width_norm * w
            height = height_norm * h
            predictions.append({
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "confidence": 1,  
                "class": class_names[class_id],
                "class_id": class_id,
                "image_path": img_path,
            })

    return {
        "predictions": predictions,
        "image": {
            "width": w,
            "height": h
        }
    }

node_class_names = ["Female", "Male", "Miscarriage", "Unknown"]
edge_class_names = ["Dz", "Horizontal_edge", "Mz", "Vertical_edge"]
symbol_class_names = ["Adopted_in", "Adopted_out", "Carrier", "Deceased", "Divorce", "Patient"]
gen_class_names = ["Generation"]
disease_class_names = {
  "CIRCLE_BOTTOM_HALF_FILLED": "bottom-half-filled female",
  "CIRCLE_CHECKERED": "checkered female",
  "CIRCLE_CROSS_FILLED": "cross-filled female",
  "CIRCLE_DIAGONAL_CHECKERED": "diagonal-checkered female",
  "CIRCLE_DIAGONAL_STROKES": "diagonal-strokes female",
  "CIRCLE_FILLED": "filled female",
  "CIRCLE_HORIZONTAL_STROKES": "horizontal-strokes female",
  "CIRCLE_LEFT_HALF_FILLED": "left-half-filled female",
  "CIRCLE_RIGHT_HALF_FILLED": "right-half-filled female",
  "CIRCLE_TOP_HALF_FILLED": "top-half-filled female",
  "CIRCLE_TOP_HALF_STROKES": "top-half-strokes female",
  "CIRCLE_TOP_LEFT_QUARTER_FILLED": "top-left-quarter-filled female",
  "CIRCLE_TOP_RIGHT_QUARTER_FILLED": "top-right-quarter-filled female",
  "CIRCLE_VERTICAL_STROKES": "vertical-strokes female",
  "SQUARE_BOTTOM_HALF_FILLED": "bottom-half-filled",
  "SQUARE_CHECKERED": "checkered",
  "SQUARE_CROSS_FILLED": "cross-filled",
  "SQUARE_DIAGONAL_CHECKERED": "diagonal-checkered",
  "SQUARE_DIAGONAL_STROKES": "diagonal-strokes",
  "SQUARE_FILLED": "filled",
  "SQUARE_HORIZONTAL_STROKES": "horizontal-strokes",
  "SQUARE_LEFT_FILLED": "left-half-filled",
  "SQUARE_RIGHT_FILLED": "right-half-filled",
  "SQUARE_TOP_HALF_FILLED": "top-half-filled",
  "SQUARE_TOP_HALF_STROKES": "top-half-strokes",
  "SQUARE_TOP_LEFT_QUARTER_FILLED": "top-left-quarter-filled",
  "SQUARE_TOP_RIGHT_QUARTER_FILLED": "top-right-quarter-filled",
  "SQUARE_VERTICAL_STROKES": "vertical-strokes"
}


def detections_to_predictions(detections, image):
    predictions = []
    for i, box in enumerate(detections.xyxy):
        x_min, y_min, x_max, y_max = box
        width = x_max- x_min
        height = y_max-y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        
        prediction = {
            "x": float(x_center),
            "y": float(y_center),
            "height": float(height),
            "width": float(width),
            "confidence": float(detections.confidence[i]),
            "class": detections.data["class_name"][i],
            "class_id": int(detections.class_id[i]),
        }
        
        predictions.append(prediction)
    
    image_height, image_width, _ = image.shape
    
    output_json = {
        "predictions": predictions,
        "image": {
            "width": image_width,
            "height": image_height
        }
    }
    return output_json
