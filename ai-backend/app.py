"""
AI Backend Service - YOLOv3 object detection using Flask
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
import json

app = Flask(__name__)

# Configuration
MODEL_PATH = '/app/models'
OUTPUT_FOLDER = '/app/outputs'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Ensure folders exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv3 model (lazy loading on first request)
yolo_model = None

def load_yolo_model():
    """Load YOLOv3 model and weights"""
    global yolo_model

    if yolo_model is not None:
        return yolo_model

    weights_path = os.path.join(MODEL_PATH, 'yolov3.weights')
    config_path = os.path.join(MODEL_PATH, 'yolov3.cfg')
    names_path = os.path.join(MODEL_PATH, 'coco.names')

    # Download model files if not present
    if not os.path.exists(weights_path):
        print("Downloading YOLOv3 weights...")
        os.system(f'cd {MODEL_PATH} && wget https://pjreddie.com/media/files/yolov3.weights')

    if not os.path.exists(config_path):
        print("Downloading YOLOv3 config...")
        os.system(f'cd {MODEL_PATH} && wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')

    if not os.path.exists(names_path):
        print("Downloading COCO names...")
        os.system(f'cd {MODEL_PATH} && wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')

    # Load network
    net = cv2.dnn.readNet(weights_path, config_path)
    # Set CPU backend if available
    if hasattr(cv2.dnn, 'DNN_BACKEND_CPU'):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CPU)
    if hasattr(cv2.dnn, 'DNN_TARGET_CPU'):
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(names_path, 'r') as f:
        classes = f.read().strip().split('\n')

    yolo_model = {
        'net': net,
        'classes': classes,
        'layer_names': net.getUnconnectedOutLayersNames()
    }

    return yolo_model

def detect_objects(image_path):
    """Perform YOLOv3 detection on image"""
    model = load_yolo_model()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image file")

    height, width, channels = image.shape

    # Prepare blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    model['net'].setInput(blob)

    # Forward pass
    outputs = model['net'].forward(model['layer_names'])

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Draw detections
    detections = []
    image_copy = image.copy()

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            class_name = model['classes'][class_id]

            x, y, w, h = box
            detections.append({
                'class': class_name,
                'confidence': float(confidence),
                'bbox': {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'x_max': x + w,
                    'y_max': y + h
                }
            })

            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(image_copy, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return detections, image_copy

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'ai-backend'}), 200

@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Object detection endpoint
    Expected: multipart/form-data with 'image' file
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save temporary image
        request_id = str(uuid.uuid4())
        temp_path = os.path.join(OUTPUT_FOLDER, f"{request_id}_temp.jpg")
        file.save(temp_path)

        # Run detection
        detections, result_image = detect_objects(temp_path)

        # Save output image
        output_image_path = os.path.join(OUTPUT_FOLDER, f"{request_id}_detected.jpg")
        cv2.imwrite(output_image_path, result_image)

        # Prepare response
        response = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'objects_detected': len(detections),
            'detections': detections,
            'output_image_path': output_image_path,
            'confidence_threshold': CONFIDENCE_THRESHOLD
        }

        # Clean up temp file
        os.remove(temp_path)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Pre-load model on startup
    load_yolo_model()
    app.run(host='0.0.0.0', port=5001, debug=False)
