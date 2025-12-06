"""
UI Backend Service - Flask application for receiving image input and communicating with AI backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import uuid
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Configuration
AI_BACKEND_URL = os.getenv('AI_BACKEND_URL', 'http://ai-backend:5001')
UPLOAD_FOLDER = '/app/uploads'
OUTPUT_FOLDER = '/app/outputs'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'ui-backend'}), 200

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """
    Main endpoint to receive image and send for object detection
    Expected: multipart/form-data with 'image' file
    """
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not file.filename.lower().split('.')[-1] in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(allowed_extensions)}), 400

        # Generate unique ID and save uploaded file
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        filename = f"{request_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Send to AI backend
        with open(filepath, 'rb') as f:
            files = {'image': f}
            ai_response = requests.post(
                f'{AI_BACKEND_URL}/api/detect',
                files=files,
                timeout=30
            )

        if ai_response.status_code != 200:
            return jsonify({'error': 'AI backend error', 'details': ai_response.text}), 500

        detection_result = ai_response.json()

        # Save detection result
        result_data = {
            'request_id': request_id,
            'timestamp': timestamp,
            'original_file': filename,
            'detection_results': detection_result
        }

        json_path = os.path.join(OUTPUT_FOLDER, f"{request_id}_detection.json")
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        return jsonify({
            'success': True,
            'request_id': request_id,
            'timestamp': timestamp,
            'detection_results': detection_result,
            'output_files': {
                'json': json_path,
                'image': detection_result.get('output_image_path')
            }
        }), 200

    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Unable to connect to AI backend service'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<request_id>', methods=['GET'])
def get_status(request_id):
    """Retrieve detection results for a specific request"""
    try:
        json_path = os.path.join(OUTPUT_FOLDER, f"{request_id}_detection.json")

        if not os.path.exists(json_path):
            return jsonify({'error': 'Request ID not found'}), 404

        with open(json_path, 'r') as f:
            result = json.load(f)

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
