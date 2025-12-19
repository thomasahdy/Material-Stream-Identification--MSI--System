"""
MSI System - Flask Web Application
Material Stream Identification for waste classification
"""

import os
import cv2
import numpy as np
import joblib
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from feature_utils import extract_features_pipeline, preprocess_image, CLASS_NAMES

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = 'uploads'


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


MODELS_PATH = './models'
svm_model = None
knn_model = None

def load_models():
    """Load trained models."""
    global svm_model, knn_model
    try:
        svm_model = joblib.load(os.path.join(MODELS_PATH, 'svm_model.pkl'))
        knn_model = joblib.load(os.path.join(MODELS_PATH, 'knn_model.pkl'))
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run the notebook first to train and save models.")
        return False


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_image(img, model_name='svm', threshold=0.6):
    """Classify an image using the specified model."""
    global svm_model, knn_model
    
    if svm_model is None:
        return None, 0, "Models not loaded"
    
    # Preprocess - convert to PIL Image
    pil_img = preprocess_image(img)
    
    # Extract features
    features = extract_features_pipeline(pil_img).reshape(1, -1)
    
    # Select model
    model = svm_model if model_name == 'svm' else knn_model
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Get confidence
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = float(np.max(proba))
        if confidence < threshold:
            prediction = 'unknown'
    else:
        confidence = 1.0
    
    return str(prediction), confidence, None



camera = None

def generate_frames():
    """Generate frames for video streaming."""
    global camera
    
  
    if camera is None or not camera.isOpened():
        
        for idx in [0, 1, 2]:
            camera = cv2.VideoCapture(idx)
            if camera.isOpened():
                print(f"Camera found at index {idx}")
                break
        else:
          
            print("No camera found!")
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "NO CAMERA FOUND", (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(error_img, "Use Upload page instead", (120, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_img)
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
    
        class_name, confidence, error = classify_image(frame.copy())
        
        if class_name:
         
            label = f"{class_name.upper()}: {confidence:.2f}"
            color = (0, 255, 0) if confidence >= 0.6 else (0, 165, 255)
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, color, 3)
            cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), 
                          color, 3)
        
     
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Home page with upload form."""
    models_loaded = svm_model is not None
    return render_template('index.html', models_loaded=models_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and classification."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        
        pil_img = Image.open(file).convert('RGB')
        
        
        model_name = request.form.get('model', 'svm')
        threshold = float(request.form.get('threshold', 0.6))
        
    
        features = extract_features_pipeline(pil_img).reshape(1, -1)
        
        model = svm_model if model_name == 'svm' else knn_model
        prediction = model.predict(features)[0]
        
     
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))
            if confidence < threshold:
                prediction = 'unknown'
        else:
            confidence = 1.0
        
        return jsonify({
            'class': str(prediction),
            'confidence': round(confidence, 4),
            'model': model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/camera')
def camera_page():
    """Camera page for live classification."""
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera')
def stop_camera():
    """Stop camera feed."""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'Camera stopped'})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("MSI System - Flask Web Application")
    print("="*50 + "\n")
    
    if load_models():
        print("\nStarting server...")
        print("Open http://localhost:5000 in your browser\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n[ERROR] Could not load models!")
        print("Please run MSI_System.ipynb first to train and save the models.")
