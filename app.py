from flask import Flask, Response, render_template, jsonify, request
import cv2
from sign2 import SignLanguageDetector
import threading
import json
import time

app = Flask(__name__)

# Global variables for sharing state between threads
detector = None
camera = None
frame_lock = threading.Lock()
current_frame = None
detection_running = False
translation_history = []



class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise RuntimeError("Could not open camera.")
        
        # Set camera properties for better performance
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return frame

def generate_frames():
    """Generate frames for video stream"""
    global current_frame, detector, detection_running
    
    while detection_running:
        if camera is None:
            break
            
        frame = camera.get_frame()
        if frame is None:
            continue

        # Process frame with detector
        with frame_lock:
            landmarks_list, processed_frame = detector.detect_landmarks(frame)
            
            for landmarks in landmarks_list:
                # If in training mode, collect samples
                if detector.capture_mode and detector.current_sign:
                    if detector.collect_training_sample(landmarks):
                        print(f"\nCompleted collecting samples for {detector.current_sign}")
                    else:
                        remaining = detector.samples_needed - len(detector.training_data)
                        cv2.putText(processed_frame, 
                                  f"Training '{detector.current_sign}': {remaining} samples left",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                else:
                    # Detection mode
                    predicted_sign, confidence = detector.predict_sign(landmarks)
                    if confidence >= detector.confidence_threshold:
                        detector.update_translation_history(predicted_sign)
                        
                    # Display prediction
                    cv2.putText(processed_frame,
                              f"Sign: {predicted_sign} ({confidence:.2f})",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 255, 0), 2)

            # Draw translation panel
            processed_frame = detector.draw_translation_panel(processed_frame)
            current_frame = processed_frame

        # Convert frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/login')
def user_login():
    """Render login page"""
    return render_template('login.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start training for a new sign"""
    global detector
    if detector:
        data = request.get_json()
        sign_name = data.get('sign_name')
        if sign_name:
            detector.start_training(sign_name)
            return jsonify({
                'status': 'success', 
                'message': f'Started training for {sign_name}'
            })
        return jsonify({
            'status': 'error', 
            'message': 'No sign name provided'
        })
    return jsonify({
        'status': 'error', 
        'message': 'Detector not initialized'
    })

@app.route('/save_training')
def save_training():
    """Save current training data"""
    global detector
    if detector:
        detector.save_training_data()
        return jsonify({
            'status': 'success',
            'message': 'Training data saved successfully'
        })
    return jsonify({
        'status': 'error',
        'message': 'Detector not initialized'
    })

@app.route('/train_classifier')
def train_classifier():
    """Train the classifier"""
    global detector
    if detector:
        success = detector.train_classifier()
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Classifier trained successfully',
                'trained_signs': detector.trained_signs
            })
        return jsonify({
            'status': 'error',
            'message': 'Training failed - ensure you have enough data'
        })
    return jsonify({
        'status': 'error',
        'message': 'Detector not initialized'
    })

@app.route('/get_status')
def get_status():
    """Get current system status"""
    global detector
    if detector:
        return jsonify({
            'status': 'success',
            'training_mode': detector.capture_mode,
            'current_sign': detector.current_sign,
            'trained_signs': detector.trained_signs if detector.trained_signs else [],
            'history': detector.translation_history
        })
    return jsonify({
        'status': 'error',
        'message': 'Detector not initialized'
    })

def start_detection():
    """Initialize and start detection"""
    global detector, camera, detection_running
    try:
        detector = SignLanguageDetector()
        camera = Camera()
        detection_running = True
        print("Detection system initialized successfully")
    except Exception as e:
        print(f"Error starting detection: {e}")
        detection_running = False

def cleanup():
    """Cleanup resources"""
    global camera, detection_running
    detection_running = False
    if camera:
        del camera
    print("Cleanup completed")

if __name__ == '__main__':
    try:
        start_detection()
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=4000)
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        cleanup()