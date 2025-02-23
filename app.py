from flask import Flask, Response, render_template, jsonify, request, session, redirect, url_for, flash
import cv2
from sign2 import SignLanguageDetector
import threading
import json
import time
import mysql.connector
from mysql.connector import Error
import hashlib

app = Flask(__name__)
app.secret_key = 'app8isCool'  # Change this to a secure secret key

# Global variables for sharing state between threads
detector = None
camera = None
frame_lock = threading.Lock()
current_frame = None
detection_running = False
translation_history = []

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'visionMaster'
}

def create_db_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def init_database():
    """Initialize database and create users table if it doesn't exist"""
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Create users table
            create_table_query = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(256) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_query)
            connection.commit()
            
        except Error as e:
            print(f"Error creating table: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

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


def remove_sign_from_db(sign_name):
    """Remove a sign from the database"""
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            delete_query = "DELETE FROM signs WHERE name = %s"
            cursor.execute(delete_query, (sign_name,))
            connection.commit()
            return True
        except Error as e:
            print(f"Error removing sign from database: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    return False



@app.route('/')
def index():
    """Render main page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login_registration.html')



@app.route('/remove_sign', methods=['POST'])
def remove_sign():
    """Remove a trained sign from the system"""
    global detector
    if not detector:
        return jsonify({
            'status': 'error',
            'message': 'Detector not initialized'
        })

    try:
        data = request.get_json()
        if not data or 'sign_name' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No sign name provided'
            })

        sign_name = data['sign_name']
        print(f"Attempting to remove sign: {sign_name}")  # Debug log

        if sign_name not in detector.trained_signs:
            return jsonify({
                'status': 'error',
                'message': f'Sign {sign_name} not found in trained signs'
            })

        success = detector.remove_sign(sign_name)
        if success:
            # Remove the sign from the database
            db_success = remove_sign_from_db(sign_name)
            if db_success:
                return jsonify({
                    'status': 'success',
                    'message': f'Successfully removed sign: {sign_name}',
                    'trained_signs': detector.trained_signs  # Send updated list
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to remove sign from database: {sign_name}'
                })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to remove sign: {sign_name}'
            })

    except Exception as e:
        print(f"Error in remove_sign route: {str(e)}")  # Debug log
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        })

    
@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not all([email, password]):
            return jsonify({
                'status': 'error',
                'message': 'Email and password are required'
            })
        
        connection = create_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                
                # First, get the user without checking password
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()

                
                if user:
                    # Compare the hashed password
                    input_password_hash = hash_password(password)
                    stored_password_hash = user['password']
                    
                    print(f"Input password hash: {input_password_hash}")
                    print(f"Stored password hash: {stored_password_hash}")
                    
                    if input_password_hash == stored_password_hash:
                        session['user_id'] = user['id']
                        session['username'] = user['username']
                        return jsonify({
                            'status': 'success',
                            'message': 'Login successful',
                            'redirect': url_for('dashboard')
                        })
                
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid email or password'
                })
                
            except Error as e:
                print(f"Database error: {e}")  # Add debug logging
                return jsonify({
                    'status': 'error',
                    'message': f'Login failed: {str(e)}'
                })
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()

@app.route('/register', methods=['POST'])
def register():
    """Handle user registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not all([username, email, password]):
            return jsonify({
                'status': 'error',
                'message': 'All fields are required'
            })
        
        connection = create_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # Check if username or email already exists
                cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", 
                             (username, email))
                if cursor.fetchone():
                    return jsonify({
                        'status': 'error',
                        'message': 'Username or email already exists'
                    })
                
                # Insert new user
                hashed_password = hash_password(password)
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                    (username, email, hashed_password)
                )
                connection.commit()
                
                return jsonify({
                    'status': 'success',
                    'message': 'Registration successful! Please login.'
                })
                
            except Error as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Registration failed: {str(e)}'
                })
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()
        return Response(status=200)

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Display dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('index.html', username=session.get('username'))

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
