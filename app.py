from flask import Flask, Response, render_template, jsonify, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
from sign2 import SignLanguageDetector
import threading
import mysql.connector
from mysql.connector import Error
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'app8isCool'
socketio = SocketIO(app)

detector = None
camera = None
frame_lock = threading.Lock()
current_frame = None
detection_running = False

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'visionMaster'
}

def create_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def init_database():
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(256) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    sign_predicted VARCHAR(50),
                    confidence FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    file_path VARCHAR(255) NOT NULL,
                    sign_count INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            connection.commit()
        except Error as e:
            print(f"Error initializing database: {e}")
        finally:
            cursor.close()
            connection.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise RuntimeError("Could not open camera.")
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        return frame if success else None


def generate_frames(user_id):
    global current_frame, detector, detection_running
    last_status = None
    while detection_running:
        if camera is None:
            break
        frame = camera.get_frame()
        if frame is None:
            continue
        with frame_lock:
            landmarks_list, processed_frame = detector.detect_landmarks(frame)
            for landmarks in landmarks_list:
                if detector.capture_mode and detector.current_sign:
                    if detector.collect_training_sample(user_id, landmarks):
                        print(f"Completed collecting samples for {detector.current_sign}")
                    else:
                        remaining = detector.samples_needed - len(detector.training_data.get(user_id, []))
                        cv2.putText(processed_frame, f"Training '{detector.current_sign}': {remaining} left",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    predicted_sign, confidence = detector.predict_sign(user_id, landmarks)
                    if not detector.idle_mode and confidence >= detector.confidence_threshold:
                        detector.update_translation_history(predicted_sign)
                    cv2.putText(processed_frame, f"Sign: {predicted_sign} ({confidence:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            processed_frame = detector.draw_translation_panel(processed_frame)
            current_frame = processed_frame
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        current_status = get_status_data(user_id)
        if last_status != current_status:
            socketio.emit('status_update', current_status)
            last_status = current_status
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def get_status_data(user_id):
    if detector:
        return {
            'status': 'success',
            'training_mode': detector.capture_mode,
            'current_sign': detector.current_sign,
            'trained_signs': detector.get_trained_signs(user_id),
            'history': detector.translation_history,
            'idle_mode': detector.idle_mode  # Add idle_mode to status
        }
    return {'status': 'error', 'message': 'Detector not initialized'}

@app.route('/')
def index():
    return redirect(url_for('dashboard')) if 'user_id' in session else render_template('login_registration.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Email and password required'})
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            if user and hash_password(password) == user['password']:
                session['user_id'] = user['id']
                session['username'] = user['username']
                return jsonify({'status': 'success', 'message': 'Login successful', 'redirect': url_for('dashboard')})
            return jsonify({'status': 'error', 'message': 'Invalid credentials'})
        except Error as e:
            print(f"Database error: {e}")
            return jsonify({'status': 'error', 'message': f'Login failed: {str(e)}'})
        finally:
            cursor.close()
            connection.close()

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    if not all([username, email, password]):
        return jsonify({'status': 'error', 'message': 'All fields required'})
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            if cursor.fetchone():
                return jsonify({'status': 'error', 'message': 'Username or email exists'})
            hashed_password = hash_password(password)
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                          (username, email, hashed_password))
            connection.commit()
            return jsonify({'status': 'success', 'message': 'Registration successful'})
        except Error as e:
            return jsonify({'status': 'error', 'message': f'Registration failed: {str(e)}'})
        finally:
            cursor.close()
            connection.close()

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('index.html', username=session.get('username'))

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required'}), 401
    user_id = session['user_id']
    return Response(generate_frames(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_training', methods=['POST'])
def start_training():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required'})
    if detector:
        data = request.get_json()
        sign_name = data.get('sign_name')
        if sign_name:
            detector.start_training(sign_name, session['user_id'])
            return jsonify({'status': 'success', 'message': f'Started training for {sign_name}'})
        return jsonify({'status': 'error', 'message': 'Sign name required'})
    return jsonify({'status': 'error', 'message': 'Detector not initialized'})

@app.route('/save_training', methods=['POST'])
def save_training():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required'})
    if detector:
        success = detector.save_training_data(session['user_id'])
        return jsonify({'status': 'success' if success else 'error',
                       'message': 'Training data saved' if success else 'Failed to save training data'})
    return jsonify({'status': 'error', 'message': 'Detector not initialized'})

@app.route('/train_classifier', methods=['POST'])
def train_classifier():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required'})
    if detector:
        success = detector.train_classifier(session['user_id'])
        return jsonify({'status': 'success' if success else 'error',
                       'message': 'Classifier trained' if success else 'Training failed',
                       'trained_signs': detector.get_trained_signs(session['user_id'])})
    return jsonify({'status': 'error', 'message': 'Detector not initialized'})

@app.route('/remove_sign', methods=['POST'])
def remove_sign():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required'})
    if detector:
        data = request.get_json()
        sign_name = data.get('sign_name')
        if not sign_name:
            return jsonify({'status': 'error', 'message': 'Sign name required'})
        success = detector.remove_sign(session['user_id'], sign_name)
        return jsonify({'status': 'success' if success else 'error',
                       'message': f'Removed {sign_name}' if success else f'Failed to remove {sign_name}',
                       'trained_signs': detector.get_trained_signs(session['user_id'])})
    return jsonify({'status': 'error', 'message': 'Detector not initialized'})

@app.route('/update_sign', methods=['POST'])
def update_sign():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required'})
    if detector:
        data = request.get_json()
        old_name = data.get('old_name')
        new_name = data.get('new_name')
        if not old_name or not new_name:
            return jsonify({'status': 'error', 'message': 'Old and new sign names required'})
        success = detector.update_sign(session['user_id'], old_name, new_name)
        return jsonify({'status': 'success' if success else 'error',
                       'message': f'Updated {old_name} to {new_name}' if success else 'Update failed',
                       'trained_signs': detector.get_trained_signs(session['user_id'])})
    return jsonify({'status': 'error', 'message': 'Detector not initialized'})

@app.route('/get_status', methods=['GET'])
def get_status():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required'})
    return jsonify(get_status_data(session['user_id']))

@app.route('/toggle_idle_mode', methods=['POST'])
def toggle_idle_mode():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required'})
    if detector:
        idle_mode = detector.toggle_idle_mode()
        return jsonify({
            'status': 'success',
            'message': f"Switched to {'Idle' if idle_mode else 'Detection'} mode",
            'idle_mode': idle_mode
        })
    return jsonify({'status': 'error', 'message': 'Detector not initialized'})


@socketio.on('connect')
def handle_connect():
    if 'user_id' in session:
        print(f"WebSocket connected for user {session['user_id']}")
    else:
        emit('status_update', {'status': 'error', 'message': 'Login required'})

def start_detection():
    global detector, camera, detection_running
    try:
        detector = SignLanguageDetector()
        camera = Camera()
        detection_running = True
        print("Detection system initialized")
    except Exception as e:
        print(f"Error starting detection: {e}")
        detection_running = False

def cleanup():
    global camera, detection_running
    detection_running = False
    if camera:
        del camera
    print("Cleanup completed")

if __name__ == '__main__':
    init_database()
    try:
        start_detection()
        socketio.run(app, debug=True, host='0.0.0.0', port=4000)
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        cleanup()