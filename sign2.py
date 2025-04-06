import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pyttsx3
import queue
import threading
import time
from datetime import datetime, timedelta
import json
import mysql.connector
from mysql.connector import Error

class ConfigManager:
    """Manages configuration settings for the sign language detector"""
    @staticmethod
    def get_default_config() -> Dict:
        return {
            'hand_detection': {
                'static_mode': False,
                'max_hands': 1,
                'detection_confidence': 0.85,
                'tracking_confidence':0.85
            },
            'training': {
                'samples_per_sign': 200,
                'test_size': 0.2,
                'random_state': 42
            },
            'prediction': {
                'confidence_threshold': 0.85,
                'cooldown_seconds': 0.000002, # Still used for history, not speech
                'voting_window': 5  # New: Voting system window size
            },
            'ui': {
                'max_history': 100,
                'overlay_opacity': 0.7
            }
        }

    @staticmethod
    def load_config(config_path: str = 'config.json') -> Dict:
        default_config = ConfigManager.get_default_config()
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        config[key] = {**value, **config[key]}
            return config
        except FileNotFoundError:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config

class SignLanguageDetector:
    """Main class for sign language detection and translation with MySQL and CSV backend"""

    def __init__(self):


        try:
            self.idle_mode = False
            self.config = ConfigManager.load_config()
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.config['hand_detection']['static_mode'],
                max_num_hands=self.config['hand_detection']['max_hands'],
                min_detection_confidence=self.config['hand_detection']['detection_confidence'],
                min_tracking_confidence=self.config['hand_detection']['tracking_confidence']
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.classifier = None
            self.label_encoder = LabelEncoder()
            self.trained_signs = {}
            self.speech_engine = pyttsx3.init()
            self.speech_enabled = True
            self.speech_queue = queue.Queue()
            self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
            self.speech_thread.start()
            self.training_data = {}
            self.current_sign = None
            self.capture_mode = False
            self.samples_needed = self.config['training']['samples_per_sign']
            self.confidence_threshold = self.config['prediction']['confidence_threshold']
            self.last_prediction = None
            self.last_prediction_time = datetime.now()
            self.translation_history = []
            self.prediction_cooldown = timedelta(seconds=self.config['prediction']['cooldown_seconds'])
            self.db_config = {
                'host': 'localhost',
                'user': 'root',
                'password': 'root',
                'database': 'visionMaster'
            }
            if not os.path.exists('AppData'):
                os.makedirs('AppData')
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SignLanguageDetector: {e}")
        

        try:
            self.speech_engine = pyttsx3.init()
            # Set properties for better performance
            self.speech_engine.setProperty('rate', 150)  # Speed of speech
            self.speech_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
            # Test if engine works
            voices = self.speech_engine.getProperty('voices')
            self.speak_text("Speech engine initialized successfully.")
            self.speak_text("Speech 1")

            if voices:
                print(f"Speech engine initialized with {len(voices)} voices")
            else:
                print("Warning: No voices found in speech engine")
        except Exception as e:
            print(f"Error initializing speech engine: {e}")
            self.speech_engine = None

    def _connect_db(self):
        try:
            return mysql.connector.connect(**self.db_config)
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None

    def detect_landmarks(self, frame) -> Tuple[List, np.ndarray]:
        try:
            if frame is None:
                raise ValueError("Invalid frame input")
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            landmarks_list = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        px = lm.x * width
                        py = lm.y * height
                        nx = min(max(px / width, 0.0), 1.0)
                        ny = min(max(py / height, 0.0), 1.0)
                        landmarks.extend([nx, ny])
                    landmarks_list.append(landmarks)
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            return landmarks_list, frame
        except Exception as e:
            print(f"Error in detect_landmarks: {e}")
            return [], frame

    def calculate_features(self, landmarks: List) -> List[float]:
        try:
            if not landmarks:
                return [0.0] * 52
            points = np.array(landmarks).reshape(-1, 2)
            angles = []
            finger_bases = [
                (0, 1, 2), (0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18),
                (1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)
            ]
            for p1_idx, p2_idx, p3_idx in finger_bases:
                try:
                    p1 = points[p1_idx]
                    p2 = points[p2_idx]
                    p3 = points[p3_idx]
                    angle = np.degrees(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) -
                                      np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))
                    angle = np.abs(angle)
                    if angle > 180:
                        angle = 360 - angle
                    angles.append(angle)
                except Exception:
                    angles.append(0.0)
            features = landmarks + angles
            if len(features) > 52:
                features = features[:52]
            elif len(features) < 52:
                features.extend([0.0] * (52 - len(features)))
            return features
        except Exception as e:
            print(f"Error in calculate_features: {e}")
            return [0.0] * 52

    def _process_speech_queue(self):
        while True:
            try:
                if not self.speech_enabled:
                    time.sleep(1)
                    continue
                try:
                    text = self.speech_queue.get(timeout=1.0)
                    if text is None:
                        break
                    if text:
                        # Add error handling around the speech operations
                        try:
                            self.speech_engine.say(text)
                            self.speech_engine.runAndWait()
                        except Exception as e:
                            print(f"Error in speech synthesis: {e}")
                    self.speech_queue.task_done()
                except queue.Empty:
                    continue
            except Exception as e:
                print(f"Speech error: {e}")
                time.sleep(1)
                continue

    def speak_text(self, text: str):
        try:
            if self.speech_engine is None:
                print(f"Cannot speak: {text} (speech engine not available)")
                return
            print(f"Queuing speech: '{text}'")  # Debug print
            self.speech_queue.put(text)
        except Exception as e:
            print(f"Error in speak_text: {e}")

    def train_classifier(self, user_id: int) -> bool:
        try:
            conn = self._connect_db()
            if not conn:
                return False
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM training_data WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            if not result:
                print("No training data found for this user.")
                return False
            csv_path = result[0]
            if not os.path.exists(csv_path):
                print(f"CSV file not found at {csv_path}")
                return False
            df = pd.read_csv(csv_path)
            unique_signs = df['sign_label'].unique()
            if len(unique_signs) < 2:
                print("Error: Need at least 2 different signs to train classifier!")
                return False
            landmark_columns = [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']]
            angle_columns = [f'angle_{i}' for i in range(10)]
            feature_columns = landmark_columns + angle_columns
            X = df[feature_columns]
            y = df['sign_label']
            self.label_encoder.fit(y)
            y_encoded = self.label_encoder.transform(y)
            self.trained_signs[user_id] = list(self.label_encoder.classes_)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state'],
                stratify=y_encoded
            )
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.trained_signs[user_id]))
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error in train_classifier: {e}")
            return False

    def toggle_idle_mode(self):
        self.idle_mode = not self.idle_mode
        print(f"Switched to {'Idle' if self.idle_mode else 'Detection'} mode")
        return self.idle_mode

    def predict_sign(self, user_id: int, landmarks: List) -> Tuple[str, float]:
        """Predict sign with confidence score and log to detection_logs"""
        if self.idle_mode:
            return "Idle - Tracking Only", 0.0
        try:
            if not self.classifier or not landmarks:
                return "No sign detected", 0.0
            features = self.calculate_features(landmarks)
            feature_names = [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']] + [f'angle_{i}' for i in range(10)]
            features_df = pd.DataFrame([features], columns=feature_names)
            prediction = self.classifier.predict(features_df)
            probabilities = self.classifier.predict_proba(features_df)
            confidence = float(np.max(probabilities))
            predicted_sign = self.label_encoder.inverse_transform(prediction)[0] if confidence >= self.confidence_threshold else "Unknown sign"
            
            # Speak every time a valid sign is detected - no cooldown here
            if confidence >= self.confidence_threshold and predicted_sign not in ["No sign detected", "Unknown sign", "Error in prediction"]:
                self.speak_text(predicted_sign)
                print("called speak_text")
            print(f"Predicted sign: {predicted_sign} with confidence: {confidence:.2f}")
            
            # Log to detection_logs
            conn = self._connect_db()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO detection_logs (user_id, sign_predicted, confidence) VALUES (%s, %s, %s)",
                        (user_id, predicted_sign, confidence)
                    )
                    conn.commit()
                except Error as e:
                    print(f"Error logging detection: {e}")
                finally:
                    cursor.close()
                    conn.close()
            return predicted_sign, confidence
        except Exception as e:
            print(f"Error in predict_sign: {e}")
            return "Error in prediction", 0.0
        
    def start_training(self, sign_name: str, user_id: int):
        print(f"Preparing to collect {self.samples_needed} samples for sign: {sign_name} for user {user_id}")
        if user_id not in self.training_data:
            self.training_data[user_id] = []
        self.current_sign = sign_name
        self.capture_mode = True

    def collect_training_sample(self, user_id: int, landmarks: List) -> bool:
        try:
            if not self.capture_mode or not self.current_sign:
                return False
            features = self.calculate_features(landmarks)
            if not features:
                return False
            training_sample = features + [self.current_sign]
            if user_id not in self.training_data:
                self.training_data[user_id] = []
            self.training_data[user_id].append(training_sample)
            if len(self.training_data[user_id]) >= self.samples_needed:
                self.save_training_data(user_id)
                return True
            return False
        except Exception as e:
            print(f"Error collecting training sample: {e}")
            return False

    def save_training_data(self, user_id: int) -> bool:
        try:
            if user_id not in self.training_data or not self.training_data[user_id]:
                print("No training data to save.")
                return False
            conn = self._connect_db()
            if not conn:
                return False
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
            result = cursor.fetchone()
            if not result:
                print(f"User {user_id} not found.")
                return False
            username = result[0]
            csv_filename = f"{user_id}_{username}_training_data.csv"
            csv_path = os.path.join('AppData', csv_filename)
            landmark_columns = [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']]
            angle_columns = [f'angle_{i}' for i in range(10)]
            columns = landmark_columns + angle_columns + ['sign_label']
            new_data = pd.DataFrame(self.training_data[user_id], columns=columns)
            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path)
                    df = pd.concat([existing_df, new_data], ignore_index=True)
                except Exception as e:
                    print(f"Error reading existing file: {e}")
                    df = new_data
            else:
                df = new_data
            df.to_csv(csv_path, index=False)
            print(f"\nTraining data saved to {csv_path}")
            print(f"Total samples: {len(df)}")
            print(f"Unique signs: {df['sign_label'].unique()}")
            cursor.execute("SELECT id FROM training_data WHERE user_id = %s", (user_id,))
            if cursor.fetchone():
                cursor.execute(
                    "UPDATE training_data SET file_path = %s, sign_count = %s WHERE user_id = %s",
                    (csv_path, len(df['sign_label'].unique()), user_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO training_data (user_id, file_path, sign_count) VALUES (%s, %s, %s)",
                    (user_id, csv_path, len(df['sign_label'].unique()))
                )
            conn.commit()
            cursor.close()
            conn.close()
            self.training_data[user_id] = []
            self.current_sign = None
            self.capture_mode = False
            return True
        except Exception as e:
            print(f"Error saving training data: {e}")
            return False

    def update_translation_history(self, sign: str):
        """Update history with cooldown, but speech handled separately"""
        try:
            if sign in ["No sign detected", "Unknown sign", "Error in prediction", "Idle - Tracking Only"]:
                return
            current_time = datetime.now()
            # Only update history if cooldown elapsed or sign changed
            if (self.last_prediction != sign or 
                (current_time - self.last_prediction_time) > self.prediction_cooldown):
                timestamp = current_time.strftime("%H:%M:%S")
                translation_entry = f"[{timestamp}] {sign}"
                self.translation_history.append(translation_entry)
                self.speak_text(sign)
                if len(self.translation_history) > self.config['ui']['max_history']:
                    self.translation_history.pop(0)
                self.last_prediction = sign
                self.last_prediction_time = current_time
        except Exception as e:
            print(f"Error in update_translation_history: {e}")

    def draw_translation_panel(self, frame: np.ndarray) -> np.ndarray:
        try:
            if len(self.translation_history) == 0:
                return frame
            overlay = frame.copy()
            panel_height = 30 + (len(self.translation_history) * 30)
            cv2.rectangle(overlay, 
                         (frame.shape[1]-300, 0),
                         (frame.shape[1], panel_height),
                         (0, 0, 0), -1)
            cv2.putText(overlay, "Translation History",
                       (frame.shape[1]-290, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
            for i, translation in enumerate(self.translation_history):
                cv2.putText(overlay, translation,
                           (frame.shape[1]-290, 55 + (i * 30)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 255, 255), 1)
            return cv2.addWeighted(overlay, 
                                 self.config['ui']['overlay_opacity'],
                                 frame,
                                 1 - self.config['ui']['overlay_opacity'],
                                 0)
        except Exception as e:
            print(f"Error in draw_translation_panel: {e}")
            return frame

    def get_trained_signs(self, user_id: int) -> List[str]:
        return self.trained_signs.get(user_id, [])

    def remove_sign(self, user_id: int, sign_name: str) -> bool:
        try:
            conn = self._connect_db()
            if not conn:
                print("Database connection failed.")
                return False
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM training_data WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            if not result:
                print(f"No training data found for user {user_id}.")
                cursor.close()
                conn.close()
                return False
            csv_path = result[0]
            if not os.path.exists(csv_path):
                print(f"CSV file not found at {csv_path}")
                cursor.close()
                conn.close()
                return False
            df = pd.read_csv(csv_path)
            if sign_name not in df['sign_label'].values:
                print(f"Sign '{sign_name}' not found in training data.")
                cursor.close()
                conn.close()
                return False
            original_sign_count = len(df['sign_label'].unique())
            df = df[df['sign_label'] != sign_name]
            new_sign_count = len(df['sign_label'].unique())
            if not df.empty:
                df.to_csv(csv_path, index=False)
                print(f"Removed '{sign_name}' from {csv_path}. Remaining signs: {new_sign_count}")
            else:
                os.remove(csv_path)
                cursor.execute("DELETE FROM training_data WHERE user_id = %s", (user_id,))
                print(f"All signs removed for user {user_id}. Deleted {csv_path}.")
                new_sign_count = 0
            self.trained_signs[user_id] = list(df['sign_label'].unique()) if not df.empty else []
            cursor.execute(
                "UPDATE training_data SET sign_count = %s WHERE user_id = %s",
                (new_sign_count, user_id)
            )
            conn.commit()
            if new_sign_count > 1:
                self.train_classifier(user_id)
                print(f"Classifier retrained after removing '{sign_name}'.")
            elif new_sign_count == 0:
                self.classifier = None
                print("No signs left; classifier reset.")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error in remove_sign: {e}")
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()
            return False

    def update_sign(self, user_id: int, old_name: str, new_name: str) -> bool:
        try:
            conn = self._connect_db()
            if not conn:
                print("Database connection failed.")
                return False
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM training_data WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            if not result:
                print(f"No training data found for user {user_id}.")
                cursor.close()
                conn.close()
                return False
            csv_path = result[0]
            if not os.path.exists(csv_path):
                print(f"CSV file not found at {csv_path}")
                cursor.close()
                conn.close()
                return False
            df = pd.read_csv(csv_path)
            if old_name not in df['sign_label'].values:
                print(f"Sign '{old_name}' not found in training data.")
                cursor.close()
                conn.close()
                return False
            if new_name in df['sign_label'].values and new_name != old_name:
                print(f"Sign '{new_name}' already exists in training data.")
                cursor.close()
                conn.close()
                return False
            df.loc[df['sign_label'] == old_name, 'sign_label'] = new_name
            df.to_csv(csv_path, index=False)
            print(f"Updated '{old_name}' to '{new_name}' in {csv_path}")
            self.trained_signs[user_id] = list(df['sign_label'].unique())
            if len(self.trained_signs[user_id]) >= 2:
                self.train_classifier(user_id)
                print(f"Classifier retrained after updating '{old_name}' to '{new_name}'.")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error in update_sign: {e}")
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()
            return False

    def run_detection(self, user_id: int = 1) -> bool:
        cap = None
        try:
            conn = self._connect_db()
            if not conn:
                print("Database connection failed.")
                return False
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM training_data WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            if not result:
                print("No training data found. Please train at least two signs to start detection.")
                return False
            csv_path = result[0]
            if not os.path.exists(csv_path):
                print(f"CSV file not found at {csv_path}. Please train at least two signs.")
                return False
            df = pd.read_csv(csv_path)
            unique_signs = df['sign_label'].unique()
            if len(unique_signs) < 2:
                print("Error: Detection requires at least 2 trained signs. Please train more signs.")
                return False
            if user_id not in self.trained_signs or not self.classifier:
                if not self.train_classifier(user_id):
                    print("Failed to load classifier. Please train at least two signs.")
                    return False

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera.")
            fps = 0
            frame_count = 0
            start_time = time.time()
            print("\nSign Language Detection Started")
            print("Commands:")
            print("  't' - Start training a new sign")
            print("  's' - Save current training data")
            print("  'c' - Train classifier with saved data")
            print("  'i' - Toggle idle mode")
            print("  'q' - Quit")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                landmarks_list, processed_frame = self.detect_landmarks(frame)
                for landmarks in landmarks_list:
                    if self.capture_mode and self.current_sign:
                        if self.collect_training_sample(user_id, landmarks):
                            print(f"\nCompleted collecting samples for {self.current_sign}")
                        else:
                            remaining = self.samples_needed - len(self.training_data.get(user_id, []))
                            cv2.putText(processed_frame, 
                                       f"Training '{self.current_sign}': {remaining} samples left",
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 255, 0), 2)
                    else:
                        predicted_sign, confidence = self.predict_sign(user_id, landmarks)
                        if confidence >= self.confidence_threshold:
                            self.update_translation_history(predicted_sign)
                        cv2.putText(processed_frame,
                                   f"Sign: {predicted_sign} ({confidence:.2f})",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame,
                           f"FPS: {fps:.1f}",
                           (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
                processed_frame = self.draw_translation_panel(processed_frame)
                cv2.imshow('Sign Language Detection', processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    sign_name = input("\nEnter the name of the sign to train: ")
                    self.start_training(sign_name, user_id)
                elif key == ord('s'):
                    self.save_training_data(user_id)
                elif key == ord('c'):
                    if self.train_classifier(user_id):
                        print("Classifier trained successfully!")
                        print("Trained signs:", self.trained_signs.get(user_id, []))
                    else:
                        print("Training failed.")
                elif key == ord('i'):
                    self.toggle_idle_mode()
        except Exception as e:
            print(f"Error in run_detection: {e}")
            return False
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            if self.speech_enabled:
                self.speech_enabled = False
                self.speech_queue.put(None)
                self.speech_thread.join(timeout=1.0)
            return True

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run_detection(user_id=1)