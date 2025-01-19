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
import torch

class ConfigManager:
    """Manages configuration settings for the enhanced detection system"""
    
    @staticmethod
    def get_default_config() -> Dict:
        return {
            'hand_detection': {
                'static_mode': False,
                'max_hands': 2,
                'detection_confidence': 0.5,
                'tracking_confidence': 0.5
            },
            'training': {
                'samples_per_sign': 100,
                'test_size': 0.2,
                'random_state': 42
            },
            'prediction': {
                'confidence_threshold': 0.6,
                'cooldown_seconds': 2
            },
            'ui': {
                'max_history': 5,
                'overlay_opacity': 0.7
            },
            'object_detection': {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'enable_gpu': True,
                'model_size': 'yolov5s'  # Options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
            }
        }

    @staticmethod
    def load_config(config_path: str = 'config.json') -> Dict:
        """Load configuration from file or create default if not exists"""
        default_config = ConfigManager.get_default_config()
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        config[key] = {**value, **config[key]}
            return config
        except FileNotFoundError:
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config

class ObjectDetector:
    """Handles real-time object detection using YOLOv5"""
    
    def __init__(self, config: Dict):
        """Initialize the object detector"""
        try:
            self.config = config['object_detection']
            
            # Load YOLOv5 model
            print(f"Loading YOLOv5 model: {self.config['model_size']}...")
            self.model = torch.hub.load('ultralytics/yolov5', 
                                      self.config['model_size'],
                                      pretrained=True)
            
            # Configure model settings
            self.model.conf = self.config['confidence_threshold']
            self.model.iou = self.config['nms_threshold']
            
            # Use GPU if available and enabled
            if self.config['enable_gpu'] and torch.cuda.is_available():
                print("Using GPU for object detection")
                self.model.cuda()
            else:
                print("Using CPU for object detection")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Store class names
            self.class_names = self.model.names
            print(f"Model loaded with {len(self.class_names)} object classes")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ObjectDetector: {e}")
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect objects in the frame
        Returns: List of detected objects and annotated frame
        """
        try:
            # Convert frame for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform detection
            results = self.model(frame_rgb)
            
            # Process results
            detections = []
            for pred in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
                x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
                class_name = self.class_names[int(cls)]
                
                detection = {
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                }
                detections.append(detection)
                
                # Draw bounding box and label
                color = (0, 255, 0)  # Green box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Speak detected object (if it's new)
                if conf > 0.8:  # Only speak high-confidence detections
                    self.speak_detection(class_name)
            
            return detections, frame
            
        except Exception as e:
            print(f"Error in detect_objects: {e}")
            return [], frame

class SignLanguageDetector:
    """Main class for sign language and object detection"""
    
    def __init__(self):
        """Initialize the detector with all necessary components"""
        try:
            # Load configuration
            self.config = ConfigManager.load_config()
            
            # Initialize MediaPipe
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.config['hand_detection']['static_mode'],
                max_num_hands=self.config['hand_detection']['max_hands'],
                min_detection_confidence=self.config['hand_detection']['detection_confidence'],
                min_tracking_confidence=self.config['hand_detection']['tracking_confidence']
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Initialize object detector
            self.object_detector = ObjectDetector(self.config)
            
            # Initialize machine learning components
            self.classifier = None
            self.label_encoder = LabelEncoder()
            self.trained_signs = []
            
            # Initialize text-to-speech
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.speech_queue = queue.Queue()
                self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
                self.speech_thread.start()
            except Exception as e:
                print(f"Warning: Text-to-speech initialization failed: {e}")
                self.engine = None
            
            # Initialize tracking variables
            self.training_data = []
            self.current_sign = None
            self.capture_mode = False
            self.samples_needed = self.config['training']['samples_per_sign']
            self.confidence_threshold = self.config['prediction']['confidence_threshold']
            self.last_prediction = None
            self.last_prediction_time = datetime.now()
            self.translation_history = []
            self.prediction_cooldown = timedelta(seconds=self.config['prediction']['cooldown_seconds'])
            
            print("Detection system initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SignLanguageDetector: {e}")

    def detect_landmarks(self, frame) -> Tuple[List, np.ndarray]:
        """Detect hand landmarks in the given frame"""
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
                    
                    try:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    except Exception as e:
                        print(f"Warning: Failed to draw landmarks: {e}")
            
            return landmarks_list, frame
            
        except Exception as e:
            print(f"Error in detect_landmarks: {e}")
            return [], frame

    def calculate_features(self, landmarks: List) -> List[float]:
        """Calculate features from landmarks"""
        try:
            if not landmarks:
                return [0.0] * 52
            
            points = np.array(landmarks).reshape(-1, 2)
            angles = []
            finger_bases = [
                (0, 1, 2),    # Thumb
                (0, 5, 6),    # Index
                (0, 9, 10),   # Middle
                (0, 13, 14),  # Ring
                (0, 17, 18),  # Pinky
                (1, 2, 3),    # Thumb bend
                (5, 6, 7),    # Index bend
                (9, 10, 11),  # Middle bend
                (13, 14, 15), # Ring bend
                (17, 18, 19)  # Pinky bend
            ]
            
            for p1_idx, p2_idx, p3_idx in finger_bases:
                try:
                    p1, p2, p3 = points[p1_idx], points[p2_idx], points[p3_idx]
                    angle = np.degrees(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) -
                                    np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))
                    angle = np.abs(angle)
                    if angle > 180:
                        angle = 360 - angle
                    angles.append(angle)
                except Exception:
                    angles.append(0.0)
            
            features = landmarks + angles
            features = features[:52] if len(features) > 52 else features + [0.0] * (52 - len(features))
            return features
            
        except Exception as e:
            print(f"Error in calculate_features: {e}")
            return [0.0] * 52

    def predict_sign(self, landmarks: List) -> Tuple[str, float]:
        """Predict sign with confidence score"""
        try:
            if not self.classifier or not landmarks:
                return "No sign detected", 0.0
            
            features = self.calculate_features(landmarks)
            feature_names = ([f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']] + 
                           [f'angle_{i}' for i in range(10)])
            
            features_df = pd.DataFrame([features], columns=feature_names)
            prediction = self.classifier.predict(features_df)
            probabilities = self.classifier.predict_proba(features_df)
            confidence = float(np.max(probabilities))
            
            if confidence >= self.confidence_threshold:
                return self.label_encoder.inverse_transform(prediction)[0], confidence
            else:
                return "Unknown sign", confidence
                
        except Exception as e:
            print(f"Error in predict_sign: {e}")
            return "Error in prediction", 0.0

    def run_detection(self):
        """Run real-time sign language and object detection"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # FPS calculation variables
            fps = 0
            frame_count = 0
            start_time = time.time()

            print("\nEnhanced Detection System Started")
            print("Commands:")
            print("  't' - Start training a new sign")
            print("  's' - Save current training data")
            print("  'c' - Train classifier with saved data")
            print("  'o' - Toggle object detection")
            print("  'q' - Quit")

            # Detection flags
            object_detection_enabled = True
            display_fps = True

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()

                # Object Detection
                if object_detection_enabled:
                    detections, frame = self.object_detector.detect_objects(frame)

                # Sign Language Detection
                landmarks_list, processed_frame = self.detect_landmarks(frame)

                for landmarks in landmarks_list:
                    if self.capture_mode and self.current_sign:
                        # Training mode
                        if self.collect_training_sample(landmarks):
                            print(f"\nCompleted collecting samples for {self.current_sign}")
                        else:
                            remaining = self.samples_needed - len(self.training_data)
                            cv2.putText(processed_frame, 
                                      f"Training '{self.current_sign}': {remaining} samples left",
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, (0, 255, 0), 2)
                    else:
                        # Detection mode
                        predicted_sign, confidence = self.predict_sign(landmarks)
                        
                        if confidence >= self.confidence_threshold:
                            self.update_translation_history(predicted_sign)
                            
                        # Display prediction
                        cv2.putText(processed_frame,
                                  f"Sign: {predicted_sign} ({confidence:.2f})",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2)

# Display FPS
                if display_fps:
                    cv2.putText(processed_frame,
                              f"FPS: {fps:.1f}",
                              (10, processed_frame.shape[0] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 255, 0), 2)

                # Draw translation history
                processed_frame = self.draw_translation_panel(processed_frame)

                # Show frame
                cv2.imshow('Enhanced Detection System', processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    sign_name = input("\nEnter the name of the sign to train: ")
                    self.start_training(sign_name)
                elif key == ord('s'):
                    self.save_training_data()
                elif key == ord('c'):
                    if self.train_classifier():
                        print("Classifier trained successfully!")
                        print("Trained signs:", self.trained_signs)
                    else:
                        print("Training failed. Check if you have enough data.")
                elif key == ord('o'):
                    object_detection_enabled = not object_detection_enabled
                    status = "enabled" if object_detection_enabled else "disabled"
                    print(f"\nObject detection {status}")

        except Exception as e:
            print(f"Error in run_detection: {e}")

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

    def _process_speech_queue(self):
        """Process text-to-speech queue"""
        while True:
            try:
                if not self.engine:
                    time.sleep(1)
                    continue
                    
                text = self.speech_queue.get()
                if text:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.speech_queue.task_done()
                
            except Exception as e:
                print(f"Speech error: {e}")
                time.sleep(1)

    def update_translation_history(self, sign: str):
        """Update translation history with cooldown"""
        try:
            if sign in ["No sign detected", "Unknown sign", "Error in prediction"]:
                return
                
            current_time = datetime.now()
            
            # Check cooldown
            if (self.last_prediction != sign or 
                (current_time - self.last_prediction_time) > self.prediction_cooldown):
                
                # Add timestamp
                timestamp = current_time.strftime("%H:%M:%S")
                translation_entry = f"[{timestamp}] {sign}"
                
                # Update history
                self.translation_history.append(translation_entry)
                if len(self.translation_history) > self.config['ui']['max_history']:
                    self.translation_history.pop(0)
                
                # Speak the sign
                if self.engine:
                    self.speech_queue.put(sign)
                
                # Update tracking
                self.last_prediction = sign
                self.last_prediction_time = current_time
                
        except Exception as e:
            print(f"Error in update_translation_history: {e}")

    def draw_translation_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw translation history panel"""
        try:
            if len(self.translation_history) == 0:
                return frame
                
            # Create overlay
            overlay = frame.copy()
            panel_height = 30 + (len(self.translation_history) * 30)
            
            # Draw background
            cv2.rectangle(overlay, 
                         (frame.shape[1]-300, 0),
                         (frame.shape[1], panel_height),
                         (0, 0, 0), -1)
            
            # Add title
            cv2.putText(overlay, "Translation History",
                       (frame.shape[1]-290, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
            
            # Add translations
            for i, translation in enumerate(self.translation_history):
                cv2.putText(overlay, translation,
                           (frame.shape[1]-290, 55 + (i * 30)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 255, 255), 1)
            
            # Blend overlay
            return cv2.addWeighted(overlay, 
                                 self.config['ui']['overlay_opacity'],
                                 frame,
                                 1 - self.config['ui']['overlay_opacity'],
                                 0)
                
        except Exception as e:
            print(f"Error in draw_translation_panel: {e}")
            return frame

    def train_classifier(self, dataset_path: str = 'sign_language_dataset.csv') -> bool:
        """Train the classifier with the collected data"""
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Check if we have enough data
            unique_signs = df.iloc[:, -1].unique()
            if len(unique_signs) < 2:
                print("Error: Need at least 2 different signs to train classifier!")
                return False
            
            # Prepare feature names
            landmark_columns = [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']]
            angle_columns = [f'angle_{i}' for i in range(10)]
            feature_columns = landmark_columns + angle_columns
            
            # Split features and labels
            X = df.iloc[:, :-1]
            X.columns = feature_columns
            y = df.iloc[:, -1]
            
            # Encode labels
            self.label_encoder.fit(y)
            y_encoded = self.label_encoder.transform(y)
            
            # Store trained signs
            self.trained_signs = list(self.label_encoder.classes_)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state'],
                stratify=y_encoded
            )
            
            # Train classifier
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.classifier.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.trained_signs))
            
            return True
            
        except Exception as e:
            print(f"Error in train_classifier: {e}")
            return False

    def start_training(self, sign_name: str):
        """Start collecting training data for a specific sign"""
        print(f"Preparing to collect {self.samples_needed} samples for sign: {sign_name}")
        self.training_data = []
        self.current_sign = sign_name
        self.capture_mode = True

    def collect_training_sample(self, landmarks: List) -> bool:
        """Collect a single training sample from landmarks"""
        try:
            if not self.capture_mode or not self.current_sign:
                return False

            # Calculate features from landmarks
            features = self.calculate_features(landmarks)
            if not features:
                return False

            # Add sample with label
            training_sample = features + [self.current_sign]
            self.training_data.append(training_sample)

            # Check if we've collected enough samples
            if len(self.training_data) >= self.samples_needed:
                self.save_training_data()
                return True

            return False

        except Exception as e:
            print(f"Error collecting training sample: {e}")
            return False

    def save_training_data(self, output_path: str = 'sign_language_dataset.csv'):
        """Save collected training data to CSV file"""
        try:
            if not self.training_data:
                print("No training data to save.")
                return

            # Create column names
            landmark_columns = [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']]
            angle_columns = [f'angle_{i}' for i in range(10)]
            columns = landmark_columns + angle_columns + ['sign_label']

            # Convert to DataFrame
            new_data = pd.DataFrame(self.training_data, columns=columns)

            # Append to existing file if it exists
            if os.path.exists(output_path):
                try:
                    existing_df = pd.read_csv(output_path)
                    df = pd.concat([existing_df, new_data], ignore_index=True)
                except Exception as e:
                    print(f"Error reading existing file: {e}")
                    df = new_data
            else:
                df = new_data

            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"\nTraining data saved to {output_path}")
            print(f"Total samples: {len(df)}")
            print(f"Unique signs: {df['sign_label'].unique()}")

            # Reset training state
            self.training_data = []
            self.current_sign = None
            self.capture_mode = False

        except Exception as e:
            print(f"Error saving training data: {e}")

if __name__ == "__main__":
    try:
        print("Enhanced Sign Language and Object Detection System")
        print("-----------------------------------------------")
        detector = SignLanguageDetector()
        detector.run_detection()
    except Exception as e:
        print(f"Fatal error: {e}")