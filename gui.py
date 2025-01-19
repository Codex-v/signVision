from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
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
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd

@dataclass
class HandDetectionConfig:
    static_mode: bool = False
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5

@dataclass
class TrainingConfig:
    samples_per_sign: int = 100
    test_size: float = 0.2
    random_state: int = 42

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _get_default_config(self) -> Dict:
        return {
            'hand_detection': asdict(HandDetectionConfig()),
            'training': asdict(TrainingConfig()),
            'prediction': {
                'confidence_threshold': 0.6,
                'cooldown_seconds': 2
            },
            'ui': {
                'max_history': 5,
                'overlay_opacity': 0.7
            },
            'paths': {
                'models': 'models',
                'data': 'data',
                'logs': 'logs'
            }
        }

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            config = self._get_default_config()
            self.save_config(config)
            return config

        with self.config_path.open('r') as f:
            config = json.load(f)
            return {**self._get_default_config(), **config}

    def save_config(self, config: Dict):
        with self.config_path.open('w') as f:
            json.dump(config, f, indent=4)


class SignLanguageDetector:
    def __init__(self):
        # Previous initializations remain the same
        self.mp_hands = mp.solutions.hands
        self.config = ConfigManager._load_config("config.json")

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.config['hand_detection']['static_mode'],
            max_num_hands=self.config['hand_detection']['max_hands'],
            min_detection_confidence=self.config['hand_detection']['detection_confidence'],
            min_tracking_confidence=self.config['hand_detection']['tracking_confidence']
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Training data storage
        self.training_data = []
        self.current_sign = None
        self.capture_mode = False
        self.samples_needed = 100
        self.confidence_threshold = 0.6
        
        # Classification model
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.trained_signs = []

        # Text-to-Speech initialization
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self.speech_thread.start()

        # Translation history
        self.last_prediction = None
        self.last_prediction_time = datetime.now()
        self.prediction_cooldown = timedelta(seconds=2)
        self.translation_history = []
        self.max_history = 5

    def detect_landmarks(self, frame) -> Tuple[List, np.ndarray]:
        """
        Optimized hand landmark detection
        """
        # Convert BGR to RGB efficiently
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with optimized memory usage
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        # Lists to store landmarks
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vectorized landmark extraction
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten().tolist()
                landmarks_list.append(landmarks)
                
                # Draw landmarks efficiently
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                    
        return landmarks_list, frame

    def calculate_features(self, landmarks: List) -> List[float]:
        """
        Calculate features with optimized numpy operations
        """
        if not landmarks:
            return [0] * 52  # 21 landmarks * 2 (x,y) + 10 angles
        
        # Convert to numpy array for vectorized operations
        landmarks_array = np.array(landmarks).reshape(-1, 2)
        
        # Calculate angles efficiently
        finger_bases = [
            (0, 1, 2),    # Thumb
            (3, 4, 5),    # Index finger
            (6, 7, 8),    # Index finger tip
            (9, 10, 11),  # Middle finger
            (12, 13, 14), # Middle finger tip
            (15, 16, 17), # Ring finger
            (18, 19, 20)  # Pinky
        ]
        
        angles = []
        for p1_idx, p2_idx, p3_idx in finger_bases:
            p1 = landmarks_array[p1_idx]
            p2 = landmarks_array[p2_idx]
            p3 = landmarks_array[p3_idx]
            
            # Vectorized angle calculation
            angle = np.degrees(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) -
                             np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))
            angle = np.abs(angle)
            if angle > 180:
                angle = 360 - angle
            angles.append(angle)
        
        # Combine landmarks and angles
        features = np.concatenate([landmarks_array.flatten(), angles])
        
        # Ensure exact feature length
        features = features[:52]  # Trim if too long
        if len(features) < 52:
            features = np.pad(features, (0, 52 - len(features)))  # Pad if too short
        
        return features.tolist()

    def process_frame(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """Process frame with hand detection and drawing"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                landmarks_list.append(landmarks.tolist())
                
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return landmarks_list, frame

    def predict_sign(self, landmarks: List) -> Tuple[str, float]:
        """Predict sign with confidence threshold"""
        if not self.classifier or not landmarks:
            return "No sign detected", 0.0
        
        features = self.calculate_features(landmarks)
        features_reshaped = features.reshape(1, -1)
        
        prediction = self.classifier.predict(features_reshaped)
        confidence = np.max(self.classifier.predict_proba(features_reshaped))
        
        if confidence >= self.config['prediction']['confidence_threshold']:
            return self.label_encoder.inverse_transform(prediction)[0], confidence
        return "Unknown sign", confidence

    def run_detection(self):
        """Run real-time detection with improved performance"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_tracker = deque(maxlen=30)
        
        try:
            while cap.isOpened():
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                landmarks_list, processed_frame = self.process_frame(frame)
                
                # Update FPS
                fps_tracker.append(1.0 / (time.time() - start_time))
                fps = np.mean(fps_tracker)
                
                # Handle predictions and UI updates
                self._handle_predictions(landmarks_list, processed_frame, fps)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown()
        if self.engine:
            self.engine.stop()
            
    def _handle_predictions(self, landmarks_list: List, frame: np.ndarray, fps: float):
        """Handle predictions and UI updates"""
        for landmarks in landmarks_list:
            if self.capture_mode:
                self._handle_training_mode(landmarks, frame)
            elif self.classifier:
                self._handle_classification_mode(landmarks, frame)
        
        # Add FPS and translation panel
        self._update_ui(frame, fps)
        cv2.imshow('Sign Language Detection', frame)

    def _handle_training_mode(self, landmarks: List, frame: np.ndarray):
            """Handle training mode data collection"""
            features = self.calculate_features(landmarks)
            if not features.any():
                return
                
            training_sample = np.append(features, self.current_sign)
            self.training_data.append(training_sample)
            
            remaining = self.samples_needed - len(self.training_data)
            training_text = f"Collecting {self.current_sign}: {len(self.training_data)}/{self.samples_needed}"
            cv2.putText(frame, training_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if len(self.training_data) >= self.samples_needed:
                self.save_training_data()
                self.logger.info(f"Completed collecting samples for {self.current_sign}")
                self.capture_mode = False

    def _handle_classification_mode(self, landmarks: List, frame: np.ndarray):
        """Handle real-time classification and display"""
        predicted_sign, confidence = self.predict_sign(landmarks)
        
        if confidence >= self.config['prediction']['confidence_threshold']:
            self.update_translation_history(predicted_sign)
        
        # Display prediction and confidence
        cv2.putText(frame, f"Sign: {predicted_sign}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _update_ui(self, frame: np.ndarray, fps: float):
        """Update UI elements including FPS and translation panel"""
        # Add FPS display
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update translation panel
        self._draw_translation_panel(frame)

    def _draw_translation_panel(self, frame: np.ndarray):
        """Draw translation history panel with improved visibility"""
        panel_height = 30 + (len(self.translation_history) * 30)
        overlay = frame.copy()
        
        # Panel background
        cv2.rectangle(overlay, 
                     (frame.shape[1]-300, 0),
                     (frame.shape[1], panel_height),
                     (0, 0, 0), -1)
        
        # Panel title
        cv2.putText(overlay, "Translation History",
                    (frame.shape[1]-290, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        
        # Translation entries
        for i, translation in enumerate(self.translation_history):
            cv2.putText(overlay, translation,
                       (frame.shape[1]-290, 55 + (i * 30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 1)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, self.config['ui']['overlay_opacity'],
                       frame, 1 - self.config['ui']['overlay_opacity'],
                       0, frame)

    def update_translation_history(self, sign: str):
        """Update translation history with cooldown check"""
        if sign in ["No sign detected", "Unknown sign"]:
            return
            
        current_time = datetime.now()
        cooldown = timedelta(seconds=self.config['prediction']['cooldown_seconds'])
        
        if (self.last_prediction != sign or
            (current_time - self.last_prediction_time) > cooldown):
            
            timestamp = current_time.strftime("%H:%M:%S")
            translation_entry = f"[{timestamp}] {sign}"
            
            self.translation_history.append(translation_entry)
            if len(self.translation_history) > self.config['ui']['max_history']:
                self.translation_history.pop(0)
            
            # Async TTS
            self.thread_pool.submit(self._speak_text, sign)
            
            self.last_prediction = sign
            self.last_prediction_time = current_time

    def _speak_text(self, text: str):
        """Threaded text-to-speech output"""
        try:
            if self.engine and text:
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech error: {e}")

    def save_training_data(self, output_path: Optional[str] = None):
        """Save training data with error handling and validation"""
        if not self.training_data:
            self.logger.warning("No training data to save.")
            return
        
        if output_path is None:
            output_path = Path(self.config['paths']['data']) / 'sign_language_dataset.csv'
        
        try:
            # Create column names
            landmark_columns = [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']]
            angle_columns = [f'angle_{i}' for i in range(10)]
            columns = landmark_columns + angle_columns + ['sign_label']
            
            # Convert to DataFrame
            df = pd.DataFrame(self.training_data, columns=columns)
            
            # Append if file exists
            if Path(output_path).exists():
                existing_df = pd.read_csv(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # Save with compression for large datasets
            df.to_csv(output_path, index=False, compression='gzip' if len(df) > 10000 else None)
            
            self.logger.info(f"Training data saved to {output_path}")
            self.logger.info(f"Total samples: {len(df)}")
            self.logger.info(f"Unique signs: {len(df['sign_label'].unique())}")
            
            # Reset training state
            self.training_data = []
            self.current_sign = None
            self.capture_mode = False
            
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")
            raise

    def train_classifier(self, dataset_path: Optional[str] = None):
        """Train classifier with improved parameter tuning and validation"""
        if dataset_path is None:
            dataset_path = Path(self.config['paths']['data']) / 'sign_language_dataset.csv'
            
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            self.logger.error(f"Dataset file {dataset_path} not found!")
            return False
            
        unique_signs = df.iloc[:, -1].unique()
        if len(unique_signs) < 2:
            self.logger.error("Need at least 2 different signs to train classifier!")
            return False
            
        # Prepare data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.trained_signs = list(self.label_encoder.classes_)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y_encoded
        )
        
        # Train with optimized parameters
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,  # Use all available cores
            random_state=self.config['training']['random_state']
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        report = classification_report(y_test, y_pred,
                                    target_names=self.trained_signs)
        self.logger.info("\nClassification Report:\n" + report)
        
        return True

if __name__ == "__main__":
    try:
        print("Enhanced Sign Language Detection System")
        print("Commands:")
        print("  't' - Start training a new sign")
        print("  's' - Save training data")
        print("  'c' - Train classifier")
        print("  'q' - Quit")
        
        detector = SignLanguageDetector()
        detector.run_detection()
        
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)


