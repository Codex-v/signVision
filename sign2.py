import os
os.environ["MPLCONFIGDIR"] = "/tmp"  # Set a writable config dir to avoid style loading issues
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mysql.connector
import json
from datetime import datetime, timedelta
import subprocess
import pyttsx3

class ConfigManager:
    @staticmethod
    def get_default_config():
        return {
            'hand_detection': {'static_mode': False, 'max_hands': 2, 'detection_confidence': 0.5, 'tracking_confidence': 0.5},
            'training': {'samples_per_sign': 100, 'test_size': 0.2, 'random_state': 42},
            'prediction': {'confidence_threshold': 0.7, 'cooldown_seconds': 1.5, 'smoothing_window': 5}
        }

class SignLanguageDetector:
    def __init__(self):
        self.config = ConfigManager.get_default_config()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.config['hand_detection']['static_mode'],
            max_num_hands=self.config['hand_detection']['max_hands'],
            min_detection_confidence=self.config['hand_detection']['detection_confidence'],
            min_tracking_confidence=self.config['hand_detection']['tracking_confidence']
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.classifiers = {}
        self.label_encoders = {}
        self.training_data = {}
        self.trained_signs = {}
        self.current_sign = None
        self.capture_mode = False
        self.samples_needed = self.config['training']['samples_per_sign']
        self.confidence_threshold = self.config['prediction']['confidence_threshold']
        self.translation_history = []
        self.prediction_cooldown = timedelta(seconds=self.config['prediction']['cooldown_seconds'])
        self.last_prediction_time = datetime.now()
        self.recent_predictions = []
        self.speaker = pyttsx3.init()

    def detect_landmarks(self, frame):
        if frame is None:
            return [], frame
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [min(max(lm.x, 0.0), 1.0) for lm in hand_landmarks.landmark] + \
                            [min(max(lm.y, 0.0), 1.0) for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                            self.mp_drawing_styles.get_default_hand_connections_style())
        return landmarks_list, frame

    def calculate_features(self, landmarks):
        if not landmarks:
            return [0.0] * 52
        points = np.array(landmarks).reshape(-1, 2)
        angles = []
        finger_bases = [(0, 1, 2), (0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18),
                        (1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]
        for p1_idx, p2_idx, p3_idx in finger_bases:
            p1, p2, p3 = points[p1_idx], points[p2_idx], points[p3_idx]
            angle = np.degrees(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) -
                               np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))
            angles.append(np.abs(angle) % 180)
        return landmarks + angles

    def predict_sign(self, user_id, landmarks):
        if not user_id or user_id not in self.classifiers or not landmarks:
            return "No sign detected", 0.0
        features = self.calculate_features(landmarks)
        feature_names = [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']] + [f'angle_{i}' for i in range(10)]
        features_df = pd.DataFrame([features], columns=feature_names)
        probs = self.classifiers[user_id].predict_proba(features_df)[0]
        max_prob = float(np.max(probs))
        pred_class = np.argmax(probs)
        pred_sign = self.label_encoders[user_id].inverse_transform([pred_class])[0]
        confidence = max_prob * (1 - np.std(probs))
        self.recent_predictions.append((pred_sign, confidence))
        if len(self.recent_predictions) > self.config['prediction']['smoothing_window']:
            self.recent_predictions.pop(0)
        if len(self.recent_predictions) >= 3 and all(p[0] == pred_sign for p in self.recent_predictions[-3:]):
            return pred_sign, confidence
        return "Unstable prediction", 0.0

    def start_training(self, sign_name, user_id):
        if user_id not in self.training_data:
            self.training_data[user_id] = []
        self.current_sign = sign_name
        self.capture_mode = True
        print(f"Started training '{sign_name}' for user {user_id}")

    def collect_training_sample(self, user_id, landmarks):
        if not self.capture_mode or not self.current_sign or not user_id:
            return False
        features = self.calculate_features(landmarks)
        self.training_data[user_id].append(features + [self.current_sign])
        return len(self.training_data[user_id]) >= self.samples_needed

    def save_training_data(self, user_id):
        if user_id not in self.training_data or not self.training_data[user_id]:
            return False
        connection = mysql.connector.connect(host='localhost', user='root', password='root', database='visionMaster')
        try:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO signs (user_id, name, training_data) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE training_data = %s",
                           (user_id, self.current_sign, json.dumps(self.training_data[user_id]), json.dumps(self.training_data[user_id])))
            connection.commit()
            self.training_data[user_id] = []
            self.current_sign = None
            self.capture_mode = False
            return True
        except mysql.connector.Error as e:
            print(f"Error saving training data: {e}")
            return False
        finally:
            cursor.close()
            connection.close()

    def train_classifier(self, user_id):
        connection = mysql.connector.connect(host='localhost', user='root', password='root', database='visionMaster')
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT name, training_data FROM signs WHERE user_id = %s", (user_id,))
            signs_data = cursor.fetchall()
            if len(signs_data) < 2:
                return False
            all_data = []
            for name, data in signs_data:
                all_data.extend(json.loads(data))
            df = pd.DataFrame(all_data, columns=[f'{axis}_{i}' for i in range(21) for axis in ['x', 'y']] +
                              [f'angle_{i}' for i in range(10)] + ['sign_label'])
            X = df.iloc[:, :-1]
            y = df['sign_label']
            self.label_encoders[user_id] = LabelEncoder()
            y_encoded = self.label_encoders[user_id].fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            self.classifiers[user_id] = RandomForestClassifier(n_estimators=200, max_depth=20)
            self.classifiers[user_id].fit(X_train, y_train)
            print(classification_report(y_test, self.classifiers[user_id].predict(X_test),
                                        target_names=self.label_encoders[user_id].classes_))
            self.trained_signs[user_id] = list(self.label_encoders[user_id].classes_)
            return True
        except mysql.connector.Error as e:
            print(f"Error training classifier: {e}")
            return False
        finally:
            cursor.close()
            connection.close()

    def remove_sign(self, user_id, sign_name):
        if user_id not in self.trained_signs or sign_name not in self.trained_signs[user_id]:
            return False
        connection = mysql.connector.connect(host='localhost', user='root', password='root', database='visionMaster')
        try:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM signs WHERE user_id = %s AND name = %s", (user_id, sign_name))
            connection.commit()
            self.trained_signs[user_id].remove(sign_name)
            if self.trained_signs[user_id]:
                self.train_classifier(user_id)
            return True
        except mysql.connector.Error as e:
            print(f"Error removing sign: {e}")
            return False
        finally:
            cursor.close()
            connection.close()

    def update_sign(self, user_id, old_name, new_name):
        if user_id not in self.trained_signs or old_name not in self.trained_signs[user_id]:
            return False
        connection = mysql.connector.connect(host='localhost', user='root', password='root', database='visionMaster')
        try:
            cursor = connection.cursor()
            cursor.execute("UPDATE signs SET name = %s WHERE user_id = %s AND name = %s", (new_name, user_id, old_name))
            connection.commit()
            self.trained_signs[user_id] = [new_name if s == old_name else s for s in self.trained_signs[user_id]]
            self.train_classifier(user_id)
            return True
        except mysql.connector.Error as e:
            print(f"Error updating sign: {e}")
            return False
        finally:
            cursor.close()
            connection.close()

    def get_trained_signs(self, user_id):
        return self.trained_signs.get(user_id, [])

    def update_translation_history(self, sign):
        if sign not in ["No sign detected", "Unstable prediction"] and \
           (datetime.now() - self.last_prediction_time) > self.prediction_cooldown:
            self.translation_history.append(f"[{datetime.now().strftime('%H:%M:%S')}] {sign}")
            if len(self.translation_history) > 5:
                self.translation_history.pop(0)
            self.speaker.say(sign)
            self.speaker.runAndWait()
            try:
                subprocess.Popen(['say', sign])
            except:
                print("macOS 'say' command failed; using pyttsx3 only")
            self.last_prediction_time = datetime.now()

    def draw_translation_panel(self, frame):
        if not self.translation_history:
            return frame
        overlay = frame.copy()
        panel_height = 30 + (len(self.translation_history) * 30)
        cv2.rectangle(overlay, (frame.shape[1]-300, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
        cv2.putText(overlay, "Translation History", (frame.shape[1]-290, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, entry in enumerate(self.translation_history):
            cv2.putText(overlay, entry, (frame.shape[1]-290, 55 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)