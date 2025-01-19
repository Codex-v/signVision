import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json
import queue
import threading
import time
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any

class ConfigManager:
    """Manages application configuration and user preferences"""
    
    DEFAULT_CONFIG = {
        'detection': {
            'confidence_threshold': 0.7,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.7,
            'max_hands': 2
        },
        'training': {
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2,
            'samples_per_sign': 100
        },
        'performance': {
            'use_gpu': True,
            'batch_preprocessing': True,
            'preprocessing_batch_size': 4,
            'max_workers': 4
        },
        'ui': {
            'show_fps': True,
            'show_confidence': True,
            'history_size': 5
        },
        'paths': {
            'models': 'models',
            'data': 'training_data',
            'logs': 'logs'
        }
    }

    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self) -> Dict:
        """Load configuration from file or create default"""
        try:
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
                return self._merge_configs(self.DEFAULT_CONFIG, user_config)
        except FileNotFoundError:
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG

    def _save_config(self, config: Dict) -> None:
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def _merge_configs(default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        merged = default.copy()
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigManager._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _setup_logging(self) -> None:
        """Configure logging"""
        os.makedirs(self.config['paths']['logs'], exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.config['paths']['logs'], 'app.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def update_config(self, updates: Dict) -> None:
        """Update configuration with new values"""
        self.config = self._merge_configs(self.config, updates)
        self._save_config(self.config)
        logging.info("Configuration updated")


class FrameProcessor:
    """Handles video frame processing and hand detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config['detection']['max_hands'],
            min_detection_confidence=config['detection']['min_detection_confidence'],
            min_tracking_confidence=config['detection']['min_tracking_confidence']
        )
        self.executor = ThreadPoolExecutor(max_workers=config['performance']['max_workers'])
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        
        # FPS calculation
        self.fps = 0
        self.fps_frames = 0
        self.fps_start = time.time()
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds

    def start_processing(self):
        """Start frame processing thread"""
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()

    def _process_frames(self):
        """Process frames from queue"""
        while True:
            try:
                frame = self.frame_queue.get()
                if frame is None:
                    break
                    
                # Process frame in parallel
                future = self.executor.submit(self._process_single_frame, frame)
                self.result_queue.put(future.result())
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Frame processing error: {e}")

    def _process_single_frame(self, frame: np.ndarray) -> Tuple[List, np.ndarray, float]:
        """Process a single frame"""
        # Update FPS calculation
        self.fps_frames += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start
        
        if elapsed_time > self.fps_update_interval:
            self.fps = self.fps_frames / elapsed_time
            self.fps_frames = 0
            self.fps_start = current_time
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(landmarks)
                
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Draw FPS counter on frame
        if self.config['ui']['show_fps']:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        return landmarks_list, frame, self.fps

    def add_frame(self, frame: np.ndarray) -> None:
        """Add frame to processing queue"""
        try:
            self.frame_queue.put(frame, timeout=1)
        except queue.Full:
            logging.warning("Frame queue full, skipping frame")

    def get_processed_frame(self) -> Optional[Tuple[List, np.ndarray]]:
        """Get processed frame from result queue"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


class SignPredictor:
    """Handles sign prediction using the trained model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._load_model()
        self.confidence_threshold = config['detection']['confidence_threshold']
        
        # Dynamic threshold adjustment
        self.threshold_history = []
        self.threshold_window = 50
        
    def _load_model(self):
        """Load the trained model"""
        model_path = os.path.join(self.config['paths']['models'], 'latest_model.h5')
        try:
            device = torch.device('cuda' if torch.cuda.is_available() and self.config['performance']['use_gpu'] else 'cpu')
            model = torch.load(model_path, map_location=device)
            model.to(device)  # Ensure model is moved to the correct device
            logging.info(f"Model loaded on device: {device}")
            return model
        except FileNotFoundError:
            logging.error("Model file not found")
            return None
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None

    def predict(self, landmarks: List) -> Tuple[str, float]:
        """Predict sign from landmarks"""
        if not self.model or not landmarks:
            return "No prediction", 0.0
            
        try:
            features = torch.tensor(landmarks).float()
            with torch.no_grad():
                output = self.model(features)
                confidence = torch.max(output).item()
                
                # Dynamic threshold adjustment
                self.threshold_history.append(confidence)
                if len(self.threshold_history) > self.threshold_window:
                    self.threshold_history.pop(0)
                    self.adjust_threshold()
                
                if confidence >= self.confidence_threshold:
                    predicted_class = torch.argmax(output).item()
                    return self.get_sign_name(predicted_class), confidence
                    
                return "Unknown", confidence
                
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return "Error", 0.0

    def adjust_threshold(self):
        """Dynamically adjust confidence threshold"""
        mean_confidence = np.mean(self.threshold_history)
        std_confidence = np.std(self.threshold_history)
        
        # Adjust threshold based on recent predictions
        new_threshold = mean_confidence - std_confidence
        self.confidence_threshold = max(0.5, min(0.9, new_threshold))


class SignLanguageUI:
    """Handles the graphical user interface"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.root = tk.Tk()
        self.root.title("Sign Language Detection")
        self.setup_ui()
        
        # FPS display
        self.fps_label = None
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5  # Update FPS display every 0.5 seconds

    def setup_ui(self):
        """Setup UI components"""
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Video frame
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack()
        
        # FPS display
        self.fps_label = ttk.Label(self.main_frame, text="FPS: 0.0")
        self.fps_label.pack(pady=5)
        
        # Controls frame
        controls = ttk.Frame(self.main_frame)
        controls.pack(fill=tk.X, pady=5)
        
        # Buttons
        ttk.Button(controls, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        
        # Settings
        settings = ttk.Frame(self.main_frame)
        settings.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings, text="Confidence Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=self.config['detection']['confidence_threshold'])
        threshold_slider = ttk.Scale(settings, from_=0.1, to=1.0, variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def update_frame(self, frame: np.ndarray, fps: float = None):
        """Update video frame in UI"""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        
        # Update FPS display if provided
        if fps is not None and time.time() - self.last_fps_update > self.fps_update_interval:
            self.fps_label.configure(text=f"FPS: {fps:.1f}")
            self.last_fps_update = time.time()

    def update_frame(self, frame: np.ndarray):
        """Update video frame in UI"""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def start_training(self):
        """Start training mode"""
        pass

    def save_model(self):
        """Save current model"""
        pass


class EnhancedSignLanguageDetector:
    """Main class coordinating all components"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.frame_processor = FrameProcessor(self.config_manager.config)
        self.sign_predictor = SignPredictor(self.config_manager.config)
        self.ui = SignLanguageUI(self.config_manager.config)
        
        self.cap = None
        self.running = False

    def start(self):
        """Start the application"""
        try:
            self.cap = cv2.VideoCapture(2)
            self.running = True
            self.frame_processor.start_processing()
            self._main_loop()
        except Exception as e:
            logging.error(f"Startup error: {e}")
        finally:
            self.cleanup()

    def _main_loop(self):
        """Main processing loop"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Add frame to processing queue
            self.frame_processor.add_frame(frame)
            
            # Get processed frame
            result = self.frame_processor.get_processed_frame()
            if result:
                landmarks_list, processed_frame, fps = result
                
                # Predict signs
                for landmarks in landmarks_list:
                    sign, confidence = self.sign_predictor.predict(landmarks)
                    if confidence > self.config_manager.config['detection']['confidence_threshold']:
                        self._update_display(processed_frame, sign, confidence)
                
                # Update UI with frame and FPS
                self.ui.update_frame(processed_frame, fps)
                
            # Update UI and handle events
            self.ui.root.update()

    def _update_display(self, frame: np.ndarray, sign: str, confidence: float):
        """Update frame with prediction results"""
        if self.config_manager.config['ui']['show_confidence']:
            cv2.putText(frame, f"{sign} ({confidence:.2f})",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0, 255, 0), 2)

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.frame_processor.executor.shutdown()


if __name__ == "__main__":
    detector = EnhancedSignLanguageDetector()
    try:
        detector.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.cleanup()