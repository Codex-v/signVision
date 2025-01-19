import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QComboBox, 
                           QProgressBar, QInputDialog, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
import cv2
import numpy as np
from sign import SignLanguageDetector  # Import your existing class

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    fps_update_signal = pyqtSignal(float)
    prediction_signal = pyqtSignal(str, float)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = True
        self.prev_time = 0

    def run(self):
        cap = cv2.VideoCapture(2)
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Get landmarks and process frame
                landmarks_list, processed_frame = self.detector.detect_landmarks(frame)
                
                # Calculate FPS
                curr_time = time.time()
                if curr_time > self.prev_time:
                    fps = 1 / (curr_time - self.prev_time)
                    self.fps_update_signal.emit(fps)
                    self.prev_time = curr_time

                # Process landmarks and get predictions
                for landmarks in landmarks_list:
                    if self.detector.classifier:
                        predicted_sign, confidence = self.detector.predict_sign(landmarks)
                        if confidence >= self.detector.confidence_threshold:
                            self.prediction_signal.emit(predicted_sign, confidence)

                self.change_pixmap_signal.emit(processed_frame)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class TranslationHistoryWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
        """)
        self.layout = QVBoxLayout()
        self.title = QLabel("Translation History")
        self.title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        self.layout.addWidget(self.title)
        self.history_labels = []
        self.setLayout(self.layout)

    def update_history(self, history):
        # Clear old labels
        for label in self.history_labels:
            self.layout.removeWidget(label)
            label.deleteLater()
        self.history_labels.clear()

        # Add new labels
        for entry in history:
            label = QLabel(entry)
            label.setStyleSheet("padding: 5px; border-bottom: 1px solid #3b3b3b;")
            self.layout.addWidget(label)
            self.history_labels.append(label)

class SignLanguageGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = SignLanguageDetector()
        self.init_ui()

    def init_ui(self):
        # Set window properties
        self.setWindowTitle('Sign Language Detection System')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1b1b1b;
            }
            QPushButton {
                background-color: #3b3b3b;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4b4b4b;
            }
            QPushButton:pressed {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
            }
            QComboBox {
                background-color: #3b3b3b;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
            }
            QProgressBar {
                border: none;
                background-color: #3b3b3b;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create left panel for video and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)

        # Controls
        controls_layout = QHBoxLayout()
        
        # Training controls
        self.train_button = QPushButton('Train New Sign')
        self.train_button.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_button)

        self.save_button = QPushButton('Save Data')
        self.save_button.clicked.connect(self.save_training_data)
        controls_layout.addWidget(self.save_button)

        self.train_classifier_button = QPushButton('Train Classifier')
        self.train_classifier_button.clicked.connect(self.train_classifier)
        controls_layout.addWidget(self.train_classifier_button)

        left_layout.addLayout(controls_layout)

        # Status indicators
        status_layout = QHBoxLayout()
        
        self.fps_label = QLabel('FPS: 0.0')
        status_layout.addWidget(self.fps_label)

        self.confidence_label = QLabel('Confidence: 0.0')
        status_layout.addWidget(self.confidence_label)

        self.current_sign_label = QLabel('Sign: None')
        status_layout.addWidget(self.current_sign_label)

        left_layout.addLayout(status_layout)

        # Training progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        main_layout.addWidget(left_panel, stretch=2)

        # Right panel for translation history
        self.translation_history = TranslationHistoryWidget()
        main_layout.addWidget(self.translation_history, stretch=1)

        # Start video thread
        self.thread = VideoThread(self.detector)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.fps_update_signal.connect(self.update_fps)
        self.thread.prediction_signal.connect(self.update_prediction)
        self.thread.start()

        # Set window size and show
        self.setMinimumSize(1024, 768)
        self.show()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), 
                                               Qt.KeepAspectRatio)
        return QPixmap.fromImage(scaled_img)

    def update_fps(self, fps):
        self.fps_label.setText(f'FPS: {fps:.1f}')

    def update_prediction(self, sign, confidence):
        self.current_sign_label.setText(f'Sign: {sign}')
        self.confidence_label.setText(f'Confidence: {confidence:.2f}')
        self.detector.update_translation_history(sign)
        self.translation_history.update_history(self.detector.translation_history)

    def start_training(self):
        sign_name, ok = QInputDialog.getText(self, 'Train New Sign', 
                                           'Enter the name of the sign to train:')
        if ok and sign_name:
            self.detector.start_training(sign_name)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(self.detector.samples_needed)

    def save_training_data(self):
        self.detector.save_training_data()
        QMessageBox.information(self, 'Success', 'Training data saved successfully!')
        self.progress_bar.setVisible(False)

    def train_classifier(self):
        if self.detector.train_classifier():
            QMessageBox.information(self, 'Success', 
                                  'Classifier trained successfully!\n' + 
                                  f'Trained signs: {", ".join(self.detector.trained_signs)}')
        else:
            QMessageBox.warning(self, 'Error', 
                              'Training failed. Make sure you have at least 2 different signs in the dataset.')

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide dark theme
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(27, 27, 27))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(27, 27, 27))
    palette.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(43, 43, 43))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = SignLanguageGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()