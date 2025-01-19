import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QPainterPath, QLinearGradient, QColor, QFont, QPen

class AnimatedText(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("PLEASE WAIT")
        self.setStyleSheet("""
            color: #9C27B0;
            font-family: Arial;
            font-size: 30px;
            font-weight: bold;
        """)
        self.setAlignment(Qt.AlignCenter)
        
        self.text_sequence = [
            "PLEASE WAIT",
            "PLEASE WAIT",
            "PANACEA OF",
            "PANACEA OF ALL",
            "PANACEA OF ALL CYBER",
            "PANACEA OF ALL CYBER ATTACKS"
        ]
        self.current_text_index = 0
        
        self.text_timer = QTimer(self)
        self.text_timer.timeout.connect(self.update_text)
        self.text_timer.start(2000)

    def update_text(self):
        self.current_text_index = (self.current_text_index + 1) % len(self.text_sequence)
        self.setText(self.text_sequence[self.current_text_index])

class AnimatedPath(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.offset = 0
        self.setMinimumHeight(300)
        
        # Create animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)
        
        # Initialize the path
        self.path = QPainterPath()
        self.init_path_data()

    def init_path_data(self):
        # Define character paths exactly as in the XAML
        def add_c():
            self.path.moveTo(34.5, -13.25)
            self.path.lineTo(34.5, -13.25)
            self.path.quadTo(36.16, -12.03, 36.16, -9.44)
            self.path.lineTo(36.16, -9.44)
            self.path.quadTo(36.16, -6.85, 35.04, -5.02)
            self.path.quadTo(33.92, -3.2, 31.74, -1.79)
            self.path.quadTo(27.07, 1.22, 18.91, 1.22)
            self.path.quadTo(10.75, 1.22, 6.3, -3.04)
            self.path.quadTo(1.86, -7.3, 1.86, -14.85)
            self.path.quadTo(1.86, -25.54, 7.42, -33.66)
            self.path.quadTo(11.65, -39.68, 19.2, -42.24)

        def add_y():
            self.path.moveTo(47.55, 0)
            self.path.lineTo(51.26, -18.82)
            self.path.quadTo(49.41, -36.16, 42.88, -39.87)
            self.path.quadTo(44.22, -41.54, 46.4, -42.46)
            self.path.quadTo(48.58, -43.39, 51.2, -43.39)
            self.path.quadTo(53.82, -43.39, 55.81, -42.46)

        def add_b():
            self.path.moveTo(78.98, -0.32)
            self.path.lineTo(87.04, -42.24)
            self.path.quadTo(94.59, -42.88, 99.46, -42.88)
            self.path.quadTo(104.32, -42.88, 107.46, -42.46)
            self.path.quadTo(110.59, -42.05, 112.7, -40.96)

        def add_e():
            self.path.moveTo(147.71, -9.86)
            self.path.quadTo(148.93, -8.32, 148.93, -5.82)
            self.path.quadTo(148.93, -2.37, 146.37, -0.54)
            self.path.quadTo(143.81, 1.28, 139.78, 1.28)

        def add_r():
            self.path.moveTo(167.94, 0)
            self.path.lineTo(154.24, 0)
            self.path.lineTo(162.05, -42.05)
            self.path.quadTo(168.38, -42.75, 177.63, -42.75)

        # Add each letter with proper spacing
        add_c()  # C
        add_y()  # Y
        add_b()  # B
        add_e()  # E
        add_r()  # R
        
        # Add PANACEA with proper spacing and scaling
        self.path.moveTo(236.61, -42.69)
        self.path.quadTo(252.29, -42.69, 252.29, -31.62)
        
        # P
        self.path.moveTo(279.17, 1.28)
        self.path.quadTo(270.98, 1.28, 270.21, -10.37)
        
        # A
        self.path.moveTo(296.51, 0)
        self.path.lineTo(289.47, 0)
        self.path.lineTo(297.15, -42.24)
        
        # N
        self.path.moveTo(363.01, 1.28)
        self.path.quadTo(354.82, 1.28, 354.05, -10.37)
        
        # A
        self.path.moveTo(406.46, -13.25)
        self.path.quadTo(408.13, -12.03, 408.13, -9.44)
        
        # C
        self.path.moveTo(440.77, -9.86)
        self.path.quadTo(441.98, -8.32, 441.98, -5.82)
        
        # E
        self.path.moveTo(475.97, 1.28)
        self.path.quadTo(467.78, 1.28, 467.01, -10.37)
        
        # Final A
        self.path.moveTo(461.5, -23.17)
        self.path.lineTo(457.98, -15.04)
        self.path.lineTo(466.82, -15.04)
        self.path.lineTo(466.05, -31.49)

    def update_animation(self):
        self.offset = (self.offset + 5) % 1000
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Adjust coordinate system to match XAML
        painter.translate(self.width() / 2, self.height() / 2 + 50)  # Adjust Y offset
        scale_factor = min(self.width() / 800, self.height() / 200)  # Adjusted scale
        painter.scale(scale_factor, scale_factor)

        # Create gradient with adjusted coordinates
        gradient = QLinearGradient(QPointF(-400, -100), QPointF(400, 100))
        gradient.setColorAt(0, QColor("#7C3AED"))  # Purple
        gradient.setColorAt(1, QColor("#F472B6"))  # Pink

        # Create animated pen
        pen = QPen()
        pen.setWidth(8)  # Adjusted width
        pen.setBrush(gradient)
        pen.setDashPattern([20, 20])
        pen.setDashOffset(self.offset)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)

        # Draw the path
        painter.drawPath(self.path)

class LoaderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("CyberPanacea is Loading")
        self.setFixedSize(1000, 600)
        self.setStyleSheet("background-color: white;")
        
        # Remove window frame
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Add animated path
        self.path_widget = AnimatedPath()
        layout.addWidget(self.path_widget)
        
        # Add animated text
        self.text_widget = AnimatedText()
        layout.addWidget(self.text_widget)
        
        # Center window on screen
        self.center_on_screen()
        
    def center_on_screen(self):
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().screenGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    window = LoaderWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()