import sys
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, 
                            QFrame, QSizePolicy)
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor, QPen, QPainterPath
from PyQt5.QtCore import Qt, QSize, QRect, QRectF
from PIL import Image
import torchvision.transforms as transforms

class RoundedImageLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = 20
        self.setMinimumSize(256, 256)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        if self.pixmap():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            path = QPainterPath()
            rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
            path.addRoundedRect(rect, self.radius, self.radius)
            
            painter.setClipPath(path)
            painter.drawPixmap(self.rect(), self.pixmap())
        else:
            super().paintEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Forgery Detector")
        self.resize(500, 700)
        self.setStyleSheet("background-color: #f5f5f7;")
        
        # Load the TorchScript model
        self.device = torch.device("cpu")
        self.model = torch.jit.load("model.pt", map_location=self.device)
        self.model.eval()
        
        # Define the transformation to match training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # UI Components
        self.setupUI()
        
    def setupUI(self):
        # Main container with margins
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Image Forgery Detector")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #1d1d1f; margin-bottom: 10px;")
        
        # Subtitle
        subtitle_label = QLabel("Upload an image to check if it's real or manipulated")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #515154; margin-bottom: 20px;")
        
        # Image display area
        self.image_label = RoundedImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: #e1e1e6;
            border: 2px dashed #c7c7cc;
            border-radius: 20px;
        """)
        
        # Placeholder text for image
        self.placeholder_label = QLabel("No image loaded\nClick 'Upload Image' to begin", self.image_label)
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setFont(QFont("Arial", 12))
        self.placeholder_label.setStyleSheet("color: #8e8e93; background: transparent; border: none;")
        self.placeholder_label.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        
        # Result frame
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        result_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                padding: 15px;
                border: 1px solid #e1e1e6;
            }
        """)
        
        result_layout = QVBoxLayout(result_frame)
        
        result_title = QLabel("Analysis Result")
        result_title.setFont(QFont("Arial", 14, QFont.Bold))
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("color: #1d1d1f; margin-bottom: 5px;")
        
        self.result_label = QLabel("Waiting for image analysis...")
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #515154;")
        
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Arial", 10, QFont.StyleItalic))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #8e8e93; font-style: italic;")
        
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.confidence_label)
        
        # Button area
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Upload Image")
        self.load_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.load_button.setCursor(Qt.PointingHandCursor)
        self.load_button.setMinimumHeight(50)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #0071e3;
                color: white;
                border-radius: 25px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #0077ED;
            }
            QPushButton:pressed {
                background-color: #005BBB;
            }
        """)
        self.load_button.clicked.connect(self.load_image)
        
        button_layout.addStretch()
        button_layout.addWidget(self.load_button)
        button_layout.addStretch()
        
        # Add everything to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addWidget(self.image_label, 1)
        main_layout.addWidget(result_frame)
        main_layout.addLayout(button_layout)
        
        # Set the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
    def resizeEvent(self, event):
        # Update placeholder position on resize
        if hasattr(self, 'placeholder_label'):
            self.placeholder_label.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        super().resizeEvent(event)

    def load_image(self):
        # Open file dialog to select an image
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.webp)"
        )
        if file_path:
            # Display the selected image
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            
            # Hide placeholder when image is loaded
            self.placeholder_label.hide()
            
            # Run model prediction
            prediction, probability = self.predict_image(file_path)
            self.update_result(prediction, probability)
    
    def update_result(self, prediction, probability):
        if prediction == "Real":
            if probability > 0.9:
                message = "<b>This appears to be a genuine image</b>"
                confidence = f"High confidence: {probability:.1%}"
                color = "#34c759"  # Green
            else:
                message = "<b>This image is likely genuine</b>, but shows some unusual patterns"
                confidence = f"Moderate confidence: {probability:.1%}"
                color = "#5ac8fa"  # Blue
        else:
            if probability < 0.1:
                message = "<b>This image appears to be digitally manipulated</b>"
                confidence = f"High confidence: {1-probability:.1%}"
                color = "#ff3b30"  # Red
            else:
                message = "<b>This image might contain manipulated elements</b>"
                confidence = f"Moderate confidence: {1-probability:.1%}"
                color = "#ff9500"  # Orange
                
        self.result_label.setText(message)
        self.confidence_label.setText(confidence)
        self.result_label.setStyleSheet(f"color: {color}; font-size: 14px;")
    
    def predict_image(self, image_path):
        # Load and process the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference on the CPU
        with torch.no_grad():
            output = self.model(input_tensor).squeeze()
            probability = torch.sigmoid(output).item()
        
        prediction = "Real" if probability >= 0.5 else "Fake"
        return prediction, probability

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())