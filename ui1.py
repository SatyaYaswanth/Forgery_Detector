import sys
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, 
                             QVBoxLayout, QWidget, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QPainter, QPainterPath
from PyQt5.QtCore import Qt
from PIL import Image
import torchvision.transforms as transforms

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Forgery Detector")
        self.resize(500, 700)
        
        # Load the TorchScript model (assumes model.pt is in the working directory)
        self.device = torch.device("cpu")
        self.model = torch.jit.load("model.pt", map_location=self.device)
        self.model.eval()
        
        # Define the transformation to match training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # scales pixel values to [0,1]
        ])
        
        # UI Components with placeholder texts
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setFixedSize(256, 256)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #AAAAAA;
                border-radius: 15px;
                background-color: #F9F9F9;
            }
        """)
        self.image_label.setScaledContents(True)
        
        self.result_label = QLabel("<i><b>Prediction result will appear here</b></i>")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                padding: 10px;
                border: 2px dashed #AAAAAA;
                border-radius: 15px;
                background-color: #F1F1F1;
                color: #333333;
            }
        """)
        
        self.load_button = QPushButton("Load Image")
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #6200EE;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #3700B3;
            }
        """)
        self.load_button.clicked.connect(self.load_image)
        
        # Responsive layout with spacers
        layout = QVBoxLayout()
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.load_button, alignment=Qt.AlignCenter)
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        container = QWidget()
        container.setLayout(layout)
        container.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-radius: 15px;
            }
        """)
        self.setCentralWidget(container)
    
    def load_image(self):
        # Open file dialog to select an image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            # Display the selected image with rounded corners
            pixmap = QPixmap(file_path)
            rounded = self.getRoundedPixmap(pixmap, 15)
            self.image_label.setPixmap(rounded.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Run model prediction
            prediction, probability = self.predict_image(file_path)
            if prediction == "Real":
                text = "<b><i>The image appears to be authentic.</i></b>"
                text += f"<br><span style='font-size:14px;'>Confidence: {int(probability*100)}%</span>"
            else:
                text = "<b><i>The image might be a forged Image.</i></b>"
                text += f"<br><span style='font-size:14px;'>Confidence: {int((1-probability)*100)}%</span>"
            # Append probability to text
            
            self.result_label.setText(text)
    
    def getRoundedPixmap(self, pixmap, radius):
        # Create a rounded QPixmap
        size = pixmap.size()
        rounded = QPixmap(size)
        rounded.fill(Qt.transparent)
        
        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, size.width(), size.height(), radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return rounded
    
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
