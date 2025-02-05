import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp

class EmotionCNN(nn.Module):
    def __init__(self, input_shape=(1, 48, 48), num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Initial conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fourth conv block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flattened_size = self._calculate_flattened_size(input_shape)
        
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)

    def _calculate_flattened_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.numel()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class EmotionDetectionApp:
    def __init__(self, model_path):
        self.model = EmotionCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        self.emotions = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 
                        'Neutral 😐', 'Sad 😢', 'Surprise 😲']
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.7)

    def detect_emotions(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                x, y = max(0, x), max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face_tensor = self.transform(Image.fromarray(face)).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(face_tensor)
                    probs = F.softmax(output, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_idx].item() * 100
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    emotion = self.emotions[pred_idx]
                    label = f"{emotion} ({confidence:.1f}%)"
                    cv2.putText(frame, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

    def run_app(self):
        st.set_page_config(page_title="Emotion Detector", page_icon="🧠")
        st.title("🧠 Emotion Detection")
        option = st.radio("Choose Input:", ["Camera", "Upload Image"])
        
        if option == "Camera":
            st.header("Live Camera Detection")
            col1, col2 = st.columns(2)
            with col1:
                start_button = st.button("Start Camera")
            with col2:
                stop_button = st.button("Stop Camera")
            
            frame_placeholder = st.empty()
            
            if start_button:
                cap = cv2.VideoCapture(0)
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if ret:
                        processed_frame = self.detect_emotions(frame)
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                cap.release()
        
        else:
            st.header("Image Upload Detection")
            uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file:
                image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption="Uploaded Image", 
                        use_container_width=True)
                
                processed_image = self.detect_emotions(image)
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), 
                        caption="Detected Emotions", 
                        use_container_width=True)

def main():
    model_path = r"D:\bhachu\new\best_model_fold_4.pth"  # my model path
    app = EmotionDetectionApp(model_path)
    app.run_app()

if __name__ == "__main__":
    main()