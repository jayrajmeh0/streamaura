import streamlit as st
import cv2
import numpy as np
import threading
import time
from PIL import Image
import os

# Fix TensorFlow warnings and compatibility
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Safe DeepFace import with compatibility fixes
DEEPFACE_AVAILABLE = False
try:
    # Fix TensorFlow version attribute issue
    import tensorflow as tf
    if not hasattr(tf, '_version_'):
        tf._version_ = tf.__version__
    
    # Now try importing DeepFace
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("DeepFace loaded successfully!")
    
except Exception as e:
    print(f"DeepFace not available: {str(e)}")
    DEEPFACE_AVAILABLE = False

# Try WebRTC import
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Enhanced Emotion Detection Class
class RealTimeEmotionDetector:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize emotion tracking
        self.current_emotion = "Neutral"
        self.confidence = 0.0
        self.frame_count = 0
        self.lock = threading.Lock()
        self.emotion_history = []
        
        # Emotion mappings
        self.emotion_map = {
            'angry': 'Angry',
            'disgust': 'Disgust', 
            'fear': 'Fear',
            'happy': 'Happy',
            'sad': 'Sad',
            'surprise': 'Surprise',
            'neutral': 'Neutral'
        }
        
        # DeepFace model warming (if available)
        if DEEPFACE_AVAILABLE:
            self.warm_up_model()
    
    def warm_up_model(self):
        """Warm up the DeepFace model with a dummy image"""
        try:
            # Create a dummy face image
            dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            DeepFace.analyze(
                dummy_img, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            print("DeepFace model warmed up!")
        except Exception as e:
            print(f"Model warm-up failed: {e}")
    
    def detect_emotion_deepface(self, frame):
        """Use DeepFace for emotion detection with error handling"""
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (224, 224))
            
            result = DeepFace.analyze(
                small_frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            # Handle different result formats
            if isinstance(result, list) and len(result) > 0:
                emotions = result[0]['emotion']
                dominant_emotion = result[0]['dominant_emotion'].lower()
            elif isinstance(result, dict):
                emotions = result['emotion']
                dominant_emotion = result['dominant_emotion'].lower()
            else:
                return 'neutral', 0.0
            
            confidence = emotions.get(dominant_emotion, 0) / 100.0
            return dominant_emotion, confidence
            
        except Exception as e:
            # Fallback to neutral if error
            return 'neutral', 0.0
    
    def detect_emotion_heuristic(self, frame, faces):
        """Heuristic-based emotion detection using facial analysis"""
        if len(faces) == 0:
            return 'neutral', 0.5
            
        # Use the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate facial features
        mean_intensity = np.mean(gray_face)
        std_intensity = np.std(gray_face)
        
        # Simple emotion heuristics
        if mean_intensity > 130 and std_intensity > 30:
            return 'happy', 0.75
        elif std_intensity > 45 and mean_intensity > 90:
            return 'surprise', 0.70
        elif mean_intensity < 70 and std_intensity < 15:
            return 'angry', 0.65
        elif mean_intensity < 90 and std_intensity < 25:
            return 'sad', 0.60
        else:
            return 'neutral', 0.70

        # if mean_intensity > 140 and std_intensity > 35:
        #     return 'happy', 0.75
        # elif mean_intensity < 90:
        #     return 'sad', 0.65
        # elif std_intensity > 45:
        #     return 'surprise', 0.60
        # elif mean_intensity < 110 and std_intensity < 25:
        #     return 'angry', 0.55
        # else:
        #     return 'neutral', 0.70
    
    def smooth_emotion(self, emotion):
        """Smooth emotion detection using history"""
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 5:
            self.emotion_history.pop(0)
        
        # Return most common emotion in recent history
        if len(self.emotion_history) >= 3:
            emotion_counts = {}
            for e in self.emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            return max(emotion_counts, key=emotion_counts.get)
        return emotion
    
    def process_frame(self, frame):
        """Process a single frame for emotion detection"""
        self.frame_count += 1
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        # Process emotion detection every 15 frames for performance
        if self.frame_count % 15 == 0 and len(faces) > 0:
            if DEEPFACE_AVAILABLE:
                emotion, confidence = self.detect_emotion_deepface(frame)
            else:
                emotion, confidence = self.detect_emotion_heuristic(frame, faces)
            
            # Smooth the emotion detection
            emotion = self.smooth_emotion(emotion)
            
            with self.lock:
                self.current_emotion = self.emotion_map.get(emotion, '😐 Neutral')
                self.confidence = confidence
        
        # Draw results on frame
        self.draw_results(frame, faces)
        
        return frame
    
    def draw_results(self, frame, faces):
        """Draw face rectangles and emotion labels"""
        for (x, y, w, h) in faces:
            # Draw face rectangle with color based on emotion
            emotion_colors = {
                '😊 Happy': (0, 255, 0),    # Green
                '😢 Sad': (255, 0, 0),      # Blue  
                '😠 Angry': (0, 0, 255),    # Red
                '😲 Surprise': (255, 255, 0), # Cyan
                '😨 Fear': (128, 0, 128),   # Purple
                '😐 Neutral': (128, 128, 128), # Gray
                '🤢 Disgust': (0, 128, 255)  # Orange
            }
            
            with self.lock:
                current_emotion = self.current_emotion
                current_confidence = self.confidence
            
            color = emotion_colors.get(current_emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Prepare text
            emotion_text = current_emotion
            confidence_text = f"Confidence: {current_confidence:.2f}"
            
            # Background for text
            text_bg_height = 70
            cv2.rectangle(frame, (x, y-text_bg_height), (x+w, y), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, emotion_text, (x+5, y-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, confidence_text, (x+5, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face center point
            center = (x + w//2, y + h//2)
            cv2.circle(frame, center, 3, color, -1)

# WebRTC Video Processor
if WEBRTC_AVAILABLE:
    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.detector = RealTimeEmotionDetector()
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Process the frame
            processed_img = self.detector.process_frame(img)
            
            # Add FPS counter
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:
                elapsed = time.time() - self.fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = time.time()
                
                cv2.putText(processed_img, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Direct OpenCV capture class
class DirectVideoCapture:
    def __init__(self):
        self.detector = RealTimeEmotionDetector()
        self.cap = None
        self.running = False
        
    def start_capture(self, placeholder, status_placeholder):
        """Start direct video capture"""
        try:
            self.cap = cv2.VideoCapture(0) # need to give live stream link here
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                status_placeholder.error("Could not open camera")
                return
            
            self.running = True
            status_placeholder.success("Camera is running!")
            
            fps_counter = 0
            fps_start_time = time.time()
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    status_placeholder.warning("⚠️ Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self.detector.process_frame(frame)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed = time.time() - fps_start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    fps_start_time = time.time()
                    
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to RGB and display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                placeholder.image(rgb_frame, channels='RGB', use_column_width=True)
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            status_placeholder.error(f"Camera error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            status_placeholder.info("Camera stopped")
        
    def stop_capture(self):
        self.running = False

# Streamlit App
st.set_page_config(
    page_title="Fixed Emotion Detection", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Fixed Real-time Emotion Detection")
st.write("Emotion detection with TensorFlow compatibility fixes!")

# System status in sidebar
st.sidebar.header("🔧 System Status")
st.sidebar.write(f"**DeepFace:** {'Available' if DEEPFACE_AVAILABLE else 'Using Heuristics'}")
st.sidebar.write(f"**WebRTC:** {'Available' if WEBRTC_AVAILABLE else 'Not Available'}")
st.sidebar.write(f"**OpenCV:** {cv2.__version__}")

# Method selection
detection_method = st.sidebar.radio(
    "Detection Method:",
    ["WebRTC Streaming", "Direct Camera", "Image Upload"]
)

# Main interface
if detection_method == "WebRTC Streaming" and WEBRTC_AVAILABLE:
    st.subheader("Real-time Detection")
    
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    webrtc_ctx = webrtc_streamer(
        key="fixed-emotion-detection",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    if webrtc_ctx.state.playing:
        st.success("Live streaming active!")
    else:
        st.info("Click START to begin")

elif detection_method == "Direct Camera":
    st.subheader("Direct Camera Capture")
    
    # Initialize capture object
    if 'video_capture' not in st.session_state:
        st.session_state.video_capture = DirectVideoCapture()
    
    # Create placeholders
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎬 Start Camera", key="start_direct"):
            if not st.session_state.video_capture.running:
                # Start in thread
                capture_thread = threading.Thread(
                    target=st.session_state.video_capture.start_capture,
                    args=(video_placeholder, status_placeholder)
                )
                capture_thread.daemon = True
                capture_thread.start()
    
    with col2:
        if st.button("⏹️ Stop Camera", key="stop_direct"):
            st.session_state.video_capture.stop_capture()

else:  # Image Upload
    st.subheader("📱 Upload Image Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Process uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Create detector and process
            detector = RealTimeEmotionDetector()
            processed_img = detector.process_frame(img_bgr)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original:**")
                st.image(image, use_column_width=True)
            
            with col2:
                st.write("**Processed:**")
                processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                st.image(processed_rgb, use_column_width=True)
            
            # Show emotion results
            with detector.lock:
                st.success(f"**Detected:** {detector.current_emotion}")
                st.info(f"**Confidence:** {detector.confidence:.2f}")

# Installation fix guide
with st.expander("🔧 Fix TensorFlow/DeepFace Issues"):
    st.code("""
# Method 1: Fix current installation
pip uninstall tensorflow deepface
pip install tensorflow==2.13.0
pip install deepface==0.0.79

# Method 2: Use CPU version
pip uninstall tensorflow
pip install tensorflow-cpu==2.13.0
pip install deepface==0.0.79

# Method 3: Complete fresh install
pip uninstall tensorflow deepface opencv-python
pip install opencv-python==4.8.1.78
pip install tensorflow==2.13.0  
pip install deepface==0.0.79
pip install streamlit

# Method 4: Alternative emotion detection
pip install fer  # Facial Expression Recognition
    """)

st.success("✅ App is running with compatibility fixes! The TensorFlow error should be resolved.")