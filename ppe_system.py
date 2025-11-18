"""
PPE Detection System
Integrates hardware camera with face detection and Hugging Face PPE detection API
"""

import cv2
import time
import os
import tempfile
from datetime import datetime
from typing import Optional
from pop import Util

# Try to import gradio_client
try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    print("Note: gradio_client not available. Install with: pip install gradio_client")


class PPEDetectionSystem:
    def __init__(
        self, 
        width: int = 640, 
        height: int = 480,
        wait_interval: int = 5,
        hf_token: Optional[str] = None,
        use_gradio_client: bool = True
    ):
        """
        Initialize PPE detection system with hardware camera
        
        Args:
            width: Camera frame width
            height: Camera frame height
            wait_interval: Seconds to wait after face detection before capturing
            hf_token: Optional Hugging Face API token
            use_gradio_client: Use gradio_client library (recommended)
        """
        self.width = width
        self.height = height
        self.wait_interval = wait_interval
        self.hf_token = hf_token
        self.use_gradio_client = use_gradio_client and GRADIO_CLIENT_AVAILABLE
        
        # Camera and detection
        self.camera = None
        self.face_cascade = None
        
        # State management
        self.face_detected = False
        self.last_detection_time = None
        
        # API client
        self.gradio_client = None
        if self.use_gradio_client:
            try:
                space_url = "https://mayarelshamy-ppe-detection-system.hf.space"
                self.gradio_client = Client(space_url, hf_token=hf_token)
                print("Using gradio_client for API calls (recommended)")
            except Exception as e:
                print(f"Failed to initialize gradio_client: {e}")
                self.use_gradio_client = False
    
    def _init_camera(self):
        """Initialize hardware camera"""
        Util.enable_imshow()
        cam = Util.gstrmer(width=self.width, height=self.height)
        self.camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
        
        if not self.camera.isOpened():
            raise RuntimeError("Camera not found or could not be opened")
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"✓ Camera initialized: {int(actual_width)}x{int(actual_height)}")
    
    def _init_face_detector(self):
        """Initialize face detection"""
        haar_path = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
        
        print("✓ Face detector initialized")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )
        return len(faces) > 0, faces
    
    def send_to_api(self, frame) -> Optional[dict]:
        """Send frame to PPE detection API"""
        if not self.use_gradio_client or not self.gradio_client:
            return None
        
        try:
            # Save frame to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            _, buffer = cv2.imencode('.jpg', frame)
            temp_file.write(buffer.tobytes())
            temp_file.close()
            
            # Send to API
            result = self.gradio_client.predict(
                image=handle_file(temp_file.name),
                api_name="/predict"
            )
            
            # Clean up
            os.unlink(temp_file.name)
            
            return {"status": "success", "result": result}
            
        except Exception as e:
            print(f"✗ API error: {e}")
            return None
    
    def run(self, show_preview: bool = True):
        """Main detection loop"""
        # Initialize
        self._init_camera()
        self._init_face_detector()
        
        print("\n=== PPE Detection System Active ===")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("✗ Failed to read frame")
                    break
                
                # Detect faces
                has_face, faces = self.detect_faces(frame)
                
                # Display preview
                if show_preview:
                    display_frame = frame.copy()
                    
                    # Draw face rectangles
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Status text
                    if self.face_detected:
                        elapsed = time.time() - self.last_detection_time
                        remaining = max(0, self.wait_interval - elapsed)
                        status = f"Waiting: {remaining:.1f}s"
                        color = (0, 255, 255)
                    else:
                        status = "Face Detected" if has_face else "Scanning..."
                        color = (0, 255, 0) if has_face else (255, 255, 255)
                    
                    cv2.putText(display_frame, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    cv2.imshow('PPE Detection System', display_frame)
                
                # Handle face detection logic
                if has_face:
                    if not self.face_detected:
                        # First detection - start waiting
                        self.face_detected = True
                        self.last_detection_time = time.time()
                    else:
                        # Check if wait period elapsed
                        elapsed = time.time() - self.last_detection_time
                        if elapsed >= self.wait_interval:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] Capturing and sending to API...")
                            
                            # Send to API
                            result = self.send_to_api(frame)
                            if result:
                                print(f"[{timestamp}] Result: {result['result']}")
                            
                            # Reset state
                            self.face_detected = False
                            self.last_detection_time = None
                            time.sleep(2)
                else:
                    # No face - reset state
                    if self.face_detected:
                        self.face_detected = False
                        self.last_detection_time = None
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n✓ Stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("✓ System shutdown complete")


def main():
    """Main entry point"""
    # Configuration
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    WAIT_INTERVAL = 5  # seconds
    HF_TOKEN = None  # Optional: add your Hugging Face token
    USE_GRADIO_CLIENT = True  # Set to False to disable API calls
    
    # Run system
    system = PPEDetectionSystem(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        wait_interval=WAIT_INTERVAL,
        hf_token=HF_TOKEN,
        use_gradio_client=USE_GRADIO_CLIENT
    )
    system.run(show_preview=True)


if __name__ == "__main__":
    main()

