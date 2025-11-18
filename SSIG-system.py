"""
SSIG Face Detection System
Combines hardware camera from car with Hugging Face PPE Detection API
"""

import cv2
import time
import requests
import base64
import os
import tempfile
from datetime import datetime
from typing import Optional
from popx import Util

# Try to import gradio_client
try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    print("Note: gradio_client not available. Falling back to HTTP API.")


class SSIGDetector:
    def __init__(self, huggingface_api_url: str = None, huggingface_api_token: Optional[str] = None, 
                 use_gradio_client: bool = True, width: int = 640, height: int = 480):
        """
        Initialize SSIG detector with hardware camera and Hugging Face API
        
        Args:
            huggingface_api_url: URL of the Hugging Face API endpoint
            huggingface_api_token: Optional API token
            use_gradio_client: Use gradio_client library (recommended) or HTTP API
            width: Camera frame width
            height: Camera frame height
        """
        self.huggingface_api_url = huggingface_api_url
        self.huggingface_api_token = huggingface_api_token
        self.use_gradio_client = use_gradio_client and GRADIO_CLIENT_AVAILABLE
        self.width = width
        self.height = height
        
        # API client
        self.gradio_client = None
        
        # Camera and detection
        self.camera = None
        self.face_cascade = None
        
        # Detection timing
        self.face_detected = False
        self.last_face_detection_time = None
        self.wait_interval = 5  # Wait 5 seconds after face detection
        
        # Initialize gradio_client if available
        if self.use_gradio_client:
            try:
                space_url = "https://mayarelshamy-ppe-detection-system.hf.space"
                self.gradio_client = Client(space_url, hf_token=huggingface_api_token)
                print("✓ Using gradio_client for API calls")
            except Exception as e:
                print(f"Failed to initialize gradio_client: {e}")
                print("Falling back to HTTP API method")
                self.use_gradio_client = False
    
    def initialize(self):
        """Initialize hardware camera and face detection"""
        # Initialize camera with hardware settings
        Util.enable_imshow()
        cam = Util.gstrmer(width=self.width, height=self.height)
        self.camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
        
        if not self.camera.isOpened():
            raise RuntimeError("Camera not found or could not be opened")
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"✓ Camera initialized: {actual_width}x{actual_height}")
        
        # Load face detection cascade
        haar_face = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_face)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
        
        print("✓ Face detector initialized")
    
    def detect_face(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100)
        )
        return len(faces) > 0, faces
    
    def send_to_huggingface_gradio_client(self, frame) -> dict:
        """Send image using gradio_client (recommended method)"""
        try:
            # Save frame to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            _, buffer = cv2.imencode('.jpg', frame)
            temp_file.write(buffer.tobytes())
            temp_file.close()
            
            # Use gradio_client
            result = self.gradio_client.predict(
                image=handle_file(temp_file.name),
                api_name="/predict"
            )
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            return {
                "status": "success",
                "raw_response": result,
                "full_response": {"data": [result]}
            }
        except Exception as e:
            print(f"Error with gradio_client: {e}")
            return None
    
    def send_to_huggingface_http(self, frame) -> dict:
        """Send image using HTTP API"""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare headers
            headers = {'Content-Type': 'application/json'}
            if self.huggingface_api_token:
                headers['Authorization'] = f'Bearer {self.huggingface_api_token}'
            
            # Prepare payload
            image_data_uri = f"data:image/jpeg;base64,{image_base64}"
            payload = {
                "data": [{"url": image_data_uri}]
            }
            
            # Send request
            response = requests.post(
                self.huggingface_api_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract result from response
            if "data" in result and len(result["data"]) > 0:
                return {
                    "status": "success",
                    "raw_response": result["data"][0],
                    "full_response": result
                }
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error sending to Hugging Face: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"Error details: {e.response.json()}")
                except:
                    print(f"Response status: {e.response.status_code}")
            return None
    
    def run(self, show_preview: bool = True):
        """Main detection loop"""
        self.initialize()
        
        print("\n" + "="*50)
        print("SSIG Face Detection System Started")
        print("="*50)
        print("Press 'q' to quit\n")
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Detect face
                face_found, faces = self.detect_face(frame)
                
                # Show preview if enabled
                if show_preview:
                    display_frame = frame.copy()
                    
                    # Draw rectangles around faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display status
                    status_text = "Face Detected - Waiting..." if self.face_detected else "Scanning..."
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('SSIG PPE Detection', display_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Handle face detection timing
                if face_found:
                    if not self.face_detected:
                        # First detection
                        self.face_detected = True
                        self.last_face_detection_time = time.time()
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Face detected, waiting {self.wait_interval} seconds...")
                    else:
                        # Check if wait period has passed
                        elapsed = time.time() - self.last_face_detection_time
                        if elapsed >= self.wait_interval:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Capturing and sending to Hugging Face...")
                            
                            # Send to API
                            if self.use_gradio_client and self.gradio_client:
                                result = self.send_to_huggingface_gradio_client(frame)
                            else:
                                result = self.send_to_huggingface_http(frame)
                            
                            if result:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Detection result: {result['raw_response']}")
                            else:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed to get detection result")
                            
                            # Reset detection state
                            self.face_detected = False
                            self.last_face_detection_time = None
                            
                            # Wait before next detection
                            time.sleep(2)
                else:
                    # No face detected, reset state
                    if self.face_detected:
                        self.face_detected = False
                        self.last_face_detection_time = None
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\nStopping SSIG system...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources and close windows"""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        print("✓ System stopped and resources released")


def main():
    """Main entry point"""
    # Configuration
    HUGGINGFACE_API_URL = "https://mayarelshamy-ppe-detection-system.hf.space/api/predict"
    HUGGINGFACE_API_TOKEN = ""  # Optional: Add token if needed
    USE_GRADIO_CLIENT = True  # Set to False to use HTTP API instead
    
    # Create and run detector
    detector = SSIGDetector(
        huggingface_api_url=HUGGINGFACE_API_URL if not USE_GRADIO_CLIENT else None,
        huggingface_api_token=HUGGINGFACE_API_TOKEN if HUGGINGFACE_API_TOKEN else None,
        use_gradio_client=USE_GRADIO_CLIENT,
        width=640,
        height=480
    )
    detector.run(show_preview=True)


if __name__ == "__main__":
    main()

