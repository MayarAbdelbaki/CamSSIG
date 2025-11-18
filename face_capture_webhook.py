"""
Face Capture with Webhook Integration (Auto Car Side)
Captures faces from camera and sends them to PC webhook endpoint
"""

import cv2
import time
import os
import requests
import base64
from typing import Optional
from datetime import datetime


class FaceCaptureWebhook:
    def __init__(
        self,
        webhook_url: str,
        save_folder: str = "captured_faces",
        width: int = 640,
        height: int = 480,
        capture_interval: int = 5,
        timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize Face Capture system with webhook integration (Auto Car Side).
        
        Args:
            webhook_url: Full URL of the webhook endpoint on PC (e.g., http://192.168.1.100:5000/webhook)
            save_folder: Directory to save captured face images locally
            width: Camera frame width
            height: Camera frame height
            capture_interval: Seconds between captures
            timeout: Webhook request timeout in seconds
            max_retries: Maximum number of retry attempts for webhook calls
        """
        self.webhook_url = webhook_url
        self.save_folder = save_folder
        self.width = width
        self.height = height
        self.capture_interval = capture_interval
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.image_counter = 1
        self.last_capture_time = 0
        self.camera = None
        self.face_cascade = None
        
        # Statistics
        self.stats = {
            'total_captures': 0,
            'webhook_successes': 0,
            'webhook_failures': 0,
            'last_response': None
        }
        
        # Setup
        self._create_save_folder()
        self._update_image_counter()
        self._init_camera()
        self._init_face_detector()
        
        print(f"✓ Face Capture Webhook Client initialized")
        print(f"  Webhook URL: {self.webhook_url}")
        print(f"  Save Folder: {self.save_folder}")
        print(f"  Capture Interval: {self.capture_interval}s")
    
    def _create_save_folder(self):
        """Create save folder if it doesn't exist."""
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            print(f"Created folder: {self.save_folder}")
    
    def _update_image_counter(self):
        """Update image counter based on existing files."""
        if not os.path.exists(self.save_folder):
            return
        existing_files = [f for f in os.listdir(self.save_folder) if f.endswith('.jpg')]
        if existing_files:
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.split('.')[0])
                    numbers.append(num)
                except ValueError:
                    continue
            if numbers:
                self.image_counter = max(numbers) + 1
    
    def _init_camera(self):
        """Initialize camera with hardware-specific settings."""
        try:
            from pop import Util
            Util.enable_imshow()
            cam = Util.gstrmer(width=self.width, height=self.height)
            self.camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
        except ImportError:
            # Fallback to default camera if pop module not available
            print("Note: pop module not found, using default camera")
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.camera.isOpened():
            raise Exception("Camera not found or could not be opened")
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"✓ Camera initialized: {actual_width}x{actual_height}")
    
    def _init_face_detector(self):
        """Initialize Haar Cascade face detector."""
        # Try multiple possible paths for haar cascade
        possible_paths = [
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]
        
        for haar_face in possible_paths:
            if os.path.exists(haar_face):
                self.face_cascade = cv2.CascadeClassifier(haar_face)
                if not self.face_cascade.empty():
                    print(f"✓ Face detector initialized: {haar_face}")
                    return
        
        raise Exception("Failed to load face cascade classifier from any known path")
    
    def detect_face(self, frame) -> bool:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            bool: True if at least one face is detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100)
        )
        return len(faces) > 0
    
    def _encode_image_base64(self, frame) -> str:
        """Encode frame to base64 string."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    
    def send_image_to_webhook(self, frame) -> Optional[dict]:
        """
        Send image to PC webhook endpoint.
        
        Args:
            frame: OpenCV frame to send
            
        Returns:
            Webhook response dict or None on failure
        """
        try:
            # Encode image to base64
            image_base64 = self._encode_image_base64(frame)
            
            # Prepare payload
            payload = {
                "image": image_base64,
                "timestamp": datetime.now().isoformat(),
                "image_id": self.image_counter,
                "format": "jpeg"
            }
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.webhook_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✓ Webhook Success: Image sent and processed")
                        return result
                    else:
                        print(f"✗ Webhook Error: {response.status_code} - {response.text}")
                        if attempt < self.max_retries - 1:
                            time.sleep(1)
                            continue
                        
                except requests.exceptions.Timeout:
                    print(f"✗ Webhook Timeout (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    
                except requests.exceptions.ConnectionError as e:
                    print(f"✗ Connection Error: Cannot reach webhook at {self.webhook_url}")
                    print(f"  Make sure PC webhook server is running and accessible")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    
                except requests.exceptions.RequestException as e:
                    print(f"✗ Request error: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
            
            return None
            
        except Exception as e:
            print(f"✗ Error sending to webhook: {e}")
            return None
    
    def capture_and_send(self, frame) -> Optional[str]:
        """
        Capture image, save locally, and send to webhook.
        
        Args:
            frame: Frame to capture
            
        Returns:
            Path to saved image or None
        """
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_capture_time < self.capture_interval:
            return None
        
        # Save image locally
        filename = f"{self.image_counter}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        cv2.imwrite(filepath, frame)
        
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Image #{self.image_counter} captured")
        print(f"Saved to: {filepath}")
        
        # Send to webhook
        result = self.send_image_to_webhook(frame)
        
        # Update statistics
        self.stats['total_captures'] += 1
        if result:
            self.stats['webhook_successes'] += 1
            self.stats['last_response'] = result
            if 'huggingface_result' in result:
                print(f"✓ HuggingFace Result: {result.get('huggingface_result', {}).get('status', 'unknown')}")
        else:
            self.stats['webhook_failures'] += 1
        
        print(f"Stats: {self.stats['webhook_successes']} successes, {self.stats['webhook_failures']} failures")
        print(f"{'='*60}\n")
        
        self.image_counter += 1
        self.last_capture_time = current_time
        
        return filepath
    
    def run(self, show_preview: bool = True):
        """
        Main loop to detect faces and send to webhook.
        
        Args:
            show_preview: Whether to show camera preview
        """
        print("\n" + "="*60)
        print("Starting Face Capture Webhook Client (Auto Car Side)")
        print("="*60)
        print(f"Webhook URL: {self.webhook_url}")
        print(f"Save folder: {self.save_folder}")
        print(f"Capture interval: {self.capture_interval} seconds")
        print(f"Press 'q' to quit, 's' to show stats")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("✗ Failed to read frame from camera")
                    break
                
                # Detect face
                face_detected = self.detect_face(frame)
                
                # If face detected, capture and send to webhook
                if face_detected:
                    saved_path = self.capture_and_send(frame)
                    if saved_path:
                        print(f"✓ Face detected, captured, and sent to webhook")
                
                # Show preview
                if show_preview:
                    display_frame = frame.copy()
                    
                    # Status text
                    status_text = "Face Detected!" if face_detected else "No Face"
                    color = (0, 255, 0) if face_detected else (0, 0, 255)
                    cv2.putText(display_frame, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Countdown text
                    time_until_next = max(0, self.capture_interval - (time.time() - self.last_capture_time))
                    if face_detected and time_until_next > 0:
                        countdown_text = f"Next capture in: {time_until_next:.1f}s"
                        cv2.putText(display_frame, countdown_text, (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Stats overlay
                    stats_text = f"Total: {self.stats['total_captures']} | Success: {self.stats['webhook_successes']} | Failed: {self.stats['webhook_failures']}"
                    cv2.putText(display_frame, stats_text, (10, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Webhook status
                    webhook_status = "Connected" if self.stats['webhook_successes'] > 0 else "Waiting..."
                    cv2.putText(display_frame, f"Webhook: {webhook_status}", (10, 140),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow("Face Capture Webhook (Auto Car)", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuit requested by user")
                        break
                    elif key == ord('s'):
                        self.print_stats()
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def print_stats(self):
        """Print current statistics."""
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Total captures: {self.stats['total_captures']}")
        print(f"Webhook successes: {self.stats['webhook_successes']}")
        print(f"Webhook failures: {self.stats['webhook_failures']}")
        if self.stats['total_captures'] > 0:
            success_rate = (self.stats['webhook_successes'] / self.stats['total_captures']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        if self.stats['last_response']:
            print(f"Last response: {self.stats['last_response']}")
        print("="*60 + "\n")
    
    def cleanup(self):
        """Release resources and close windows."""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        self.print_stats()
        print(f"\n✓ System stopped. Total images captured: {self.image_counter - 1}")


def main():
    """Main entry point for auto car side."""
    # Try to load from config file, otherwise use defaults
    try:
        from config_webhook import (
            PC_IP_ADDRESS, PC_PORT, AUTO_CAR_SAVE_FOLDER,
            AUTO_CAR_WIDTH, AUTO_CAR_HEIGHT, AUTO_CAR_CAPTURE_INTERVAL,
            WEBHOOK_TIMEOUT, WEBHOOK_MAX_RETRIES, get_webhook_url
        )
        PC_IP = PC_IP_ADDRESS
        WEBHOOK_URL = get_webhook_url()
        SAVE_FOLDER = AUTO_CAR_SAVE_FOLDER
        WIDTH = AUTO_CAR_WIDTH
        HEIGHT = AUTO_CAR_HEIGHT
        CAPTURE_INTERVAL = AUTO_CAR_CAPTURE_INTERVAL
        TIMEOUT = WEBHOOK_TIMEOUT
        MAX_RETRIES = WEBHOOK_MAX_RETRIES
        print("✓ Loaded configuration from config_webhook.py")
    except ImportError:
        # Fallback to hardcoded values if config file doesn't exist
        print("⚠️  config_webhook.py not found, using default values")
        PC_IP = "192.168.1.100"  # ⚠️ CHANGE THIS TO YOUR PC's IP ADDRESS
        PC_PORT = 5000
        WEBHOOK_URL = f"http://{PC_IP}:{PC_PORT}/webhook"
        SAVE_FOLDER = "captured_faces"
        WIDTH = 640
        HEIGHT = 480
        CAPTURE_INTERVAL = 5
        TIMEOUT = 10
        MAX_RETRIES = 3
    
    print("\n" + "="*60)
    print("Face Capture Webhook Client (Auto Car Side)")
    print("="*60)
    print(f"Webhook URL: {WEBHOOK_URL}")
    print(f"Make sure the webhook server is running on your PC!")
    print("="*60 + "\n")
    
    # Initialize and run
    try:
        face_capture = FaceCaptureWebhook(
            webhook_url=WEBHOOK_URL,
            save_folder=SAVE_FOLDER,
            width=WIDTH,
            height=HEIGHT,
            capture_interval=CAPTURE_INTERVAL,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES
        )
        
        face_capture.run(show_preview=True)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

