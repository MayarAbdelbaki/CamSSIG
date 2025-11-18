import cv2
import time
import os
from pop import Util

class FaceCapture:
    def __init__(self, save_folder="captured_faces", width=640, height=480):
        """
        Initialize the face capture system.
        
        Args:
            save_folder: Directory to save captured face images
            width: Camera frame width
            height: Camera frame height
        """
        self.save_folder = save_folder
        self.width = width
        self.height = height
        self.image_counter = 1
        self.last_capture_time = 0
        self.capture_interval = 5  # Wait 5 seconds between captures
        
        # Create save folder if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            print(f"Created folder: {self.save_folder}")
        else:
            # Get the next image number based on existing files
            self._update_image_counter()
        
        # Initialize camera
        self.camera = None
        self._init_camera()
        
        # Initialize face detection
        self.face_cascade = None
        self._init_face_detector()
    
    def _init_camera(self):
        """Initialize camera using hardware-specific settings."""
        Util.enable_imshow()
        cam = Util.gstrmer(width=self.width, height=self.height)
        self.camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
        
        if not self.camera.isOpened():
            raise Exception("Camera not found or could not be opened")
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera initialized: {actual_width}x{actual_height}")
    
    def _init_face_detector(self):
        """Initialize Haar Cascade face detector."""
        haar_face = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_face)
        
        if self.face_cascade.empty():
            raise Exception("Failed to load face cascade classifier")
        
        print("Face detector initialized")
    
    def _update_image_counter(self):
        """Update image counter based on existing files in the folder."""
        existing_files = [f for f in os.listdir(self.save_folder) if f.endswith('.jpg')]
        if existing_files:
            # Extract numbers from filenames and get the max
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.split('.')[0])
                    numbers.append(num)
                except ValueError:
                    continue
            if numbers:
                self.image_counter = max(numbers) + 1
    
    def detect_face(self, frame):
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            bool: True if at least one face is detected, False otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3,
            minNeighbors=5,  # Increased from 1 for more reliable detection
            minSize=(100, 100)
        )
        
        return len(faces) > 0
    
    def capture_image(self, frame):
        """
        Capture and save the image without bounding boxes.
        
        Args:
            frame: Frame to save
            
        Returns:
            str: Path to saved image or None if capture was skipped
        """
        current_time = time.time()
        
        # Check if enough time has passed since last capture
        if current_time - self.last_capture_time < self.capture_interval:
            return None
        
        # Save the original frame without any bounding boxes
        filename = f"{self.image_counter}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Face captured and saved: {filepath}")
        
        self.image_counter += 1
        self.last_capture_time = current_time
        
        return filepath
    
    def run(self, show_preview=True):
        """
        Main loop to continuously detect faces and capture images.
        
        Args:
            show_preview: Whether to show camera preview window
        """
        print("Starting face detection and capture system...")
        print(f"Images will be saved to: {self.save_folder}")
        print(f"Capture interval: {self.capture_interval} seconds")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Detect face in the frame
                face_detected = self.detect_face(frame)
                
                # If face detected, capture the image
                if face_detected:
                    saved_path = self.capture_image(frame)
                    if saved_path:
                        print(f"✓ Face detected and captured!")
                
                # Show preview if enabled
                if show_preview:
                    # Display status on frame
                    display_frame = frame.copy()
                    status_text = "Face Detected!" if face_detected else "No Face"
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if face_detected else (0, 0, 255), 2)
                    
                    # Show next capture countdown
                    time_until_next = max(0, self.capture_interval - (time.time() - self.last_capture_time))
                    if face_detected and time_until_next > 0:
                        countdown_text = f"Next capture in: {time_until_next:.1f}s"
                        cv2.putText(display_frame, countdown_text, (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow("Face Capture System", display_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit requested by user")
                        break
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources and close windows."""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        print(f"System stopped. Total images captured: {self.image_counter - 1}")


def main():
    """Main entry point for the face capture system."""
    # Initialize and run the face capture system
    face_capture = FaceCapture(save_folder="captured_faces", width=640, height=480)
    face_capture.run(show_preview=True)


if __name__ == "__main__":
    main()

