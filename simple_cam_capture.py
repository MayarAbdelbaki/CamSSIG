import cv2
import os
from pop import Util

# Global camera instance
_camera = None
_image_counter = 1
_save_folder = "captured_images"

def init_camera(width=640, height=480, save_folder="captured_images"):
    """
    Initialize the camera. Call this once at the start of your program.
    
    Args:
        width: Camera width
        height: Camera height
        save_folder: Folder to save images
    """
    global _camera, _save_folder, _image_counter
    
    _save_folder = save_folder
    
    # Create folder if it doesn't exist
    if not os.path.exists(_save_folder):
        os.makedirs(_save_folder)
    
    # Initialize camera
    Util.enable_imshow()
    cam = Util.gstrmer(width=width, height=height)
    _camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
    
    if not _camera.isOpened():
        print("Camera not found")
        return False
    
    print(f"Camera initialized: {width}x{height}")
    return True


def cap_cam():
    """
    Capture a single image from the camera and save it.
    Returns the saved filename or None if failed.
    """
    global _camera, _image_counter, _save_folder
    
    if _camera is None or not _camera.isOpened():
        print("Camera not initialized. Call init_camera() first.")
        return None
    
    # Read frame
    ret, frame = _camera.read()
    
    if not ret:
        print("Failed to capture image")
        return None
    
    # Save image
    filename = f"{_image_counter}.jpg"
    filepath = os.path.join(_save_folder, filename)
    cv2.imwrite(filepath, frame)
    
    print(f"Image saved: {filepath}")
    _image_counter += 1
    
    return filename


def release_camera():
    """Release the camera when done."""
    global _camera
    
    if _camera is not None:
        _camera.release()
        cv2.destroyAllWindows()
        print("Camera released")


# Example usage
if __name__ == "__main__":
    # Initialize camera
    init_camera()
    
    # Simulate person detection
    print("Simulating person detection...")
    
    # When person detected, capture image
    cap_cam()
    cap_cam()
    cap_cam()
    
    # Release when done
    release_camera()
