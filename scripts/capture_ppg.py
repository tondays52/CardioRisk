"""
CardioRisk AI - PPG Camera Capture
Captures fingertip video for PPG analysis.
"""
import cv2
import time
import numpy as np

def capture_fingertip_video(duration=30, output_path="data/raw/ppg/capture.mp4"):
    """
    Capture fingertip video using webcam.
    
    Instructions:
    - Place your index finger over the rear camera
    - Turn on the flashlight for better signal
    - Keep still during recording
    """
    import os
    os.makedirs("data/raw/ppg", exist_ok=True)
    
    print("Starting camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))
    
    print(f"\nRecording {duration} seconds of PPG...")
    print("Place your fingertip over the camera lens")
    print("Press 'q' to stop early\n")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show live feed
        cv2.imshow("PPG Capture - Fingertip", frame)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Show timer
        elapsed = time.time() - start_time
        if frame_count % 30 == 0:
            print(f"Recording: {elapsed:.0f}/{duration}s", end='\r')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"\nRecorded {frame_count} frames ({frame_count/30:.1f} seconds)")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_path

if __name__ == "__main__":
    video_path = capture_fingertip_video(duration=30)
    
    if video_path:
        print(f"\nVideo saved to: {video_path}")
        print("\nNow run:")
        print(f"python scripts/analyze_capture.py {video_path}")