import cv2
import time
import mediapipe as mp
from landmarking.landmarker import Landmarker
from png_management.png_selector import PngSelector
from video_processing.video_capture import VideoCapture

if __name__ == "__main__":
    video_capture = VideoCapture()
    landmarker = Landmarker()
    png_selector = PngSelector()

    try:
        while True:
            image = video_capture.read_frame()
            if image is None:
                break

            # Convert OpenCV BGR to RGB, then to MediaPipe Image format
            rgb_image = video_capture.to_rgb(image)
            mp_image = landmarker.to_mp_image(rgb_image)

            # LIVE_STREAM requires a monotonically increasing timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)

            # Send the frame to MediaPipe (this runs asynchronously and triggers update_result)
            landmarker.detect_async(mp_image, timestamp_ms)

            latest_landmarks = landmarker.get_latest_landmarks()
            cat_image = png_selector.get_cat_image(latest_landmarks)

            # Draw the landmark points
            if latest_landmarks:
                height, width = image.shape[:2]
                for lm in latest_landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    cv2.circle(image, (x, y), 1, (0, 100, 0), -1)

            cv2.imshow("Face Detection", image)

            # Cat Display
            cat = cv2.imread(cat_image)
            if cat is not None:
                cat = cv2.resize(cat, (640, 480))
                cv2.imshow("Cat Image", cat)
            else:
                blank = image * 0
                cv2.putText(blank, f"Missing: {cat_image}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Cat Image", blank)

            # Press 'ESC' to exit
            key = cv2.waitKey(1)
            if key == 27:
                break
    finally:
        cv2.destroyAllWindows()
        png_selector.terminate()
        landmarker.terminate()
        video_capture.terminate()
