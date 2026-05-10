import time
from landmarker import Landmarker
from png_selector import PngSelector
from video_capture import VideoCapture

if __name__ == "__main__":
    video_capture = VideoCapture()
    landmarker = Landmarker()
    png_selector = PngSelector()

    try:
        while True:
            image = video_capture.read_frame()
            if image is None:
                break

            rgb_image = video_capture.to_rgb(image)
            mp_image = landmarker.to_mp_image(rgb_image)

            # Increasing timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)

            # Asynchronous mediapipe sending
            landmarker.detect_async(mp_image, timestamp_ms)

            latest_landmarks = landmarker.get_latest_landmarks()
            cat_image = png_selector.get_cat_image(latest_landmarks)

            video_capture.draw_landmarks(image, latest_landmarks)

            video_capture.show_frame(image)

            video_capture.show_cat(cat_image, fallback_image=image)

            # Press 'ESC' to exit
            if video_capture.wait_esc():
                break
    finally:
        png_selector.terminate()
        landmarker.terminate()
        video_capture.terminate()
