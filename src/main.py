import time
from landmarker import Landmarker
from png_selector import PngSelector
from video_capture import VideoCapture
from facial_features.facial_analysis import FacialAnalyzer
from audio_processor import AudioProcessor


def main():
    """Main execution loop for the PNG Tuber application."""
    video_capture = VideoCapture()
    landmarker = Landmarker()
    png_selector = PngSelector()
    audio_processor = AudioProcessor()
    facial_analyzer = FacialAnalyzer(flap_interval_sec=0.15)

    try:
        while True:
            # 1. Capture and prepare frame
            image = video_capture.read_frame()
            if image is None:
                break

            rgb_image = video_capture.to_rgb(image)
            mp_image = landmarker.to_mp_image(rgb_image)

            # 2. Run MediaPipe Inference
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            # 3. Analyze Facial State
            latest_landmarks = landmarker.get_latest_landmarks()
            latest_blendshapes = landmarker.get_latest_blendshapes()

            status = facial_analyzer.create_status(latest_blendshapes, is_talking=audio_processor.is_talking)

            # 4. Select and Display Assets
            cat_image_path = png_selector.select_image(status)

            video_capture.draw_landmarks(image, latest_landmarks)
            video_capture.show_frame(image)
            video_capture.show_cat(cat_image_path, facial_status=status)

            # 5. Handle Exit
            if video_capture.wait_esc():
                break
    finally:
        # 6. Cleanup
        audio_processor.terminate()  # <-- Shut down STT/Mic safely
        png_selector.terminate()
        landmarker.terminate()
        video_capture.terminate()


if __name__ == "__main__":
    main()
