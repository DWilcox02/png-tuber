from landmarking.landmarker import Landmarker
from png_management.png_selector import PngSelector
from video_processing.video_capture import VideoCapture

if __name__ == "__main__":
    video_capture = VideoCapture()
    landmarker = Landmarker()
    png_selector = PngSelector()

    while True:
        pass

    png_selector.terminate()
    landmarker.terminate()
    video_capture.terminate()