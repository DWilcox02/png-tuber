import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker_v2_with_blendshapes.task"

class Landmarker():

    

    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Global variable to hold the most recent results from the async callback
        latest_landmarks = None


        # Callback function to update global state
        def update_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            global latest_landmarks
            if result.face_landmarks:
                # We only care about the first face detected
                latest_landmarks = result.face_landmarks[0]
            else:
                latest_landmarks = None


        # Thresholds
        eye_opening_threshold = 0.025
        mouth_open_threshold = 0.03
        squinting_threshold = 0.018
