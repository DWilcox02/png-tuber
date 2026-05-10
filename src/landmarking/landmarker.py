import mediapipe as mp

MODEL_PATH = "models/face_landmarker_v2_with_blendshapes.task"

class Landmarker():
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.latest_landmarks = None

        # Callback function to update instance state
        def update_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            if result.face_landmarks:
                # We only care about the first face detected
                self.latest_landmarks = result.face_landmarks[0]
            else:
                self.latest_landmarks = None

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=update_result,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def to_mp_image(self, rgb_image):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    def detect_async(self, mp_image, timestamp_ms):
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def get_latest_landmarks(self):
        return self.latest_landmarks

    def terminate(self):
        self.landmarker.close()
