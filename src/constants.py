

# Model constants
MODEL_PATH = "models/face_landmarker_v2_with_blendshapes.task"

# Processing thresholds
EYE_OPENING_THRESHOLD = 0.025
MOUTH_OPEN_THRESHOLD = 0.03
SQUINTING_THRESHOLD = 0.018

# Landmark dictionary mapping descriptive names to MediaPipe indices
# These indices are specific to the FaceMesh model used by MediaPipe
LANDMARK_INDICES = {
    "LEFT_EYE_TOP": 159,
    "LEFT_EYE_BOTTOM": 145,
    "RIGHT_EYE_TOP": 386,
    "RIGHT_EYE_BOTTOM": 374,
    "UPPER_LIP": 13,
    "LOWER_LIP": 14
}