from facial_features.facial_enums import EyeStatus, MouthStatus, SmileStatus
from utils import EYE_OPENING_THRESHOLD, MOUTH_OPEN_THRESHOLD, SQUINTING_THRESHOLD, LANDMARK_INDICES

def calc_eye_status(blink_value):
    if (blink_value > 0.55): 
        return EyeStatus.CLOSED
    if (blink_value < 0.05): 
        return EyeStatus.WIDE
    return EyeStatus.OPEN