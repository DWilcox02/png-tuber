from dataclasses import dataclass
from facial_features.facial_enums import EyeStatus, MouthStatus, SmileStatus
from utils import EYE_OPENING_THRESHOLD, MOUTH_OPEN_THRESHOLD, SQUINTING_THRESHOLD, LANDMARK_INDICES

@dataclass
class FacialStatus:
    left_eye: EyeStatus
    right_eye: EyeStatus
    mouth: MouthStatus
    smile: SmileStatus

def calc_eye_status(blink_value):
    if (blink_value > 0.5): 
        return EyeStatus.CLOSED
    if (blink_value < 0.03): 
        return EyeStatus.WIDE
    return EyeStatus.OPEN

def calc_mouth_status(jaw_open_value):
    if (jaw_open_value > 0.3): 
        return MouthStatus.OPEN
    return MouthStatus.CLOSED

def calc_smile_status(smile_value, frown_value):
    if (smile_value > 0.3):
        return SmileStatus.SMILING
    if (frown_value > 0.2):
        return SmileStatus.FROWNING
    return SmileStatus.NEUTRAL