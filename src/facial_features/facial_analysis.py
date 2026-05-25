from dataclasses import dataclass
from typing import List, Optional
from facial_features.facial_enums import EyeStatus, MouthStatus, SmileStatus
from constants import (
    EYE_OPEN_THRESHOLD, 
    EYE_WIDE_THRESHOLD,
    MOUTH_OPEN_THRESHOLD
)

@dataclass
class FacialStatus:
    """Container for discrete facial expression states."""
    left_eye: EyeStatus
    right_eye: EyeStatus
    mouth: MouthStatus
    smile: SmileStatus

class FacialStatusFactory:
    """Factory class to create FacialStatus objects from blendshape data."""

    @staticmethod
    def _calc_eye_status(blink_value: float) -> EyeStatus:
        if blink_value > EYE_OPEN_THRESHOLD:
            return EyeStatus.CLOSED
        if blink_value < EYE_WIDE_THRESHOLD:
            return EyeStatus.WIDE
        return EyeStatus.OPEN

    @staticmethod
    def _calc_mouth_status(jaw_open_value: float) -> MouthStatus:
        return MouthStatus.OPEN if jaw_open_value > MOUTH_OPEN_THRESHOLD else MouthStatus.CLOSED

    @staticmethod
    def _calc_smile_status(smile_value: float, frown_value: float) -> SmileStatus:
        if smile_value > 0.3:
            return SmileStatus.SMILING
        if frown_value > 0.3:
            return SmileStatus.FROWNING
        return SmileStatus.NEUTRAL

    @classmethod
    def create(cls, blendshapes) -> FacialStatus:
        """
        Extracts relevant scores from MediaPipe blendshapes and returns
        a structured FacialStatus object.
        """
        scores = {b.category_name: b.score for b in blendshapes} if blendshapes else {}
        get = lambda name: scores.get(name, 0.0)

        # Average smile and frown scores across both sides
        avg_smile = (get("mouthSmileLeft") + get("mouthSmileRight")) / 2
        avg_frown = (get("mouthFrownLeft") + get("mouthFrownRight")) / 2

        return FacialStatus(
            left_eye=cls._calc_eye_status(get("eyeBlinkLeft")),
            right_eye=cls._calc_eye_status(get("eyeBlinkRight")),
            mouth=cls._calc_mouth_status(get("jawOpen")),
            smile=cls._calc_smile_status(avg_smile, avg_frown)
        )
