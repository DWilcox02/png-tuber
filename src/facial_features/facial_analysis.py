import time
from dataclasses import dataclass
from facial_features.facial_enums import EyeStatus, MouthStatus, SmileStatus
from constants import EYE_OPEN_THRESHOLD, EYE_WIDE_THRESHOLD, MOUTH_OPEN_THRESHOLD


@dataclass
class FacialStatus:
    """Container for discrete facial expression states."""

    left_eye: EyeStatus
    right_eye: EyeStatus
    mouth: MouthStatus
    smile: SmileStatus
    is_talking: bool = False


class FacialAnalyzer:
    """Stateful class to analyze blendshapes and animate facial states."""

    def __init__(self, flap_interval_sec=0.15):
        # Time in seconds between open/close toggles when simulating talking
        self.flap_interval_sec = flap_interval_sec
        self._last_flap_time = 0.0
        self._talking_mouth_state = MouthStatus.CLOSED

    @staticmethod
    def _calc_eye_status(blink_value: float) -> EyeStatus:
        if blink_value > EYE_OPEN_THRESHOLD:
            return EyeStatus.CLOSED
        if blink_value < EYE_WIDE_THRESHOLD:
            return EyeStatus.WIDE
        return EyeStatus.OPEN

    @staticmethod
    def _calc_smile_status(smile_value: float, frown_value: float) -> SmileStatus:
        if smile_value > 0.3:
            return SmileStatus.SMILING
        if frown_value > 0.3:
            return SmileStatus.FROWNING
        return SmileStatus.NEUTRAL

    def create_status(self, blendshapes, is_talking: bool = False) -> FacialStatus:
        """Extracts blendshapes, applies state machine logic, and returns a FacialStatus."""
        scores = {b.category_name: b.score for b in blendshapes} if blendshapes else {}

        def get(name):
            return scores.get(name, 0.0)

        avg_smile = (get("mouthSmileLeft") + get("mouthSmileRight")) / 2
        avg_frown = (get("mouthFrownLeft") + get("mouthFrownRight")) / 2

        # 1. Check if the user is physically dropping their jaw
        physical_jaw_open = get("jawOpen") > MOUTH_OPEN_THRESHOLD

        # 2. State Machine for Mouth Status
        current_time = time.time()

        if physical_jaw_open:
            # Physical jaw drop overrides everything. Keep the mouth explicitly open.
            mouth_state = MouthStatus.OPEN
            self._talking_mouth_state = MouthStatus.OPEN  # Reset flap memory

        elif is_talking:
            # User is generating audio, but jaw is shut. Simulate flapping.
            if (current_time - self._last_flap_time) > self.flap_interval_sec:
                self._last_flap_time = current_time
                # Toggle the internal state
                self._talking_mouth_state = (
                    MouthStatus.CLOSED if self._talking_mouth_state == MouthStatus.OPEN else MouthStatus.OPEN
                )
            mouth_state = self._talking_mouth_state

        else:
            # Not talking, jaw is closed.
            mouth_state = MouthStatus.CLOSED
            self._talking_mouth_state = MouthStatus.CLOSED

        return FacialStatus(
            left_eye=self._calc_eye_status(get("eyeBlinkLeft")),
            right_eye=self._calc_eye_status(get("eyeBlinkRight")),
            mouth=mouth_state,
            smile=self._calc_smile_status(avg_smile, avg_frown),
            is_talking=is_talking,
        )
