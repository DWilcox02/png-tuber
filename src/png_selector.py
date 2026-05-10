from utils import EYE_OPENING_THRESHOLD, MOUTH_OPEN_THRESHOLD, SQUINTING_THRESHOLD, LANDMARK_INDICES
from facial_features.facial_enums import EyeStatus, MouthStatus, SmileStatus
from facial_features.facial_analysis import FacialStatus, calc_eye_status, calc_mouth_status, calc_smile_status

class PngSelector():
    def __init__(self):
        # Thresholds
        self.eye_opening_threshold = EYE_OPENING_THRESHOLD
        self.mouth_open_threshold = MOUTH_OPEN_THRESHOLD
        self.squinting_threshold = SQUINTING_THRESHOLD

    def get_blendshape_score(self, blendshapes, name):
        if not blendshapes:
            return 0.0
        for category in blendshapes:
            if category.category_name == name:
                return category.score
        return 0.0

    def get_cat_image(self, latest_landmarks, latest_blendshapes=None):
        # Default state
        cat_image = "assets/larry.jpeg"
        
        # Calculate statuses
        left_blink = self.get_blendshape_score(latest_blendshapes, "eyeBlinkLeft")
        right_blink = self.get_blendshape_score(latest_blendshapes, "eyeBlinkRight")
        jaw_open = self.get_blendshape_score(latest_blendshapes, "jawOpen")
        smile_l = self.get_blendshape_score(latest_blendshapes, "mouthSmileLeft")
        smile_r = self.get_blendshape_score(latest_blendshapes, "mouthSmileRight")
        frown_l = self.get_blendshape_score(latest_blendshapes, "mouthFrownLeft")
        frown_r = self.get_blendshape_score(latest_blendshapes, "mouthFrownRight")

        status = FacialStatus(
            left_eye=calc_eye_status(left_blink),
            right_eye=calc_eye_status(right_blink),
            mouth=calc_mouth_status(jaw_open),
            smile=calc_smile_status((smile_l + smile_r) / 2, (frown_l + frown_r) / 2)
        )

        # Logic for image selection
        if latest_landmarks:
            if status.mouth == MouthStatus.OPEN:
                cat_image = "assets/cat-tongue.jpeg"
            elif status.left_eye == EyeStatus.WIDE or status.right_eye == EyeStatus.WIDE:
                cat_image = "assets/cat-shock.jpeg"
            elif status.left_eye == EyeStatus.CLOSED or status.right_eye == EyeStatus.CLOSED:
                cat_image = "assets/cat-glare.jpeg"

        return cat_image, status

    def terminate(self):
        pass
