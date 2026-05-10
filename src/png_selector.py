from utils import EYE_OPENING_THRESHOLD, MOUTH_OPEN_THRESHOLD, SQUINTING_THRESHOLD, LANDMARK_INDICES

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

    def cat_shock(self, face_landmarks, blendshapes=None):
        if blendshapes:
            # Using blendshapes: sum of eye blink scores (0 is fully open, 1 is fully closed)
            # So a small score means eyes are wide open
            l_blink = self.get_blendshape_score(blendshapes, "eyeBlinkLeft")
            r_blink = self.get_blendshape_score(blendshapes, "eyeBlinkRight")
            return (l_blink + r_blink) < 0.1 # Very open eyes
        
        # Fallback to landmarks
        l_top = face_landmarks[LANDMARK_INDICES["LEFT_EYE_TOP"]]
        l_bot = face_landmarks[LANDMARK_INDICES["LEFT_EYE_BOTTOM"]]
        r_top = face_landmarks[LANDMARK_INDICES["RIGHT_EYE_TOP"]]
        r_bot = face_landmarks[LANDMARK_INDICES["RIGHT_EYE_BOTTOM"]]

        eye_opening = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
        return eye_opening > self.eye_opening_threshold

    def cat_tongue(self, face_landmarks, blendshapes=None):
        if blendshapes:
            # Using blendshapes for mouth open
            return self.get_blendshape_score(blendshapes, "jawOpen") > 0.4
            
        # Fallback to landmarks
        top_lip = face_landmarks[LANDMARK_INDICES["UPPER_LIP"]]
        bottom_lip = face_landmarks[LANDMARK_INDICES["LOWER_LIP"]]

        mouth_open = abs(top_lip.y - bottom_lip.y)
        return mouth_open > self.mouth_open_threshold

    def cat_glare(self, face_landmarks, blendshapes=None):
        if blendshapes:
            # Using blendshapes: high blink score means squinting/closed
            l_blink = self.get_blendshape_score(blendshapes, "eyeBlinkLeft")
            r_blink = self.get_blendshape_score(blendshapes, "eyeBlinkRight")
            return (l_blink + r_blink) > 1.2 # Squinting
            
        # Fallback to landmarks
        l_top = face_landmarks[LANDMARK_INDICES["LEFT_EYE_TOP"]]
        l_bot = face_landmarks[LANDMARK_INDICES["LEFT_EYE_BOTTOM"]]
        r_top = face_landmarks[LANDMARK_INDICES["RIGHT_EYE_TOP"]]
        r_bot = face_landmarks[LANDMARK_INDICES["RIGHT_EYE_BOTTOM"]]

        eye_squint = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
        return eye_squint < self.squinting_threshold

    def get_cat_image(self, latest_landmarks, latest_blendshapes=None):
        # Default cat state
        cat_image = "assets/larry.jpeg"

        if latest_landmarks:
            if self.cat_tongue(latest_landmarks, latest_blendshapes):
                cat_image = "assets/cat-tongue.jpeg"
            elif self.cat_shock(latest_landmarks, latest_blendshapes):
                cat_image = "assets/cat-shock.jpeg"
            elif self.cat_glare(latest_landmarks, latest_blendshapes):
                cat_image = "assets/cat-glare.jpeg"

        return cat_image

    def terminate(self):
        pass
