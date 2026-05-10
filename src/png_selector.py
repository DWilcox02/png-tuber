from utils import EYE_OPENING_THRESHOLD, MOUTH_OPEN_THRESHOLD, SQUINTING_THRESHOLD

class PngSelector():
    def __init__(self):
        # Thresholds
        self.eye_opening_threshold = EYE_OPENING_THRESHOLD
        self.mouth_open_threshold = MOUTH_OPEN_THRESHOLD
        self.squinting_threshold = SQUINTING_THRESHOLD

    def cat_shock(self, face_landmarks):
        l_top = face_landmarks[159]
        l_bot = face_landmarks[145]
        r_top = face_landmarks[386]
        r_bot = face_landmarks[374]

        eye_opening = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
        return eye_opening > self.eye_opening_threshold

    def cat_tongue(self, face_landmarks):
        top_lip = face_landmarks[13]
        bottom_lip = face_landmarks[14]

        mouth_open = abs(top_lip.y - bottom_lip.y)
        return mouth_open > self.mouth_open_threshold

    def cat_glare(self, face_landmarks):
        l_top = face_landmarks[159]
        l_bot = face_landmarks[145]
        r_top = face_landmarks[386]
        r_bot = face_landmarks[374]

        eye_squint = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
        return eye_squint < self.squinting_threshold

    def get_cat_image(self, latest_landmarks):
        # Default cat state
        cat_image = "assets/cat-shock.jpeg"

        if latest_landmarks:
            if self.cat_tongue(latest_landmarks):
                cat_image = "assets/cat-tongue.jpeg"
            elif self.cat_shock(latest_landmarks):
                cat_image = "assets/cat-shock.jpeg"
            elif self.cat_glare(latest_landmarks):
                cat_image = "assets/cat-glare.jpeg"
            else:
                cat_image = "assets/larry.jpeg"

        return cat_image

    def terminate(self):
        pass
