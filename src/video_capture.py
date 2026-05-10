import cv2
import numpy as np
from facial_features.facial_analysis import calc_eye_status

class VideoCapture():
    def __init__(self, camera_index=0):
        self.cam = cv2.VideoCapture(camera_index)

    def read_frame(self):
        ret, image = self.cam.read()
        if not ret:
            return None
        return cv2.flip(image, 1)
    
    def to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def draw_landmarks(self, image, landmarks):
        if landmarks:
            height, width = image.shape[:2]
            for lm in landmarks:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(image, (x, y), 1, (0, 100, 0), -1)

    def show_frame(self, image, title="Face Detection"):
        cv2.imshow(title, image)

    def show_cat(self, cat_image_path, fallback_image=None, blendshapes=None):
        cat = cv2.imread(cat_image_path)
        if cat is not None:
            cat = cv2.resize(cat, (640, 480))
        elif fallback_image is not None:
            cat = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(cat, f"Missing: {cat_image_path}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if cat is not None and blendshapes is not None:
            # Extract blink scores
            left_blink = 0.0
            right_blink = 0.0
            for category in blendshapes:
                if category.category_name == "eyeBlinkLeft":
                    left_blink = category.score
                elif category.category_name == "eyeBlinkRight":
                    right_blink = category.score

            # Calculate statuses
            left_status = calc_eye_status(left_blink)
            right_status = calc_eye_status(right_blink)

            # Display statuses
            left_text = f"Left Eye: {left_status.value}"
            right_text = f"Right Eye: {right_status.value}"

            # Shadowed text for visibility
            cv2.putText(cat, left_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(cat, left_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.putText(cat, right_text, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(cat, right_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if cat is not None:
            cv2.imshow("Cat Image", cat)

    def wait_esc(self, delay=1):
        key = cv2.waitKey(delay)
        return key == 27

    def terminate(self):
        self.cam.release()
        cv2.destroyAllWindows()
