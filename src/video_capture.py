import cv2
import numpy as np
from facial_features.facial_analysis import FacialStatus

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

    def show_cat(self, cat_image_path, facial_status: FacialStatus = None):
        cat = cv2.imread(cat_image_path)
        if cat is not None:
            cat = cv2.resize(cat, (640, 480))
        else:
            cat = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(cat, f"Missing: {cat_image_path}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if facial_status is not None:
            # Display statuses
            status_lines = [
                f"Left Eye: {facial_status.left_eye.value}",
                f"Right Eye: {facial_status.right_eye.value}",
                f"Mouth: {facial_status.mouth.value}",
                f"Smile: {facial_status.smile.value}"
            ]

            for i, text in enumerate(status_lines):
                y_pos = 35 + (i * 35)
                # Shadowed text for visibility
                cv2.putText(cat, text, (12, y_pos + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(cat, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("Cat Image", cat)

    def wait_esc(self, delay=1):
        key = cv2.waitKey(delay)
        return key == 27

    def terminate(self):
        self.cam.release()
        cv2.destroyAllWindows()
