import cv2
import numpy as np

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

    def show_cat(self, cat_image_path, fallback_image=None):
        cat = cv2.imread(cat_image_path)
        if cat is not None:
            cat = cv2.resize(cat, (640, 480))
            cv2.imshow("Cat Image", cat)
        elif fallback_image is not None:
            blank = np.zeros_like(fallback_image)
            cv2.putText(blank, f"Missing: {cat_image_path}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Cat Image", blank)

    def wait_esc(self, delay=1):
        key = cv2.waitKey(delay)
        return key == 27

    def terminate(self):
        self.cam.release()
        cv2.destroyAllWindows()
