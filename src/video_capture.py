import cv2
import numpy as np
from facial_features.facial_analysis import FacialStatus


class VideoCapture:
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
        # 1. Read the image WITH the alpha channel (transparency) using IMREAD_UNCHANGED
        cat = cv2.imread(cat_image_path, cv2.IMREAD_UNCHANGED)

        # 2. Create a solid green background (perfect for Chroma Keying in OBS)
        # BGR format: (0, 255, 0) is pure green
        display_img = np.full((480, 640, 3), (0, 255, 0), dtype=np.uint8)

        if cat is not None:
            cat = cv2.resize(cat, (640, 480))

            # 3. Check if the image has an alpha channel (4 channels)
            if len(cat.shape) == 3 and cat.shape[2] == 4:
                # Extract alpha channel and normalize it to 0.0 - 1.0
                alpha = cat[:, :, 3] / 255.0

                # Blend each color channel (B, G, R) using the alpha mask
                for c in range(3):
                    display_img[:, :, c] = (alpha * cat[:, :, c] + (1.0 - alpha) * display_img[:, :, c]).astype(
                        np.uint8
                    )
            else:
                # If no transparency is found, just use the image directly
                display_img = cat
        else:
            cv2.putText(
                display_img, f"Missing: {cat_image_path}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        if facial_status is not None:
            # Display statuses
            status_lines = [
                f"Left Eye: {facial_status.left_eye.value}",
                f"Right Eye: {facial_status.right_eye.value}",
                f"Mouth: {facial_status.mouth.value}",
                f"Smile: {facial_status.smile.value}",
            ]

            for i, text in enumerate(status_lines):
                y_pos = 35 + (i * 35)
                # Shadowed text for visibility
                cv2.putText(display_img, text, (12, y_pos + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(display_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("Cat Image", display_img)

    def wait_esc(self, delay=1):
        key = cv2.waitKey(delay)
        return key == 27

    def terminate(self):
        self.cam.release()
        cv2.destroyAllWindows()
