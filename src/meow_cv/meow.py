import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "models/face_landmarker_v2_with_blendshapes.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to hold the most recent results from the async callback
latest_landmarks = None


# Callback function to update global state
def update_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks
    if result.face_landmarks:
        # We only care about the first face detected
        latest_landmarks = result.face_landmarks[0]
    else:
        latest_landmarks = None


# Thresholds
eye_opening_threshold = 0.025
mouth_open_threshold = 0.03
squinting_threshold = 0.018


def cat_shock(face_landmarks):
    l_top = face_landmarks[159]
    l_bot = face_landmarks[145]
    r_top = face_landmarks[386]
    r_bot = face_landmarks[374]

    eye_opening = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
    return eye_opening > eye_opening_threshold


def cat_tongue(face_landmarks):
    top_lip = face_landmarks[13]
    bottom_lip = face_landmarks[14]

    mouth_open = abs(top_lip.y - bottom_lip.y)
    return mouth_open > mouth_open_threshold


def cat_glare(face_landmarks):
    l_top = face_landmarks[159]
    l_bot = face_landmarks[145]
    r_top = face_landmarks[386]
    r_bot = face_landmarks[374]

    eye_squint = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
    return eye_squint < squinting_threshold


def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=update_result,
    )

    cam = cv2.VideoCapture(0)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, image = cam.read()
            if not ret:
                break

            image = cv2.flip(image, 1)

            # Convert OpenCV BGR to RGB, then to MediaPipe Image format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # LIVE_STREAM requires a monotonically increasing timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)

            # Send the frame to MediaPipe (this runs asynchronously and triggers update_result)
            landmarker.detect_async(mp_image, timestamp_ms)

            # Default cat state
            cat_image = "assets/cat-shock.jpeg"

            # Use the global state updated by the callback
            if latest_landmarks:
                if cat_tongue(latest_landmarks):
                    cat_image = "assets/cat-tongue.jpeg"
                elif cat_shock(latest_landmarks):
                    cat_image = "assets/cat-shock.jpeg"
                elif cat_glare(latest_landmarks):
                    cat_image = "assets/cat-glare.jpeg"
                else:
                    cat_image = "assets/larry.jpeg"

                # Draw the landmark points
                height, width = image.shape[:2]
                for lm in latest_landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    cv2.circle(image, (x, y), 1, (0, 100, 0), -1)

            cv2.imshow("Face Detection", image)

            # Cat Display
            cat = cv2.imread(cat_image)
            if cat is not None:
                cat = cv2.resize(cat, (640, 480))
                cv2.imshow("Cat Image", cat)
            else:
                blank = image * 0
                cv2.putText(blank, f"Missing: {cat_image}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Cat Image", blank)

            # Press 'ESC' to exit
            key = cv2.waitKey(1)
            if key == 27:
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
