# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import _RunningMode

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1,
                                       running_mode=_RunningMode.VIDEO)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Open the input video.
cap = cv2.VideoCapture("media/video.mp4")

# CAP_PROP = capture property
fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0

# STEP 4: Detect face landmarks frame by frame.
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # OpenCV gives BGR, MediaPipe expects RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int(frame_index * (1000 / fps))

    detection_result = detector.detect_for_video(image, timestamp_ms)

    print(detection_result.facial_transformation_matrixes)

    frame_index += 1

cap.release()
