import inspect

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import _RunningMode

print(mp.__version__, "\n\n")
print(inspect.signature(vision.FaceLandmarkerOptions), "\n\n")

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1,
                                       running_mode=_RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("media/image.png")

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)

print(detection_result.facial_transformation_matrixes)
