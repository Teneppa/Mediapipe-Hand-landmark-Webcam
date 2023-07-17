""" A modified example for using a Webcam as the source

https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

# Create VideoCapture object for webcam feed
cam = cv2.VideoCapture(0)

# This part is form site-packages/mediapipe/python/image_test.py
from mediapipe.python._framework_bindings import image_frame
from mediapipe.python._framework_bindings import image

# This part is form site-packages/mediapipe/python/image_test.py
ImageFormat = image_frame.ImageFormat
Image = image.Image

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

while True:
    success, frame = cam.read()
    if success:
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # This part is form site-packages/mediapipe/python/image_test.py
        mat = frame.astype(np.uint8)
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        rgb_image = Image(image_format=ImageFormat.SRGB, data=mat)

        # Detect hand landmarks from the input image.
        detection_result = detector.detect(rgb_image)

        # Process and visualize
        annotated_image = draw_landmarks_on_image(rgb_image.numpy_view(), detection_result)
        cv2.imshow("Window", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.waitKey(0)
