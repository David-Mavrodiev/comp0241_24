import cv2
import numpy as np

# Path to the video
video_path = "bottom_view.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Unable to read video.")

# Let the user select the ROI manually
print("Select the ROI on the first frame. Press ENTER or SPACE when done.")
roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = [int(v) for v in roi]
if w == 0 or h == 0:
    raise ValueError("No valid ROI selected.")

print(x, y, w, h)