import cv2

input_path = "video.mp4"
output_path = "multiple_rotations.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Duration to extract: 5 minutes 26 seconds = 5*60 + 26 = 326 seconds
extract_duration = 5*60 + 26  # 326 seconds
frames_to_extract = int(fps * extract_duration)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read the specified duration from the original video
extracted_frames = []
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        # If video ends before 5:26, break
        break
    extracted_frames.append(frame)
    frame_count += 1
    if frame_count >= frames_to_extract:
        break

# Now, cap might still have frames if video is longer. Close and reopen for full video
cap.release()

# Write first extracted portion (5:26) to the output
for f in extracted_frames:
    out.write(f)

# Reopen original video to write full video again
cap = cv2.VideoCapture(input_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()

print("Created video_doubled.mp4 with first 5:26 + full video appended.")
