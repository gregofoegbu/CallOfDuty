import cv2
import os

video_path = "../../vods/VOD-2025-11-16.mov"
output_dir = "../../ExtractedFrames/VOD-2025-11-16-Frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_id % 5 == 0:   # keep every 5th frame
        out_path = f"{output_dir}/{count:06d}.jpg"
        cv2.imwrite(out_path, frame)
        count += 1

    frame_id += 1

cap.release()
print("Done. Extracted:", count)
