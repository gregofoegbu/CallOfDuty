import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("../../my_model/my_model.pt")

# Minimap crop coordinates
map_x, map_y = 30, 535
map_width, map_height = 420, 280

# Input video
video_path = "../../vods/VOD-2025-11-16.mov"
cap = cv2.VideoCapture(video_path)

# Video writer for cropped output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(
    "output_cropped.mp4",
    fourcc,
    fps,
    (map_width, map_height) 
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the region of interest (minimap)
    roi = frame[map_y : map_y + map_height, map_x : map_x + map_width]

    # Run YOLO detection on cropped area
    results = model(roi, imgsz=1024, conf=0.3)

    # Draw detections on cropped frame
    annotated = results[0].plot()

    # Write annotated cropped frame to output video
    out.write(annotated)

    # Optional: live display
    cv2.imshow("Cropped YOLO Output", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done! Saved output_cropped.mp4")
