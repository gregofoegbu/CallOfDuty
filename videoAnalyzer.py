import cv2
import numpy as np

# Define the purple HSV range
lower_purple = np.array([0, 0, 240])
upper_purple = np.array([180, 30, 255]) 

# Parameters
width = 160  # Hue variation
height = 100  # Saturation/Value variation
font_scale = 0.3
font_thickness = 1
step_h = (upper_purple[0] - lower_purple[0]) / width
step_sv = (upper_purple[1] - lower_purple[1]) / height

# Create a blank HSV image
hsv_patch = np.zeros((height, width, 3), dtype=np.uint8)

# Fill HSV values
for x in range(width):
    h = int(lower_purple[0] + step_h * x)
    for y in range(height):
        s = int(lower_purple[1] + step_sv * y)
        v = int(lower_purple[2] + step_sv * y)
        hsv_patch[y, x] = [h, s, v]

# Convert to BGR for display
bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)

# Convert to color image (to draw text)
overlay = bgr_patch.copy()

# Overlay hue and value labels
for x in range(0, width, 20):
    h = int(lower_purple[0] + step_h * x)
    cv2.putText(overlay, f'H:{h}', (x, 10), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)

for y in range(0, height, 20):
    s = int(lower_purple[1] + step_sv * y)
    v = int(lower_purple[2] + step_sv * y)
    cv2.putText(overlay, f'S:{s}', (2, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)
    cv2.putText(overlay, f'V:{v}', (60, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)

cv2.imshow("Purple HSV Range with Labels", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
