import cv2
import numpy as np

# Define HSV range
# lower_purple = np.array([0, 0, 200])
# upper_purple = np.array([10, 0, 255])
lower_purple = np.array([0, 0, 240])
upper_purple = np.array([50, 30, 255]) 

# Create a grid of HSV values within the range
hue_vals = np.linspace(lower_purple[0], upper_purple[0], 180).astype(np.uint8)
sat_vals = np.linspace(lower_purple[1], upper_purple[1], 100).astype(np.uint8)
val_vals = np.linspace(lower_purple[2], upper_purple[2], 100).astype(np.uint8)

# Initialize a color patch image
patch = np.zeros((100, 180, 3), dtype=np.uint8)

# Fill in each pixel with its HSV value
for i, s in enumerate(sat_vals):
    for j, h in enumerate(hue_vals):
        patch[i, j] = [h, s, val_vals[i]]  # HSV

# Convert to BGR for imshow
patch_bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)

color_patch = np.full((100, 100, 3), lower_purple, dtype=np.uint8)
color_patch_bgr = cv2.cvtColor(color_patch, cv2.COLOR_HSV2BGR)
cv2.imshow("Lower Bound Color", color_patch_bgr)

color_patch = np.full((100, 100, 3), upper_purple, dtype=np.uint8)
color_patch_bgr = cv2.cvtColor(color_patch, cv2.COLOR_HSV2BGR)
cv2.imshow("Upper Bound Color", color_patch_bgr)

cv2.imshow("Visualized HSV Range (Black-ish)", patch_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
