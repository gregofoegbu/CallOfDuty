import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import pytesseract

# Define the minimap dimensions
map_x = 30       # Top-left corner x-coordinate
map_y = 550      # Top-left corner y-coordinate
map_width = 420  # Width of the map region
map_height = 300 # Height of the map region

# Load all skull templates
skull_templates = {}
template_folder = "skull_templates"

for filename in os.listdir(template_folder):
    if filename.endswith(".png"):
        color = filename.split("_")[-1].replace(".png", "")  # Extract color from filename
        skull_templates[color] = cv2.imread(os.path.join(template_folder, filename), cv2.IMREAD_UNCHANGED)

# Open your video stream or capture file
cap = cv2.VideoCapture('CODControlRecording.mov')

# Data structures for tracking:
# player_tracks stores a list of lives (each life is a list of (x, y) positions)
player_tracks = {str(i): [] for i in range(1, 9)}
# current_life holds the positions for the ongoing life for each player
current_life = {str(i): [] for i in range(1, 9)}

frame_count = 0
progress_interval = 100  # print progress every 100 frames

def preprocess_image(image):
    """ Convert to grayscale and apply edge detection """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Edge detection
    return edges

def detect_death_markers(minimap):
    """Detects skulls in the minimap"""
    minimap_edges = preprocess_image(minimap)
    
    for color, template in skull_templates.items():
        template_edges = preprocess_image(template)
        result = cv2.matchTemplate(minimap_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        threshold = 0.75  # Adjust if needed
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            return pt  # Return first detected skull position
    return None  # No skull detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % progress_interval == 0:
        print(f"Processing frame {frame_count}")

    # Crop the minimap region from the frame
    minimap = frame[map_y:map_y + map_height, map_x:map_x + map_width]

    # Detect deaths
    death_position = detect_death_markers(minimap)
    if death_position:
        print(f"Death detected at {death_position}, ending player lives.")

        # End all ongoing lives
        for pid in current_life:
            if current_life[pid]:  # If the player had movement data
                player_tracks[pid].append(current_life[pid])
                current_life[pid] = []  # Reset life for new spawn

    # Convert the minimap to HSV for segmentation; adjust thresholds as needed
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find contours of arrow (or number) regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:  # filter out noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = minimap[y:y+h, x:x+w]
        
        # Use OCR to read the overlaid text
        text = pytesseract.image_to_string(roi, config='--psm 7').strip()
        
        # If the detected text is a number, assume it's a player icon
        if text in current_life:
            pos = (x + w // 2, y + h // 2)  # position relative to the minimap
            current_life[text].append(pos)
    
    # (Optional) display the minimap region for debugging purposes
    cv2.imshow("Minimap", minimap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalize any remaining lives (if the life did not end with a skull)
for pid, positions in current_life.items():
    if positions:
        player_tracks[pid].append(positions)

cap.release()
cv2.destroyAllWindows()

# Save the tracking data to a JSON file for later plotting
with open("player_tracks.json", "w") as f:
    json.dump(player_tracks, f)
print("Tracking data saved to player_tracks.json")

# --------------------- Plotting the Results ---------------------
plt.figure(figsize=(8, 8))

for pid, lives in player_tracks.items():
    pid_int = int(pid)
    cmap = plt.cm.Reds if pid_int <= 4 else plt.cm.Blues
    
    num_lives = len(lives)
    for i, life_coords in enumerate(lives):
        if len(life_coords) < 2:
            continue
        xs, ys = zip(*life_coords)
        color = cmap((i + 1) / (num_lives + 1))
        plt.plot(xs, ys, color=color, label=f'Player {pid} Life {i+1}')

plt.gca().set_facecolor('white')
plt.title("Player Movements on the Minimap")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.show()
