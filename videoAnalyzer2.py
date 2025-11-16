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
cap = cv2.VideoCapture('TrimmedClip.mov')

# Data structures for tracking:
player_tracks = {str(i): [] for i in range(1, 9)}
current_life = {str(i): [] for i in range(1, 9)}
previous_positions = {}  # Stores the last known position of each player

frame_count = 0
progress_interval = 100

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def detect_death_markers(minimap):
    minimap_edges = preprocess_image(minimap)
    for color, template in skull_templates.items():
        template_edges = preprocess_image(template)
        result = cv2.matchTemplate(minimap_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        threshold = 0.75
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            return pt
    return None

def get_closest_teammate(player_id, previous_positions):
    """ Find the closest teammate for the missing player """
    int_player_id = int(player_id);
    if int_player_id <= 4:
        teammates = [str(i) for i in range(1, 5)]
    else:
        teammates = [str(i) for i in range(5, 9)]

    closest_player = None
    min_distance = float("inf")
    
    for teammate in teammates:
        if teammate in previous_positions and teammate != player_id:
            x1, y1 = previous_positions[player_id]
            x2, y2 = previous_positions[teammate]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < min_distance:
                min_distance = distance
                closest_player = teammate

    return closest_player

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % progress_interval == 0:
        print(f"Processing frame {frame_count}")

    minimap = frame[map_y:map_y + map_height, map_x:map_x + map_width]
    death_position = detect_death_markers(minimap)
    
    if death_position:
        print(f"Death detected at {death_position}, ending player lives.")
        for pid in current_life:
            if current_life[pid]:
                player_tracks[pid].append(current_life[pid])
                current_life[pid] = []

    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_players = set()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = minimap[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 7').strip()
        
        if text in current_life:
            pos = (x + w // 2, y + h // 2)
            current_life[text].append(pos)
            previous_positions[text] = pos
            detected_players.add(text)

    # Check for missing players and assume they overlap with closest teammate
    for pid in current_life.keys():
        if pid not in detected_players and pid in previous_positions:
            closest_teammate = get_closest_teammate(pid, previous_positions)
            if closest_teammate:
                print(f"Player {pid} is missing. Assigning to closest teammate {closest_teammate}")
                current_life[pid].append(previous_positions[closest_teammate])

    cv2.imshow("Minimap", minimap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for pid, positions in current_life.items():
    if positions:
        player_tracks[pid].append(positions)

cap.release()
cv2.destroyAllWindows()

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
