import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import pytesseract
from sklearn.cluster import DBSCAN

# Define the minimap dimensions
map_x = 30       # Top-left corner x-coordinate
map_y = 550      # Top-left corner y-coordinate
map_width = 420  # Width of the map region
map_height = 300 # Height of the map region

# Load all skull templates (as before)
skull_templates = {}
template_folder = "skull_templates"
for filename in os.listdir(template_folder):
    if filename.endswith(".png"):
        color = filename.split("_")[-1].replace(".png", "")
        skull_templates[color] = cv2.imread(os.path.join(template_folder, filename), cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture('TrimmedTrim.mov')

# Tracking structures
player_tracks = {str(i): [] for i in range(1, 9)}
current_life = {str(i): [] for i in range(1, 9)}
previous_positions = {}  # Last known positions (for lookahead)

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
            return pt  # Return first detected skull position
    return None

def compute_hu_moments(cnt):
    moments = cv2.moments(cnt)
    huMoments = cv2.HuMoments(moments).flatten()
    # Log-scale transform for comparison
    huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-7)
    return huMoments

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
        print(f"Death detected at {death_position}. Ending current lives.")
        for pid in current_life:
            if current_life[pid]:
                player_tracks[pid].append(current_life[pid])
                current_life[pid] = []  # Reset for new life

    # Convert minimap to HSV and threshold (for arrow detection)
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold detections: each detection is a tuple (player_id, centroid, hu_moments)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = minimap[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 7').strip()

        # If OCR returns a player number, treat it as a clean detection
        if text in current_life:
            centroid = (x + w // 2, y + h // 2)
            detections.append((text, centroid, None))
            current_life[text].append(centroid)
            previous_positions[text] = centroid
        else:
            # Otherwise, compute a shape descriptor (Hu moments) for later clustering
            centroid = (x + w // 2, y + h // 2)
            hu = compute_hu_moments(cnt)
            detections.append((None, centroid, hu))
    
    # Separate clean detections (with player id) from uncertain ones (overlapping detections)
    clean_detections = [d for d in detections if d[0] is not None]
    uncertain_detections = [d for d in detections if d[0] is None]

    # For uncertain detections, perform clustering based on position and shape descriptor
    if uncertain_detections:
        # Construct feature vectors: [x, y, hu1, hu2, ...] for each detection
        features = []
        for _, centroid, hu in uncertain_detections:
            # Use only the spatial coordinates for clustering if shape descriptor is not reliable,
            # or combine them (here we use spatial data only for simplicity)
            features.append([centroid[0], centroid[1]])
        features = np.array(features)
        
        # Use DBSCAN clustering to group nearby detections
        clustering = DBSCAN(eps=15, min_samples=1).fit(features)
        labels = clustering.labels_
        
        # For each cluster, assign a player id based on lookahead:
        # For each missing player on a team, choose the detection cluster that is closest
        # to the player's previous position.
        for team in [list(map(str, range(1,5))), list(map(str, range(5,9)))]:
            # Find players on the team that did not have a clean detection this frame
            missing = [pid for pid in team if pid not in [d[0] for d in clean_detections]]
            for pid in missing:
                if pid in previous_positions:
                    prev = np.array(previous_positions[pid])
                    # For each cluster, compute the cluster centroid
                    unique_labels = np.unique(labels)
                    best_cluster = None
                    min_dist = float("inf")
                    for label in unique_labels:
                        indices = np.where(labels == label)[0]
                        cluster_pts = features[indices]
                        cluster_centroid = np.mean(cluster_pts, axis=0)
                        dist_val = np.linalg.norm(prev - cluster_centroid)
                        if dist_val < min_dist:
                            min_dist = dist_val
                            best_cluster = cluster_centroid
                    if best_cluster is not None:
                        # Assume this missing player's position is the cluster centroid
                        pos = (int(best_cluster[0]), int(best_cluster[1]))
                        current_life[pid].append(pos)
                        previous_positions[pid] = pos
                        print(f"Missing player {pid} assigned overlapping detection at {pos}")

    cv2.imshow("Minimap", minimap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalize remaining lives
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
