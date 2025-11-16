import json
import matplotlib.pyplot as plt

# Load the tracking data from the JSON file
with open('player_tracks.json', 'r') as f:
    player_tracks = json.load(f)

plt.figure(figsize=(8, 8))

# Define 8 distinct colors (these colors come from a qualitative palette)
# You can adjust these hex codes to any colors that stand out to you.
player_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                 "#ff7f00", "#ffff33", "#a65628", "#f781bf"]

# Loop over each player (assuming player ids are "1" to "8")
for pid, lives in player_tracks.items():
    pid_int = int(pid)
    # Get the color for this player from the list (player id 1 maps to index 0, etc.)
    player_color = player_colors[pid_int - 1]

    # Plot each life for this player using the assigned color
    for i, life in enumerate(lives):
        if len(life) < 2:
            continue  # Need at least two points to draw a line
        xs, ys = zip(*life)
        plt.plot(xs, ys, color=player_color, label=f'Player {pid} Life {i+1}')
        # Mark the start point with a circle
        plt.scatter(xs[0], ys[0], color=player_color, marker='o', s=100,
                    edgecolors='black', zorder=5)
        # Mark the end point with an 'X'
        plt.scatter(xs[-1], ys[-1], color=player_color, marker='X', s=100,
                    edgecolors='black', zorder=5)

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Player Movements with Distinct Colors per Player')
plt.gca().set_facecolor('white')
plt.legend()
plt.show()
