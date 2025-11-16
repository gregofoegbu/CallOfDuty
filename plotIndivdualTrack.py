import json
import matplotlib.pyplot as plt

# Load the tracking data from the JSON file
with open('player_tracks.json', 'r') as f:
    player_tracks = json.load(f)

# Define 8 distinct colors (these colors come from a qualitative palette)
player_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                 "#ff7f00", "#ffff33", "#a65628", "#f781bf"]

# Loop over each player and create a separate window for each one
for pid, lives in player_tracks.items():
    pid_int = int(pid)
    player_color = player_colors[pid_int - 1]  # Assign color based on player ID

    # Create a new figure for each player
    plt.figure(figsize=(8, 8))
    plt.title(f"Player {pid} Movements")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.gca().set_facecolor("white")

    # Plot each life for this player
    for i, life in enumerate(lives):
        if len(life) < 2:
            continue  # Need at least two points to draw a line
        xs, ys = zip(*life)
        plt.plot(xs, ys, color=player_color, label=f'Life {i+1}')
        plt.scatter(xs[0], ys[0], color=player_color, marker='o', s=100,
                    edgecolors='black', zorder=5)  # Start point
        plt.scatter(xs[-1], ys[-1], color=player_color, marker='X', s=100,
                    edgecolors='black', zorder=5)  # End point

    plt.legend()
    plt.show(block=False)  # Show without blocking execution

# Add a pause to ensure all windows are displayed
plt.pause(0.1)

# Keep the windows open until the user closes them
plt.show()
