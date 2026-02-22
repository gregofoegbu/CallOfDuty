import json

import cv2
import utils  # assuming this has check_game_start
import getPlayers  # your module for get_players
import imagesplitter
import roi
import easyocr

# Main variables
killfeed_all = []   # {killer: victim}
gameActive = False
reader = easyocr.Reader(['en'], gpu=False)
cap = cv2.VideoCapture('../Videos/TrimmedSnDOpticvMinnesotaMajorIIQW1D2.mp4')
if not cap.isOpened():
    print("Error opening video")
    exit()

frame_idx = 0
frame_step = 1  # start by checking every frame

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        # End of video
        break

    # Check if game has started on this frame
    if utils.check_game_start(frame, reader):
        if not gameActive:
            # Game just started, initialize players
            print("Team found!")
            gameActive = True
            player_dict = getPlayers.get_players(frame, reader)
            all_players = [player for team in player_dict.values() for player in team]
            print("Game started! Players:", player_dict)
        frame_step = 3
        killfeed_clahe_img = imagesplitter.get_clahe_img(imagesplitter.getimgregion(frame, roi.kill_feed_bbox))

        killfeed_ocr_text = reader.readtext(killfeed_clahe_img);
        kills_this_frame = utils.get_killfeed_relations(killfeed_ocr_text, all_players)

        # Add only new kills to main killfeed
        for killer, victim in kills_this_frame.items():
            event = (killer, victim)

            if event not in killfeed_all:
                killfeed_all.append(event)
                print(f"New kill: {killer} → {victim}")

    else:
        if gameActive:
            # Game has ended
            print("Game ended!")
            break

    frame_idx += frame_step
cap.release()
cv2.destroyAllWindows()

# Optionally print the full killfeed at the end
print("Final killfeed:", killfeed_all)

with open("players.json", "w") as f:
    json.dump(player_dict, f, indent=4)
    json.dump(killfeed_all, f, indent=4)