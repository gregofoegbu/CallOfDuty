import cv2
import easyocr
import matplotlib.pyplot as plt
import roi
import imagesplitter
import json
import getPlayers

image_path = 'TestImages/CodHDFullScreen.png'
output_file = 'output.txt'

img = cv2.imread(image_path);
killfeed = imagesplitter.get_clahe_img(imagesplitter.getimgregion(img, roi.kill_feed_bbox))
reader = easyocr.Reader(['en'], gpu=False)

text = reader.readtext(killfeed);

# with open(output_file, 'w', encoding='utf-8') as f:
#     for bbox, detected_text, confidence in text:
#         line = f"{detected_text} (confidence: {confidence:.2f})\n"
#         f.write(line)

# print("Text written to", output_file)

for t in text:
    bbox, _text, score = t
    cv2.rectangle(killfeed, bbox[0], bbox[2], (0, 255, 0), 5)
    cv2.putText(killfeed, _text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(killfeed, cv2.COLOR_BGR2RGB))
plt.show()


# for title, bbox in roi.all_regions_dict.items():
#     img_region = imagesplitter.getimgregion(img, bbox)
#     imagesplitter.show_processing_stages(img_region, title)


# plt.show()

# players_dict = getPlayers.get_players(img)
# with open("players.json", "w") as f:
#     json.dump(players_dict, f, indent=4)

