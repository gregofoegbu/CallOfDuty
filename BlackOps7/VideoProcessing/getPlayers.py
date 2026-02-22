import roi
import cv2
import easyocr
import imagesplitter

allowList = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

def get_players(img):
    t1_name_str: str
    t2_name_str: str
    players_team1 = []
    players_team2 = []
    
    reader = easyocr.Reader(['en'], gpu=False)
    team_one_name_img = imagesplitter.getimgregion(img, roi.team_one_name_bbox)
    team_two_name_img = imagesplitter.getimgregion(img, roi.team_two_name_bbox)
    team_one_players_img = imagesplitter.getimgregion(img, roi.players_team_one)
    team_two_players_img = imagesplitter.getimgregion(img, roi.players_team_two)
    
    t1_name = reader.readtext(team_one_name_img, allowlist=allowList)
    t2_name = reader.readtext(team_two_name_img, allowlist=allowList)
    t1_players = reader.readtext(team_one_players_img, allowlist=allowList)
    t2_players = reader.readtext(team_two_players_img, allowlist=allowList)
    
    if t1_name:
    # result structure: [bounding box, text, confidence]
        t1_name_str = t1_name[0][1] 
    if t2_name:
    # result structure: [bounding box, text, confidence]
        t2_name_str = t2_name[0][1] 
    for t in t1_players:
        bbox, _text, score = t
        players_team1.append(_text);
    for t in t2_players:
        bbox, _text, score = t
        players_team2.append(_text);
    
    return {t1_name_str: players_team1, t2_name_str: players_team2}