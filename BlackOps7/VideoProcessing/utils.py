import imagesplitter
import easyocr
import roi
#Get frame from video at 54 seconds
# video_path = "../Videos/OpticvMinnesotaMajorIIQW1D2.mp4"
# cap = cv2.VideoCapture(video_path)

# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_id = int(54 * fps)

# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
# ret, frame = cap.read()

# if ret:
#     cv2.imwrite("sample_54s.jpg", frame)
#     print("Frame saved!")
# else:
#     print("Failed to read frame.")

# cap.release()

cdl_teams = ["BREACH", "RAVENS", "MINNESOTA", "THIEVES", "HERETICS", "TEXAS", "VEGAS", "NEWYORK"]

def get_killfeed_relations(ocr_results, player_list, line_tolerance=5):
    """
    Given OCR results from a Killfeed, determine who killed whom.
    
    Args:
        ocr_results: List of tuples from EasyOCR
                     [(bbox, text, confidence), ...]
                     bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        line_tolerance: Max vertical difference to consider names on the same line
    
    Returns:
        Dictionary: {killer_name: victim_name, ...}
    """
    from collections import defaultdict

    def bbox_center_y(bbox):
        return sum([pt[1] for pt in bbox]) / 4

    def bbox_center_x(bbox):
        return sum([pt[0] for pt in bbox]) / 4

    # Step 1: Group by line
    lines = defaultdict(list)  # key = representative y, value = list of (text, bbox)
    for bbox, text, conf in ocr_results:
        y_center = bbox_center_y(bbox)
        # try to find an existing line within tolerance
        matched = False
        for line_y in lines:
            if abs(y_center - line_y) <= line_tolerance:
                lines[line_y].append((text, bbox))
                matched = True
                break
        if not matched:
            lines[y_center].append((text, bbox))

    # Step 2: Sort horizontally and assign killer → victim
    kill_dict = {}
    for items in lines.values():
        # Sort left → right
        items.sort(key=lambda x: bbox_center_x(x[1]))
        if len(items) >= 2:
            killer = items[0][0]
            victim = items[1][0]
            if killer in player_list and victim in player_list:
                kill_dict[killer] = victim

    return kill_dict

def check_game_start(img, reader):
    t1_name_str: str
    allowList = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    
    team_one_name_img = imagesplitter.getimgregion(img, roi.team_one_name_bbox)
    
    t1_name = reader.readtext(team_one_name_img, allowlist=allowList)
    
    if t1_name:
    # result structure: [bounding box, text, confidence]
        t1_name_str = t1_name[0][1]
        if any(word in t1_name_str.upper() for word in cdl_teams):
            return True

    return False;
