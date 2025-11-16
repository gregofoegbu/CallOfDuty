import cv2
import numpy as np

def detect_arrows_with_info(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    lower_black = np.array([0, 0, 200])
    upper_black = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((3, 3), np.uint8)

    closed_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)
    border_emphasis = cv2.subtract(closed_white, mask_white)
    enhanced_white = cv2.bitwise_or(mask_white, border_emphasis)
    enhanced_white = cv2.erode(enhanced_white, kernel, iterations=1)

    dilated_purple = cv2.dilate(mask_purple, kernel, iterations=2)
    dilated_white = cv2.dilate(enhanced_white, kernel, iterations=2)

    output = image.copy()

    # Detect arrows for both masks and collect info
    purple_detections = find_arrows_and_record(dilated_purple, output, 'P', (255, 0, 255))
    white_detections = find_arrows_and_record(dilated_white, output, 'W', (255, 255, 255))

    # Combine all detections
    all_detections = purple_detections + white_detections

    return output, all_detections

def find_arrows_and_record(mask, output, color_label, box_color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 110 or area > 3000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        corners = len(approx)

        if 5 <= corners <= 9:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:
                # Optional: check solidity or extent
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity < 0.7:  # discard very hollow shapes
                    continue
                
                # Draw box and label
                cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(output, f"{color_label} {corners} area: {area}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2

                detections.append({
                    'color': color_label,
                    'corners': corners,
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'solidity': solidity,
                })
    return detections

# === MAIN: Video loop example ===

map_x, map_y, map_width, map_height = 30, 580, 420, 280

cap = cv2.VideoCapture('TrimmedClip.mov')   # or video filename

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[map_y:map_y+map_height, map_x:map_x+map_width]

    processed_roi, detections = detect_arrows_with_info(roi)

    # Put back the processed ROI with boxes
    frame[map_y:map_y+map_height, map_x:map_x+map_width] = processed_roi

    # Optionally: print detection info
    for det in detections:
        print(f"Color: {det['color']}, Center: {det['center']}, Area: {det['area']}, Solidity: {det['solidity']:.2f}")

    cv2.imshow('Frame with ROI Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
