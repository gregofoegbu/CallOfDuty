import cv2
import numpy as np

def detect_arrows_with_info(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Purple arrow mask ---
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    # --- Black arrow mask (refined to avoid gray and dark gray) ---
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 35])  # Restrict V to stay dark
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((3, 3), np.uint8)

    # Morphological cleanup
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_purple = cv2.dilate(mask_purple, kernel, iterations=1)

    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_black = cv2.dilate(mask_black, kernel, iterations=2)

    output = image.copy()
    cv2.imshow('black mask', mask_black)
    cv2.imshow('purple mask', mask_purple)
    # Detect contours
    purple_detections = find_arrows_and_record(mask_purple, output, 'P', (255, 0, 255))
    black_detections = find_arrows_and_record(mask_black, output, 'B', (0, 0, 0))

    return output, purple_detections + black_detections

def find_arrows_and_record(mask, output, color_label, box_color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 2000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        corners = len(approx)

        if 5 <= corners <= 9:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity < 0.7:
                    continue

                # Draw box
                cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 1)
                cv2.putText(output, f"{color_label}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                center_x = x + w // 2
                center_y = y + h // 2

                detections.append({
                    'color': color_label,
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'corners': corners,
                    'area': area,
                    'solidity': solidity
                })

    return detections

# === MAIN: Video loop example ===

map_x, map_y, map_width, map_height = 30, 580, 420, 280

cap = cv2.VideoCapture('TrimmedClip.mov')

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[map_y:map_y+map_height, map_x:map_x+map_width]
    processed_roi, detections = detect_arrows_with_info(roi)
    frame[map_y:map_y+map_height, map_x:map_x+map_width] = processed_roi

    for det in detections:
        print(f"{det['color']} - Center: {det['center']}, Area: {det['area']:.0f}, Solidity: {det['solidity']:.2f}")

    cv2.imshow('Frame with ROI Detection', frame)
    cv2.imshow('ROI ', processed_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
