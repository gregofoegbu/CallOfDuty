import cv2
import numpy as np

def detect_arrows(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Purple mask
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    # White mask
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)

    closed_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)
    border_emphasis = cv2.subtract(closed_white, mask_white)
    enhanced_white = cv2.bitwise_or(mask_white, border_emphasis)
    enhanced_white = cv2.erode(enhanced_white, kernel, iterations=1)

    dilated_purple = cv2.dilate(mask_purple, kernel, iterations=2)
    dilated_white = cv2.dilate(enhanced_white, kernel, iterations=2)

    output = image.copy()
    find_and_draw_arrows(dilated_purple, output, color_label='P', box_color=(255, 0, 255))
    find_and_draw_arrows(dilated_white, output, color_label='W', box_color=(255, 255, 255))

    return output, dilated_purple, dilated_white

def find_and_draw_arrows(mask, output, color_label, box_color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 3000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        corners = len(approx)

        if 5 <= corners <= 9:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:
                cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(output, f"{color_label} {corners}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

# === MAIN: Video Processing Loop with ROI ===
map_x = 30
map_y = 580
map_width = 420
map_height = 280

cap = cv2.VideoCapture('TrimmedTrim.mov')  # or filename

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract ROI from the frame
    roi = frame[map_y:map_y+map_height, map_x:map_x+map_width]

    # Process the ROI
    processed_roi, _, _ = detect_arrows(roi)

    # Put processed ROI back into frame
    frame[map_y:map_y+map_height, map_x:map_x+map_width] = processed_roi

    cv2.imshow('Frame with ROI Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
