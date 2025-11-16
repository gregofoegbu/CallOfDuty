import cv2
import numpy as np

def preprocess_black_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("Raw", hsv)
    # Black mask: very low V, low S to avoid dark grays
    # lower_white = np.array([0, 0, 240])
    # upper_white = np.array([50, 30, 255]) 
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_black = cv2.inRange(hsv, lower_white, upper_white)

    # Show raw black mask
    cv2.imshow("Black Mask - Raw", mask_black)

    # Morphology to enhance regions (dilate arrows)
    kernel = np.ones((3, 3), np.uint8)
    mask_black_dilated = cv2.dilate(mask_black, kernel, iterations=2)

    # Show mask after dilation
    cv2.imshow("Black Mask - Dilated", mask_black_dilated)

    return mask_black_dilated

def detect_black_arrows(image, mask):
    output = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 110 or area > 2000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 2.0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(output, "B", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return output

# === MAIN ===
map_x, map_y, map_width, map_height = 30, 580, 420, 280

cap = cv2.VideoCapture("TrimmedClip.mov")
if not cap.isOpened():
    print("Failed to open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop minimap
    roi = frame[map_y:map_y+map_height, map_x:map_x+map_width]

    # Black arrow processing
    black_mask = preprocess_black_mask(roi)
    output = detect_black_arrows(roi, black_mask)

    # Show results
    cv2.imshow("Black Arrow Detection Output", output)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
