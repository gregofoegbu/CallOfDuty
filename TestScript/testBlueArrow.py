import cv2 as cv
import numpy as np

cap = cv.VideoCapture("../../vods/VOD-2025-11-16.mov")

map_x, map_y, map_width, map_height = 30, 535, 420, 280

# good starting range
lower_blue = np.array([90, 108, 80])
upper_blue = np.array([120, 255, 255])

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    # Crop ROI
    roi = frame[map_y:map_y + map_height, map_x:map_x + map_width]

    # Blur + HSV
    blur = cv.GaussianBlur(roi, (5,5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    # Mask
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Optional filtering by area
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    filtered = np.zeros_like(mask)
    for cnt in contours:
        if cv.contourArea(cnt) > 80:
            cv.drawContours(filtered, [cnt], -1, 255, -1)

    # ---- SHOW WINDOWS ----
    cv.imshow("ROI (Map)", roi)
    cv.imshow("Mask", mask)
    cv.imshow("Filtered", filtered)

    # Quit
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break

cap.release()
cv.destroyAllWindows()
