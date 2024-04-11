import numpy as np
import cv2 as cv

ComputerCamera = 0
Black = (0, 0, 0)

TextFont = cv.FONT_HERSHEY_SIMPLEX

# Blue
BlueMin = np.array([94, 80, 2], np.uint8)
BlueMax = np.array([120, 255, 255], np.uint8)

# Red
RedMin = np.array([136, 87, 111], np.uint8)
RedMax = np.array([180, 255, 255], np.uint8)

# Green
GreenMin = np.array([25, 52, 72], np.uint8)
GreenMax = np.array([102, 255, 255], np.uint8)

# White
WhiteMin = np.array([0, 0, 200], np.uint8)
WhiteMax = np.array([180, 30, 255], np.uint8)

# Black
BlackMin = np.array([0, 0, 0], np.uint8)
BlackMax = np.array([180, 255, 30], np.uint8)

cap = cv.VideoCapture(ComputerCamera)

if not cap.isOpened():
    print("Camera cannot be opened")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error accessing camera data")
        break

    BgrToHsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    BlueMask = cv.inRange(BgrToHsvFrame, BlueMin, BlueMax)
    RedMask = cv.inRange(BgrToHsvFrame, RedMin, RedMax)
    GreenMask = cv.inRange(BgrToHsvFrame, GreenMin, GreenMax)
    WhiteMask = cv.inRange(BgrToHsvFrame, WhiteMin, WhiteMax)
    BlackMask = cv.inRange(BgrToHsvFrame, BlackMin, BlackMax)

    kernel = np.ones((5, 5), "uint8")

    masks = [BlueMask, RedMask, GreenMask, WhiteMask, BlackMask]

    for idx, color_mask in enumerate(masks):
        color = None
        if idx == 0:
            color = (255, 0, 0)  # Blue
            color_text = "Bleu"
        elif idx == 1:
            color = (0, 0, 255)  # Red
            color_text = "Rouge"
        elif idx == 2:
            color = (0, 255, 0)  # Green
            color_text = "Vert"
        elif idx == 3:
            color = (255, 255, 255)  # White
            color_text = "Blanc"
        elif idx == 4:
            color = (0, 0, 0)  # Black
            color_text = "Noir"

        color_mask = cv.dilate(color_mask, kernel)
        res = cv.bitwise_and(frame, frame, mask=color_mask)

        contours, hierarchy = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv.contourArea(contour)
            if area > 300:
                x, y, w, h = cv.boundingRect(contour)
                imageFrame = cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(imageFrame, color_text, (x, y), TextFont, 1.0, color)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
