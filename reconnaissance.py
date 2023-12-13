import numpy as np
import cv2 as cv

ComputerCamera=0;
Black=(0,0,0)

TextFont=cv.FONT_HERSHEY_SIMPLEX

# Blue
BlueMin = np.array([94, 80, 2], np.uint8)
BlueMax = np.array([120, 255, 255], np.uint8)

cap=cv.VideoCapture(ComputerCamera)

if not cap.isOpened():
    print("Camera cannot be opened")
    exit()
while True :
    ret, frame=cap.read()

    if not ret:
        print("Error access to camera data")
        break

    BgrToHsvFrame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    BgrToGrayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    BgrToGrayFrame = np.float32(BgrToGrayFrame)

    CornerHarris=cv.cornerHarris(BgrToGrayFrame,2,3,0.04)
    CornerHarris=cv.dilate(CornerHarris,None)

    frame[CornerHarris>0.01*CornerHarris.max()]=[0,0,255]

    BlueMask = cv.inRange(BgrToHsvFrame, BlueMin, BlueMax)

    kernel = np.ones((5, 5), "uint8")

    BlueMask = cv.dilate(BlueMask, kernel)
    res_blue = cv.bitwise_and(frame, frame,
                               mask=BlueMask)

    # Creating contour to track blue color
    contours, hierarchy = cv.findContours(BlueMask,
                                           cv.RETR_TREE,
                                           cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv.boundingRect(contour)
            imageFrame = cv.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)

            cv.putText(frame, "Bleu", (x, y),
                        TextFont,
                        1.0, (255, 0, 0))

    #gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #cv.imshow('frame')

    #cv.rectangle(frame,RectangleTopLeftCorner,RectangleBotRightCorner,Black,2)
    #cv.putText(frame,"Testing...",(10,180),TextFont,4,Black,2)

    cv.imshow('frame',frame)

    if cv.waitKey(1)==ord('q'):
        break

cap.release()
cv.destroyAllWindows()