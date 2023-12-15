import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedCandidates = detector.detectMarkers(frame)

    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        TextOnScreen="Inconnu"
        if ids[0] == 132:
            TextOnScreen = "NOUS"
        if ids[0] == 0:
            TextOnScreen = "HGCarte"
        if ids[0] == 1:
            TextOnScreen = "HDCarte"
        if ids[0] == 2:
            TextOnScreen = "BDCarte"
        if ids[0] == 3:
            TextOnScreen = "BGCarte"
        frame = cv2.putText(img=frame, text=TextOnScreen, org=(int(corners[0][0][0][0]),int(corners[0][0][0][1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3.0, color=(255,0,0),thickness=2)


    cv2.imshow('Aruco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()