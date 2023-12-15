import cv2
import cv2.aruco as aruco

# Mettre à 0 pour camera, 1 pour photos
TEST=1

if TEST!=1:
    cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

nomImages=("CarteVideCote.jpg","CarteVideDessus.jpg","ConfigBlocsAvecArucoCote1.jpg","ConfigBlocsAvecArucoCote2.jpg","ConfigBlocsAvecArucoCote3.jpg","ConfigBlocsAvecArucoDessusLoin1.jpg","ConfigBlocsAvecArucoDessusLoin2.jpg","ConfigBlocsAvecArucoDessusProche1.jpg","ConfigBlocsAvecArucoDessusProche2.jpg","ConfigBlocsSansAruco1.jpg","ConfigBlocsSansAruco2.jpg")
nbImages= len(nomImages)

frame=cv2.imread(nomImages[0])

cpt=0

while True:
    if TEST == 0:
        ret, frame = cap.read()
    if TEST == 1:
        frame = cv2.imread(nomImages[cpt])
        frame=cv2.resize(frame,(0,0),fx=0.7,fy=0.7)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cpt+=1
            if cpt==nbImages:
                cpt=0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedCandidates = detector.detectMarkers(gray)

    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            TextOnScreen="Inconnu"
            ColorText = (0, 0, 0)
            if ids[i] == 132:
                TextOnScreen = "NOUS"
                ColorText=(0,0,255)
            if ids[i] == 0:
                TextOnScreen = "HGCarte"
                ColorText = (255, 0, 255)
            if ids[i] == 1:
                TextOnScreen = "HDCarte"
                ColorText = (255, 0, 255)
            if ids[i] == 2:
                TextOnScreen = "BDCarte"
                ColorText = (255, 0, 255)
            if ids[i] == 3:
                TextOnScreen = "BGCarte"
                ColorText = (255, 0, 255)
            frame = cv2.putText(img=frame, text=TextOnScreen, org=(int(corners[i][0][0][0]),int(corners[i][0][0][1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, color=ColorText,thickness=2)


    cv2.imshow('Aruco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if TEST!=1:
    cap.release()
cv2.destroyAllWindows()