import cv2
import cv2.aruco as aruco
import numpy as np

def calculate_perspective_transform_matrix(Corner1, Corner2, Corner3, Corner4, size):
    # Define the destination points for perspective transformation
    dst_points = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])

    # Extract the corners of the detected ArUco markers
    src_points = np.float32([Corner1, Corner2, Corner3, Corner4])

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    return M

# Mettre à 0 pour camera, 1 pour photos
TEST=1

final_size = (1200, 800)

# Taille réelle de l'ArUco en millimètres
taille_reelle_aruco_mm = 42.0  # Modifier cette valeur selon la taille réelle de votre ArUco

if TEST!=1:
    cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

nomImages=("Images\CarteVideCote.jpg","Images\CarteVideDessus.jpg","Images\ConfigBlocsAvecArucoCote1.jpg","Images/PhotoCarteTelephone1.jpg","Images\ConfigBlocsAvecArucoCote3.jpg","Images\ConfigBlocsAvecArucoDessusLoin1.jpg","Images\ConfigBlocsAvecArucoDessusLoin2.jpg","Images\ConfigBlocsAvecArucoDessusProche1.jpg","Images\ConfigBlocsAvecArucoDessusProche2.jpg","Images\ConfigBlocsSansAruco1.jpg","Images\ConfigBlocsSansAruco2.jpg")
nbImages= len(nomImages)

frame=cv2.imread(nomImages[0])

cpt=3

while True:
    if TEST == 0:
        ret, frame = cap.read()
    elif TEST == 1:
        frame = cv2.imread(nomImages[cpt])
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cpt += 1
            if cpt == nbImages:
                cpt = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedCandidates = detector.detectMarkers(gray)

    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # Recherche des indices des marqueurs BGCarte et BDCarte dans la liste ids
        bg_index = None
        bd_index = None
        Corner1 = None
        Corner2 = None
        Corner3 = None
        Corner4 = None
        for i in range(len(ids)):
            if ids[i] == 0:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner points
                Corner1 = p1
                cv2.circle(frame, (int(p1[0]), int(p1[1])), 2, (0, 0, 255), -1)
            elif ids[i] == 1:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner points
                Corner2 = p2
                cv2.circle(frame, (int(p2[0]), int(p2[1])), 2, (0, 0, 255), -1)
            elif ids[i] == 2:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner points
                Corner3 = p3
                cv2.circle(frame, (int(p3[0]), int(p3[1])), 2, (0, 0, 255), -1)
            elif ids[i] == 3:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner points
                Corner4 = p4
                cv2.circle(frame, (int(p4[0]), int(p4[1])), 2, (0, 0, 255), -1)

        M = calculate_perspective_transform_matrix(Corner1, Corner2, Corner4, Corner3, final_size)

        # Apply perspective transformation
        dst = cv2.warpPerspective(frame, M, final_size)

        # Display the result
        cv2.imshow('Perspective Transformation Result', dst)

        for i in range(len(ids)):
            TextOnScreen = "Inconnu"
            ColorText = (0, 0, 0)
            if ids[i] == 132:
                TextOnScreen = "NOUS"
                ColorText=(0,0,255)
            else:
                TextOnScreen = None
            frame = cv2.putText(img=frame, text=TextOnScreen, org=(int(corners[i][0][0][0]),int(corners[i][0][0][1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=ColorText,thickness=2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if TEST!=1:
    cap.release()
cv2.destroyAllWindows()
