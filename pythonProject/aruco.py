import cv2
import cv2.aruco as aruco

# Mettre à 0 pour camera, 1 pour photos
TEST=0


# Taille réelle de l'ArUco en millimètres
taille_reelle_aruco_mm = 42.0  # Modifier cette valeur selon la taille réelle de votre ArUco

if TEST!=1:
    cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

nomImages=("Images\CarteVideCote.jpg","Images\PhotoCarteRobot1.jpg","Images\ConfigBlocsAvecArucoCote1.jpg","Images/Transformed_image.jpg","Images\ConfigBlocsAvecArucoCote3.jpg","Images\ConfigBlocsAvecArucoDessusLoin1.jpg","Images\ConfigBlocsAvecArucoDessusLoin2.jpg","Images\ConfigBlocsAvecArucoDessusProche1.jpg","Images\ConfigBlocsAvecArucoDessusProche2.jpg","Images\ConfigBlocsSansAruco1.jpg","Images\ConfigBlocsSansAruco2.jpg")
nbImages= len(nomImages)

frame=cv2.imread(nomImages[0])

cpt=0

while True:
    if TEST == 0:
        ret, frame = cap.read()
    elif TEST == 1:
        frame = cv2.imread(nomImages[cpt])
        frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cpt += 1
            if cpt == nbImages:
                cpt = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedCandidates = detector.detectMarkers(frame)

    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # Recherche des indices des marqueurs BGCarte et BDCarte dans la liste ids
        bg_index = None
        bd_index = None
        for i in range(len(ids)):
            if ids[i] == 3:  # ID de BGCarte
                bg_index = i
            elif ids[i] == 2:  # ID de BDCarte
                bd_index = i

        # Si les deux marqueurs sont détectés, calcul de la distance entre eux
        if bg_index is not None and bd_index is not None:
            # Coins spécifiques des marqueurs BGCarte et BDCarte
            bg_corners = corners[bg_index][0]
            bd_corners = corners[bd_index][0]

            # Calcul de la distance horizontale entre les coins BGCarte et BDCarte
            pixel_distance = abs(bg_corners[0][0] - bd_corners[0][0])

            # Calcul de la distance horizontale entre les coins inférieurs gauche et droite du même ArUco
            pixel_distance2 = abs(bd_corners[1][0] - bd_corners[2][0])

            # Conversion en échelle en millimètres par pixel
            echelle_mm_par_pixel = taille_reelle_aruco_mm / pixel_distance2

            # Calcul de la distance en millimètres
            distance_mm = pixel_distance * echelle_mm_par_pixel

            # Affichage de la distance dans la variable TextOnScreen
            TextOnScreen = f"Distance BGCarte-BDCarte : {distance_mm:.2f} mm"

            # Affichage de la distance entre les deux marqueurs sur l'image
            frame = cv2.putText(frame, TextOnScreen, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)

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