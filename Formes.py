import cv2 as cv
import numpy as np

# Capturer la vidéo depuis la webcam (index 0 pour la webcam par défaut)
cap = cv.VideoCapture(0)

while True:
    # Lire l'image depuis la caméra
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture de l'image.")
        break



    # Convertir l'image redimensionnée en niveaux de gris
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Appliquer un flou et effectuer la détection de contours
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150, apertureSize=3)

    # Trouver les contours dans l'image
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Fonction pour reconnaître et dessiner les polygones
    def detect_polygons(cnt):
        approx = cv.approxPolyDP(cnt, 0.03 * cv.arcLength(cnt, True), True)
        # Si le contour a entre 3 et 8 côtés, considérez-le comme un polygone
        if len(approx) >= 3 and len(approx) <= 8:
            # Dessiner le polygone détecté
            cv.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            # Déterminez le type de polygone en fonction du nombre de côtés
            shape = "Polygone"
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Rectangle"  # Peut être carré ou rectangle
            elif len(approx) == 5:
                shape = "Pentagone"
            elif len(approx) == 6:
                shape = "Hexagone"
            elif len(approx) == 7:
                shape = "Heptagone"
            elif len(approx) == 8:
                shape = "Octogone"
            # Afficher le type de polygone détecté
            cv.putText(frame, shape, (approx.ravel()[0], approx.ravel()[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Parcourir tous les contours pour les détecter et les dessiner
    for c in contours:
        detect_polygons(c)

    # Afficher l'image redimensionnée avec les polygones détectés
    cv.imshow('Polygones détectés', frame)

    # Sortir de la boucle si la touche 'q' est enfoncée
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture et fermer les fenêtres
cap.release()
cv.destroyAllWindows()
