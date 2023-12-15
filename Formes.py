import cv2 as cv
import numpy as np

# Charger l'image
image = cv.imread('Formes.jpg')  # Remplacez par le chemin de votre image

# Redimensionner l'image
scale_percent = 50  # Modifier cette valeur selon vos besoins
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

# Convertir l'image redimensionnée en niveaux de gris
gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)

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
        cv.drawContours(resized_image, [approx], 0, (0, 255, 0), 2)
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
        cv.putText(resized_image, shape, (approx.ravel()[0], approx.ravel()[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Parcourir tous les contours pour les détecter et les dessiner
for c in contours:
    detect_polygons(c)

# Afficher l'image redimensionnée avec les polygones détectés
cv.imshow('Polygones détectés', resized_image)
cv.waitKey(0)
cv.destroyAllWindows()
