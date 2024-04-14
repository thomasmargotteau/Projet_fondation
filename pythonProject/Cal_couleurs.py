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


def get_mean_hsv(image, center):
    """
    Get the mean HSV value in a 5 pixel radius circle around a specified center.
    Args:
        image: Input image in BGR format.
        center: Tuple containing the (x, y) coordinates of the circle center.
    Returns:
        Tuple containing the mean HSV value (Hue, Saturation, Value).
    """
    x, y = center
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    circle_mask = np.zeros_like(hsv_image[:, :, 0], dtype=np.uint8)
    cv2.circle(circle_mask, center, 2, 255, -1)
    mean_hsv = cv2.mean(hsv_image, mask=circle_mask)[:3]
    return tuple(map(round, mean_hsv))


def get_mean_bgr(image, center):
    """
    Get the mean HSV value in a 5 pixel radius circle around a specified center.
    Args:
        image: Input image in BGR format.
        center: Tuple containing the (x, y) coordinates of the circle center.
    Returns:
        Tuple containing the mean HSV value (Hue, Saturation, Value).
    """
    x, y = center
    bgr_image = image
    circle_mask = np.zeros_like(bgr_image[:, :, 0], dtype=np.uint8)
    cv2.circle(circle_mask, center, 2, 255, -1)
    mean_bgr = cv2.mean(bgr_image, mask=circle_mask)[:3]
    return tuple(map(round, mean_bgr))


# Mettre à 0 pour camera, 1 pour photos
TEST = 1

final_size = (1200, 800)

# Taille réelle de l'ArUco en millimètres
taille_reelle_aruco_mm = 42.0  # Modifier cette valeur selon la taille réelle de votre ArUco

if TEST != 1:
    cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

nomImages = (
"Images\PhotoCam2_1.jpg", "Images\PhotoCarteRobot5.jpg", "Images\PhotoCarteRobot9.jpg", "Images\PhotoCarteRobot1.jpg",
"Images\PhotoCarteRobot2.jpg", "Images\PhotoCarteRobot3.jpg", "Images\PhotoCarteRobot6.jpg",
"Images\PhotoCarteRobot8.jpg", "Images\PhotoCarteRobot10.jpg", "Images\ConfigBlocsAvecArucoCote1.jpg",
"Images\PhotoCarteTelephone5.jpg",
"Images\ConfigBlocsAvecArucoCote3.jpg", "Images\ConfigBlocsAvecArucoDessusLoin1.jpg",
"Images\ConfigBlocsAvecArucoDessusProche1.jpg")
nbImages = len(nomImages)

frame = cv2.imread(nomImages[0])

cpt = 2

# 130*120      der = 30   h = b = 55

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
            elif ids[i] == 1:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner points
                Corner2 = p2
            elif ids[i] == 2:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner points
                Corner3 = p3
            elif ids[i] == 3:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner points
                Corner4 = p4

        M = calculate_perspective_transform_matrix(Corner1, Corner2, Corner4, Corner3, final_size)

        for i in range(len(ids)):
            TextOnScreen = "Inconnu"
            ColorText = (0, 0, 0)
            if ids[i] == 132:
                cv2.rectangle(frame, (int(corners[i][0][0][0]), int(corners[i][0][0][1])),
                              (int(corners[i][0][2][0]), int(corners[i][0][2][1])),
                              (170, 0, 190), -1)
            else:
                TextOnScreen = None




        if ids is not None and 132 in ids:
            # Find the index of the marker with ID 132
            idx = list(ids).index(132)

            # Get the corners of the detected marker

            # Draw a black filled rectangle over the detected marker
            cv2.rectangle(frame, (int(corners[idx][0][0][0]), int(corners[idx][0][0][1])),
                          (int(corners[idx][0][0][0]) + 130, int(corners[idx][0][0][1]) + 120),
                          (0, 0, 0), -1)

            # Display the image with the rectangle drawn

        '''else:
            print("Marker with ID 132 not found.")'''
        dst = cv2.warpPerspective(frame, M, final_size)

        cv2.imwrite("Images\Frame.jpg", frame)
        cv2.imwrite("Images\Transformed_image.jpg", dst)

        # Open the image
        image = cv2.imread(
            "Images/Transformed_image.jpg")  # Replace "Images/Transformed_image.jpg" with the path to your image file

        # Define the dimensions of the centered rectangle
        crop_width = 1110
        crop_height = 710

        # Calculate the coordinates for cropping
        left = (image.shape[1] - crop_width) // 2
        top = (image.shape[0] - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        cropped_image = image[top:bottom, left:right]

        # Save the cropped image
        cv2.imwrite("Images/cropped_frame.jpg", cropped_image)  # Save the cropped image to a file

        cv2.imshow('Perspective Transformation Result', dst)
        cv2.imshow('Cropped image', cropped_image)

        img_copy = dst.copy()

        # Define the centers of the circles
        circle_centers = [(103, 797), (140, 797), (175, 797), (215, 797), (250, 797)]
        circle_centers_grey = [(802, 3), (839, 3), (877, 3), (914, 3), (949, 3)]
        circle_radius = 10
        # Get the mean HSV values for each circle
        mean_hsv_values = []
        mean_bgr_values = []

        for center in circle_centers:
            mean_hsv = get_mean_hsv(dst, center)
            mean_hsv_values.append(mean_hsv)

        for center_grey in circle_centers_grey:
            mean_bgr = get_mean_bgr(dst, center_grey)
            mean_bgr_values.append(mean_bgr)

        # Print the mean HSV values
        '''for i in range(5):
            print(f"Circle {i + 1}: Mean HSV = {mean_hsv_values[i][0]} {mean_hsv_values[i][1]} {mean_hsv_values[i][2]}")
            bgr_color = cv2.cvtColor(
                np.uint8([[[mean_hsv_values[i][0], mean_hsv_values[i][1], mean_hsv_values[i][2]]]]), cv2.COLOR_HSV2BGR)
            bgr_color = (int(bgr_color[0][0][0]), int(bgr_color[0][0][1]), int(bgr_color[0][0][2]))
            cv2.circle(img_copy, (circle_centers[i][0], circle_centers[i][1]), circle_radius, bgr_color, -1)

            cv2.circle(img_copy, (circle_centers_grey[i][0], circle_centers_grey[i][1]), circle_radius,
                       (mean_bgr_values[i][0], mean_bgr_values[i][1], mean_bgr_values[i][2]), -1)  # Draw filled circles'''

        '''cv2.imshow("Image with Circles", img_copy)'''

        hsv_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        for i in range(5):
            if i == 1:
                # Blue
                bound_lower = np.array(
                    [mean_hsv_values[i][0] - 8, mean_hsv_values[i][1] - 10, mean_hsv_values[i][2] - 10])
                bound_upper = np.array([mean_hsv_values[i][0] + 8, 255, 190])
                blue_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 3:
                # Magenta
                bound_lower = np.array(
                    [mean_hsv_values[i][0] - 8, mean_hsv_values[i][1] - 100, mean_hsv_values[i][2] - 100])
                bound_upper = np.array([mean_hsv_values[i][0] + 8, 255, 255])
                magenta_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 4:
                # Red
                bound_lower = np.array(
                    [mean_hsv_values[i][0] - 20, mean_hsv_values[i][1] - 100, mean_hsv_values[i][2] - 100])
                bound_upper = np.array([mean_hsv_values[i][0], 255, 255])
                red_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 2:
                # Grey
                bound_lower = np.array(
                    [mean_bgr_values[0][0] - 50, mean_bgr_values[0][1] - 50, mean_bgr_values[0][2] - 50])
                bound_upper = np.array(
                    [mean_hsv_values[3][0], mean_bgr_values[3][1], mean_bgr_values[3][1]])
                grey_mask = cv2.inRange(cropped_image, bound_lower, bound_upper)

        kernel = np.ones((7, 7), np.uint8)

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        display_blue_mask = cv2.bitwise_and(cropped_image, cropped_image, mask=blue_mask)

        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_CLOSE, kernel)
        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_OPEN, kernel)

        display_magenta_mask = cv2.bitwise_and(cropped_image, cropped_image, mask=magenta_mask)

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        display_red_mask = cv2.bitwise_and(cropped_image, cropped_image, mask=red_mask)

        grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel)
        grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_OPEN, kernel)

        display_grey_mask = cv2.bitwise_and(cropped_image, cropped_image, mask=grey_mask)

        '''cv2.imshow("Blue mask", display_blue_mask)
        cv2.imshow("Magenta mask", display_magenta_mask)
        cv2.imshow("Red mask", display_red_mask)
        cv2.imshow("Grey mask", display_grey_mask)'''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if TEST != 1:
    cap.release()
cv2.destroyAllWindows()
