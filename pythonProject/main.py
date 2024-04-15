import cv2
import cv2.aruco as aruco
import numpy as np
import Grid

def calculate_perspective_transform_matrix(Corner1, Corner2, Corner3, Corner4, size):
    # Define the destination points for perspective transformation
    dst_points = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])

    # Extract the corners of the detected ArUco markers
    src_points = np.float32([Corner1, Corner2, Corner3, Corner4])

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    return M


def get_mean_hsv(image, center):
    #Get the mean HSV value in a 5 pixel radius circle around a specified center.

    x, y = center
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    circle_mask = np.zeros_like(hsv_image[:, :, 0], dtype=np.uint8)
    cv2.circle(circle_mask, center, 2, 255, -1)
    mean_hsv = cv2.mean(hsv_image, mask=circle_mask)[:3]
    return tuple(map(round, mean_hsv))


def get_mean_bgr(image, center):
    #Get the mean BGR value in a 5 pixel radius circle around a specified center.

    x, y = center
    bgr_image = image
    circle_mask = np.zeros_like(bgr_image[:, :, 0], dtype=np.uint8)
    cv2.circle(circle_mask, center, 2, 255, -1)
    mean_bgr = cv2.mean(bgr_image, mask=circle_mask)[:3]
    return tuple(map(round, mean_bgr))


# 0 for camera, 1 for photos
TEST = 1

final_size = (1200, 800)


if TEST != 1:
    cap = cv2.VideoCapture(0)

# Defintion of the aruco detection parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

nomImages = (
    "Images\PhotoCam2_1.jpg",                              #cpt =  0
    "Images\PhotoCarteRobot5.jpg",                         #cpt =  1
    "Images\PhotoCarteRobot9.jpg",                         #cpt =  2
    "Images\PhotoCarteRobot1.jpg",                         #cpt =  3
    "Images\PhotoCarteRobot2.jpg",                         #cpt =  4
    "Images\PhotoCarteRobot3.jpg",                         #cpt =  5
    "Images\PhotoCarteRobot6.jpg",                         #cpt =  6
    "Images\PhotoCarteRobot8.jpg",                         #cpt =  7
    "Images\PhotoCarteRobot10.jpg",                        #cpt =  8
    "Images\ConfigBlocsAvecArucoCote1.jpg",                #cpt =  9
    "Images\PhotoCarteTelephone5.jpg",                     #cpt =  10
    "Images\ConfigBlocsAvecArucoCote3.jpg",                #cpt =  11
    "Images\ConfigBlocsAvecArucoDessusLoin1.jpg",          #cpt =  12
    "Images\ConfigBlocsAvecArucoDessusProche1.jpg")        #cpt =  13
nbImages = len(nomImages)

frame = cv2.imread(nomImages[0])

cpt = 2

while True:
    if TEST == 0:
        ret, frame = cap.read()
    elif TEST == 1:
        frame = cv2.imread(nomImages[cpt])
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        if cv2.waitKey(1) & 0xFF == ord('p'):                   # While the code is running press 'p' to go to the next image
            cpt += 1
            if cpt == nbImages:
                cpt = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedCandidates = detector.detectMarkers(gray)

    if ids is not None:
        Corner1 = None
        Corner2 = None
        Corner3 = None
        Corner4 = None
        for i in range(len(ids)):
            if ids[i] == 0:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner1 points
                Corner1 = p1
            elif ids[i] == 1:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner2 points
                Corner2 = p2
            elif ids[i] == 2:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner3 points
                Corner3 = p3
            elif ids[i] == 3:
                p1, p2, p3, p4 = corners[i][0]  # Extract corner4 points
                Corner4 = p4

        M = calculate_perspective_transform_matrix(Corner1, Corner2, Corner4, Corner3, final_size)
        hit_img = frame.copy()
        for i in range(len(ids)):
            if ids[i] == 132:
                # Calculate the midpoint between the bottom two corners
                bottom_midpoint_x = (int(corners[i][0][2][0]) + int(corners[i][0][3][0])) // 2
                bottom_midpoint_y = (int(corners[i][0][2][1]) + int(corners[i][0][3][1])) // 2

                # Calculate the direction vector from the bottom two corners
                direction_vector_x = int(corners[i][0][2][0]) - int(corners[i][0][3][0])
                direction_vector_y = int(corners[i][0][2][1]) - int(corners[i][0][3][1])

                # Normalize the direction vector
                direction_vector_length = np.sqrt(direction_vector_x ** 2 + direction_vector_y ** 2)
                normalized_direction_vector_x = direction_vector_x / direction_vector_length
                normalized_direction_vector_y = direction_vector_y / direction_vector_length

                # Move the midpoint 10 pixels in front of the equidistance of the bottom two corners
                new_center_x = bottom_midpoint_x + int(65 * normalized_direction_vector_x)
                new_center_y = bottom_midpoint_y + int(65 * normalized_direction_vector_y)
                # Draw a circle in front of the aruco (New Hitbox)
                cv2.circle(frame, (new_center_x, new_center_y),
                           115,
                           (0, 0, 0), -1)

        dst = cv2.warpPerspective(frame, M, final_size)
        hit_img = cv2.warpPerspective(hit_img, M, final_size)

        # Saving the images to a file
        cv2.imwrite("Images\Frame.jpg", frame)
        cv2.imwrite("Images\Transformed_image.jpg", dst)
        cv2.imwrite("Images\Hitbox.jpg", hit_img)

        # Open the image
        image = cv2.imread(
            "Images/Transformed_image.jpg")

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
        cv2.imwrite("Images/cropped_frame.jpg", cropped_image)

        # Adding a Grid
        square_size = 16  # Size of each square in pixels
        grid_img = Grid.add_grid(dst, square_size)

        #Displaying different results
        cv2.imshow('Hitbox', hit_img)
        cv2.imshow('Perspective Transformation Result', dst)
        cv2.imshow('Cropped image', cropped_image)
        cv2.imshow('Grid', grid_img)

        img_copy = dst.copy()

        # Define the centers of the circles
        circle_centers = [(103, 797), (140, 797), (175, 797), (215, 797), (250, 797)]
        circle_centers_grey = [(802, 3), (839, 3), (877, 3), (914, 3), (949, 3)]
        circle_radius = 10
        # Get the mean HSV values for each circle
        mean_hsv_values = []
        # Get the mean BGR values for each circle
        mean_bgr_values = []

        for center in circle_centers:
            mean_hsv = get_mean_hsv(dst, center)
            mean_hsv_values.append(mean_hsv)

        for center_grey in circle_centers_grey:
            mean_bgr = get_mean_bgr(dst, center_grey)
            mean_bgr_values.append(mean_bgr)

        # Print the mean HSV values
        for i in range(5):
            '''print(f"Circle {i + 1}: Mean HSV = {mean_hsv_values[i][0]} {mean_hsv_values[i][1]} {mean_hsv_values[i][2]}")'''
            # Convert HSV mean values to BGR in order to draw on the frame 'img_copy'
            bgr_color = cv2.cvtColor(
                np.uint8([[[mean_hsv_values[i][0], mean_hsv_values[i][1], mean_hsv_values[i][2]]]]), cv2.COLOR_HSV2BGR)
            bgr_color = (int(bgr_color[0][0][0]), int(bgr_color[0][0][1]), int(bgr_color[0][0][2]))
            # Draw filled circles
            cv2.circle(img_copy, (circle_centers[i][0], circle_centers[i][1]), circle_radius, bgr_color, -1)

            cv2.circle(img_copy, (circle_centers_grey[i][0], circle_centers_grey[i][1]), circle_radius,
                       (mean_bgr_values[i][0], mean_bgr_values[i][1], mean_bgr_values[i][2]), -1)  # Draw filled circles

        cv2.imshow("Color calibration", img_copy)

        hsv_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        for i in range(5):
        # Definition of mask boundaries with the previously acquired mean values
            if i == 1:
                # Blue
                bound_lower = np.array(
                    [mean_hsv_values[i][0] - 8, mean_hsv_values[i][1] - 15, mean_hsv_values[i][2] - 10])
                bound_upper = np.array([mean_hsv_values[i][0] + 8, 255, 210])
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
                    [mean_bgr_values[0][0] - 45, mean_bgr_values[0][1] - 45, mean_bgr_values[0][2] - 45])
                bound_upper = np.array(
                    [mean_bgr_values[3][0], mean_bgr_values[3][1], mean_bgr_values[3][1]])
                grey_mask = cv2.inRange(cropped_image, bound_lower, bound_upper)

        kernel = np.ones((7, 7), np.uint8)
        # Definition and displaying of different masks
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

        cv2.imshow("Blue mask", display_blue_mask)
        cv2.imshow("Magenta mask", display_magenta_mask)
        cv2.imshow("Red mask", display_red_mask)
        cv2.imshow("Grey mask", display_grey_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if TEST != 1:
    cap.release()
cv2.destroyAllWindows()


