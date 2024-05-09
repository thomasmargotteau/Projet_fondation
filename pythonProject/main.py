import cv2
import cv2.aruco as aruco
import numpy as np
import Grid
import JPS_Pathfinding
import os
import json

ARUCO_DICT = cv2.aruco.DICT_6X6_250  # Dictionary ID
SQUARES_VERTICALLY = 7  # Number of squares vertically
SQUARES_HORIZONTALLY = 5  # Number of squares horizontally
SQUARE_LENGTH = 300  # Square side length (in pixels)
MARKER_LENGTH = 150  # ArUco marker side length (in pixels)
MARGIN_PX = 100  # Margins size (in pixels)

IMG_SIZE = tuple(i * SQUARE_LENGTH + 2 * MARGIN_PX for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY))
OUTPUT_NAME = 'Images\ChArUco_Marker.png'


def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    # Load images from directory
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if
                   f.startswith("ChAruco_Board_") and f.endswith(".jpg")]

    all_charuco_ids = []
    all_charuco_corners = []
    imgSize = None

    # Loop over images and extraction of corners
    for image_file in image_files:
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if imgSize is None:  # Calculate imgSize only once
            imgSize = gray.shape
        image_copy = gray.copy()
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(gray)
        if marker_ids is not None and len(marker_ids) > 0:  # Check if markers are detected
            ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray,
                                                                                  board)
            if ret:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)
        else:
            print(f"No marker detected in {image_file}")

    if imgSize is None:
        print("Error: No images found in the directory")
        return None, None

    # Calibrate camera with extracted information
    result, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board,
                                                                       imgSize, None, None)
    return mtx, dist.flatten()


def calculate_perspective_transform_matrix(Corner1, Corner2, Corner3, Corner4, size):
    # Define the destination points for perspective transformation
    dst_points = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])

    # Extract the corners of the detected ArUco markers
    src_points = np.float32([Corner1, Corner2, Corner3, Corner4])

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    return M


def get_mean_hsv(image, center):
    # Get the mean HSV value in a 5 pixel radius circle around a specified center.

    x, y = center
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    circle_mask = np.zeros_like(hsv_image[:, :, 0], dtype=np.uint8)
    cv2.circle(circle_mask, center, 2, 255, -1)
    mean_hsv = cv2.mean(hsv_image, mask=circle_mask)[:3]
    return tuple(map(round, mean_hsv))


def get_mean_bgr(image, center):
    # Get the mean BGR value in a 5 pixel radius circle around a specified center.

    x, y = center
    bgr_image = image
    circle_mask = np.zeros_like(bgr_image[:, :, 0], dtype=np.uint8)
    cv2.circle(circle_mask, center, 2, 255, -1)
    mean_bgr = cv2.mean(bgr_image, mask=circle_mask)[:3]
    return tuple(map(round, mean_bgr))


# 0 for camera, 1 for photos
TEST = 1

final_size = (1200, 800)

img_dir = "C:\Documents\Foundation_project\pythonProject\Images"

SENSOR = 'monochrome'
LENS = 'S10_Arthur_0.7'
OUTPUT_JSON = 'Calibration.json'

'''mtx, dist = get_calibration_parameters(img_dir)
data = {"sensor": SENSOR, "lens": LENS, "mtx": mtx.tolist(), "dist": dist.tolist()}

with open(OUTPUT_JSON, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f'Data has been saved to {OUTPUT_JSON}')'''

json_file_path = './Calibration.json'

with open(json_file_path, 'r') as file:  # Read the JSON file
    json_data = json.load(file)

mtx = np.array(json_data['mtx'])
dist = np.array(json_data['dist'])

if TEST != 1:
    cap = cv2.VideoCapture(0)

# Defintion of the aruco detection parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

nomImages = (
    "Images/Photo_Cam_finale_1.jpg",  # cpt =  0
    "Images/Photo_Cam_finale_2.jpg",  # cpt =  1
    "Images/Photo_Cam_finale_3.jpg",  # cpt =  2
    "Images/Photo_Cam_finale_4.jpg",  # cpt =  3
    "Images/Photo_Cam_finale_5.jpg",  # cpt =  4
    "Images/Photo_Cam_finale_6.jpg",  # cpt =  5
    "Images/Photo_Cam_finale_7.jpg",  # cpt =  6
    "Images/Photo_Cam_finale_8.jpg",  # cpt =  7
    "Images/PhotoCarteRobot9.jpg",  # cpt =  8
    "Images/Photo_Cam_finale_10.jpg",  # cpt =  9
    "Images/Photo_Cam_finale_11.jpg",  # cpt =  10
    "Images/Photo_Cam_finale_12.jpg")  # cpt =  11

nbImages = len(nomImages)

frame = cv2.imread(nomImages[0])

# Define the rectangles
rectangles = [
    ((224, 605), (309, 690), (1, 37, 255)),  # Red zone
    ((806, 21), (883, 106), (255, 88, 2)),  # Blue zone
    ((806, 605), (883, 690), (23, 143, 26)),  # Green zone
    ((224, 21), (309, 106), (167, 3, 255))  # Yellow zone
]

cpt = 5

while True:
    if TEST == 0:
        ret, frame = cap.read()
    elif TEST == 1:
        frame = cv2.imread(nomImages[cpt])
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        if cv2.waitKey(1) & 0xFF == ord('p'):  # While the code is running press 'p' to go to the next image
            cpt += 1
            if cpt == nbImages + 1:
                cpt = 0

    cv2.imwrite("Images\Frame.jpg", frame)
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    cv2.imwrite("Images\Corrected_frame.jpg", frame)

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

        dst = cv2.warpPerspective(frame, M, final_size)

        RB = 132
        tailleRB = 75
        # Saving the images to a file
        cv2.imwrite("Images\Transformed_image.jpg", dst)

        vector_img = cv2.imread("Images\Transformed_image.jpg")
        # Detect ArUco markers in the image.
        corners, marker_ids, rejected = detector.detectMarkers(vector_img)

        # If markers are detected, draw them on the image.
        if marker_ids is not None:
            # Looping through detected markers and marker ids at same time.
            for corner, marker_id in zip(corners, marker_ids):
                # Draw the marker corners.
                if (marker_id[0] == RB):
                    cv2.polylines(
                        vector_img, [corner.astype(np.int32)], True, (0, 0, 202), 1, cv2.LINE_AA
                    )

                # Get the top-right, top-left, bottom-right, and bottom-left corners of the marker.
                # Change the shape of numpy array to 4 by 2
                corner = corner.reshape(4, 2)

                # Change the type of numpy array values integers
                corner = corner.astype(int)

                # Extracting corners
                top_left, top_right, bottom_right, bottom_left = corner

                # Calculate the midpoint between the bottom two corners
                bottom_midpoint_x = (top_right[0] + bottom_right[0]) // 2
                bottom_midpoint_y = (top_right[1] + bottom_right[1]) // 2

                # Calculate the center of the Aruco for future vector
                mid_midpoint_x = (top_right[0] + bottom_left[0]) // 2
                mid_midpoint_y = (top_right[1] + bottom_left[1]) // 2

                cv2.circle(vector_img, (bottom_midpoint_x, bottom_midpoint_y), 3, (0, 255, 0), -1)
                # Calculate the direction vector from the bottom two corners
                direction_vector_x = bottom_midpoint_x - mid_midpoint_x
                direction_vector_y = bottom_midpoint_y - mid_midpoint_y

                # Normalize the direction vector
                direction_vector_length = np.sqrt(direction_vector_x ** 2 + direction_vector_y ** 2)
                normalized_direction_vector_x = direction_vector_x / direction_vector_length
                normalized_direction_vector_y = direction_vector_y / direction_vector_length

                # Move the midpoint 10 pixels in front of the equidistance of the bottom two corners
                new_center_x = bottom_midpoint_x + int(25 * normalized_direction_vector_x)
                new_center_y = bottom_midpoint_y + int(25 * normalized_direction_vector_y)

                centre = (new_center_x, new_center_y)

                if (marker_id[0] == RB):
                    RBCoords = centre
                    '''cv2.circle(dst, RBCoords, tailleRB, (0, 0, 0), -1)'''

                    # Points of the front line
                    point1 = top_right
                    point2 = bottom_right

                    # Start points for the second line
                    start_point = centre

                    # Calculate direction vector
                    direction_vector = np.array([(point2[0] - point1[0]), (point2[1] - point1[1])])

                    # Normalize direction vector
                    direction_vector = direction_vector / np.linalg.norm(direction_vector)

                    # Vector Length
                    taille_ligne = 60

                    # Calculate final points for the vector
                    new_point1 = (int(start_point[0]), int(start_point[1]))
                    new_point2 = (int(start_point[0] + taille_ligne * normalized_direction_vector_x),
                                  int(start_point[1] + taille_ligne * normalized_direction_vector_y))

                    # Draw the second parallel line (vector)
                    cv2.line(vector_img, new_point1, new_point2, (220, 33, 20), 3)
                    # Draw the rectangle oriented by the vector
                    rectangle_width = 110
                    rectangle_height = 130
                    angle = np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi
                    center = (int(RBCoords[0]), int(RBCoords[1]))
                    rectangle_points = cv2.boxPoints(((center), (rectangle_width, rectangle_height), angle))
                    cv2.drawContours(dst, [np.intp(rectangle_points)], 0, (0, 0, 0), -1)
                    cv2.drawContours(vector_img, [np.intp(rectangle_points)], 0, (220, 33, 20), 3)

        # Save the image.
        cv2.imwrite("Images\Transformed_image.jpg", dst)
        cv2.imwrite("Images\Cache.jpg", vector_img)

        cv2.imshow("Cache", vector_img)

        Cal_img = dst.copy()
        # Draw the zones
        cv2.rectangle(dst, (1150, 50), (1200, 0), (0, 0, 0), -1)
        cv2.rectangle(dst, (1150, 750), (1200, 800), (0, 0, 0), -1)
        cv2.rectangle(dst, (0, 0), (50, 50), (0, 0, 0), -1)
        cv2.rectangle(dst, (0, 750), (50, 800), (0, 0, 0), -1)
        cv2.rectangle(dst, (0, 0), (1200, 12), (0, 0, 0), -1)
        cv2.rectangle(dst, (0, 0), (12, 800), (0, 0, 0), -1)
        cv2.rectangle(dst, (1200, 800), (1188, 0), (0, 0, 0), -1)
        cv2.rectangle(dst, (1200, 800), (0, 788), (0, 0, 0), -1)

        # Adding a Grid
        square_size = 10  # Size of each square in pixels

        # Displaying different results
        cv2.imshow('Hitbox', dst)

        img_copy = dst.copy()

        # Define the centers of the circles
        circle_centers = [(103, 798), (140, 798), (175, 798), (215, 798), (250, 799), (760, 3)]
        circle_centers_grey = [(802, 2), (839, 2), (877, 2), (914, 2), (950, 2), (987, 2)]
        circle_radius = 13
        # Get the mean HSV values for each circle
        mean_hsv_values = []
        # Get the mean BGR values for each circle
        mean_bgr_values = []

        for center in circle_centers:
            mean_hsv = get_mean_hsv(Cal_img, center)
            mean_hsv_values.append(mean_hsv)

        for center_grey in circle_centers_grey:
            mean_bgr = get_mean_bgr(Cal_img, center_grey)
            mean_bgr_values.append(mean_bgr)

        # Print the mean HSV values
        for i in range(6):
            print(
                f"Circle {i + 1}: Mean HSV = {mean_hsv_values[i][0]} / {mean_hsv_values[i][1]} / {mean_hsv_values[i][2]}")
            '''print(f" Grey Circle {i + 1}: Mean BGR = {mean_bgr_values[i][0]} / {mean_bgr_values[i][1]} / {mean_bgr_values[i][2]}")
            print(f"\n")'''
            # Convert HSV mean values to BGR in order to draw on the frame 'img_copy'
            bgr_color = cv2.cvtColor(
                np.uint8([[[mean_hsv_values[i][0], mean_hsv_values[i][1], mean_hsv_values[i][2]]]]), cv2.COLOR_HSV2BGR)
            bgr_color = (int(bgr_color[0][0][0]), int(bgr_color[0][0][1]), int(bgr_color[0][0][2]))
            # Draw filled circles
            cv2.circle(img_copy, (circle_centers[i][0], circle_centers[i][1]), circle_radius, bgr_color, -1)

            cv2.circle(img_copy, (circle_centers_grey[i][0], circle_centers_grey[i][1]), circle_radius,
                       (mean_bgr_values[i][0], mean_bgr_values[i][1], mean_bgr_values[i][2]), -1)  # Draw filled circles

        cv2.imshow("Color calibration", img_copy)

        hsv_img = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

        for i in range(6):
            # Definition of mask boundaries with the previously acquired mean values
            if i == 1:
                # Blue
                bound_lower = np.array(
                    [mean_hsv_values[i][0] - 15, mean_hsv_values[i][1], 0])
                bound_upper = np.array([mean_hsv_values[i][0] + 30, 255, 255])
                blue_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 3:
                # Magenta
                bound_lower = np.array(
                    [mean_hsv_values[i][0] - 8, mean_hsv_values[i][1] - 100, 0])
                bound_upper = np.array([255, 255, 255])
                magenta_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 4:
                # Red
                bound_lower = np.array(
                    [0, 130, 0])
                bound_upper = np.array([35, 255, 255])
                red_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 5:
                # White
                bound_lower = np.array(
                    [0, 0, mean_hsv_values[i][2] + 25])
                bound_upper = np.array([180, 255, 255])
                white_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)

            elif i == 2:
                # Grey
                bound_lower = np.array(
                    [mean_bgr_values[0][0] - 25, mean_bgr_values[0][1] - 25, mean_bgr_values[0][2] - 25])
                bound_upper = np.array(
                    [mean_bgr_values[5][0] + 35, mean_bgr_values[5][1] + 35, mean_bgr_values[5][1] + 35])
                grey_mask = cv2.inRange(dst, bound_lower, bound_upper)

        kernel = np.ones((13, 13), np.uint8)
        grey_kernel = np.ones((17, 17), np.uint8)
        black_mask = cv2.inRange(dst, (0, 0, 0), (1, 1, 1))

        # Definition and displaying of different masks
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

        display_black_mask = cv2.bitwise_and(dst, dst, mask=black_mask)

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        display_blue_mask = cv2.bitwise_and(dst, dst, mask=blue_mask)

        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_CLOSE, kernel)
        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_OPEN, kernel)

        display_magenta_mask = cv2.bitwise_and(dst, dst, mask=magenta_mask)

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        display_red_mask = cv2.bitwise_and(dst, dst, mask=red_mask)

        grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, grey_kernel)
        grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_OPEN, grey_kernel)

        display_grey_mask = cv2.bitwise_and(dst, dst, mask=grey_mask)

        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        display_white_mask = cv2.bitwise_and(dst, dst, mask=white_mask)
        # Making a new red mask out of magenta and red
        final_red_mask = cv2.bitwise_or(magenta_mask, red_mask)

        # Display the blended mask
        display_final_red_mask = cv2.bitwise_and(dst, dst, mask=final_red_mask)

        cv2.imshow("Blue mask", display_blue_mask)
        cv2.imshow("Final red mask", display_final_red_mask)
        cv2.imshow("Grey mask", display_grey_mask)
        cv2.imshow("White mask", display_white_mask)
        '''cv2.imshow("Magenta mask", display_magenta_mask)
        cv2.imshow("Red mask", display_red_mask)'''

        # Combine all masks into a dictionary with mask names as keys
        all_masks = {
            'blue': blue_mask,
            'red': final_red_mask,
            'grey': grey_mask,
            'white': white_mask,
            'black': black_mask
        }

        # Apply the function to color the boxes
        img_with_colored_boxes, grid = Grid.color_boxes_with_masks(dst, all_masks)
        img_with_colored_boxes_corrected = Grid.remove_small_color_groups(img_with_colored_boxes)
        img_with_colored_boxes_corrected = Grid.add_grid(img_with_colored_boxes_corrected, square_size)
        img_JPS = img_with_colored_boxes_corrected.copy()
        cv2.imshow('Colored Grid', img_with_colored_boxes_corrected)
        # Draw numbers on the image representing grid values
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (0, 0, 0)  # White text color

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                # Determine text position
                text_pos = (x * 10 + 2, y * 10 + 9)
                # Draw text on the image
                cv2.putText(img_with_colored_boxes_corrected, str(grid[y, x]), text_pos, font, font_scale, text_color,
                            font_thickness)

        # Display or save the image with numbers
        cv2.imshow('Grid with Numbers', img_with_colored_boxes_corrected)

        '''green_zone = JPS_Pathfinding.find_color_centers(grid, 9)
        blue_cube = JPS_Pathfinding.find_color_centers(grid, 1)
        red_zone = JPS_Pathfinding.find_color_centers(grid, 8)
        yellow_zone = JPS_Pathfinding.find_color_centers(grid, 6)
        blue_zone = JPS_Pathfinding.find_color_centers(grid, 7)

        img_JPS2 = img_JPS.copy()
        img_JPS3 = img_JPS.copy()

        JPS_Pathfinding.jps_algorithm(grid, green_zone[0], blue_cube[0], square_size, img_JPS)
        JPS_Pathfinding.jps_algorithm(grid, blue_cube[0], blue_zone[0], square_size, img_JPS)

        white_cubes = JPS_Pathfinding.find_color_centers(grid, 3)

        JPS_Pathfinding.jps_algorithm(grid, blue_zone[0], white_cubes[0], square_size, img_JPS2)
        JPS_Pathfinding.jps_algorithm(grid, white_cubes[0], yellow_zone[0], square_size, img_JPS2)
        JPS_Pathfinding.jps_algorithm(grid, yellow_zone[0], white_cubes[1], square_size, img_JPS2)
        JPS_Pathfinding.jps_algorithm(grid, white_cubes[1], yellow_zone[0], square_size, img_JPS2)
        JPS_Pathfinding.jps_algorithm(grid, yellow_zone[0], white_cubes[2], square_size, img_JPS2)
        JPS_Pathfinding.jps_algorithm(grid, white_cubes[2], yellow_zone[0], square_size, img_JPS2)

        red_cubes = JPS_Pathfinding.find_color_centers(grid, 4)
        JPS_Pathfinding.jps_algorithm(grid, yellow_zone[0], red_cubes[0], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_cubes[0], red_zone[0], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_zone[0], red_cubes[1], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_cubes[1], red_zone[0], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_zone[0], red_cubes[2], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_cubes[2], red_zone[0], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_zone[0], red_cubes[3], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_cubes[3], red_zone[0], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_zone[0], red_cubes[4], square_size, img_JPS3)
        JPS_Pathfinding.jps_algorithm(grid, red_cubes[4], red_zone[0], square_size, img_JPS3)

        cv2.imshow('JPS* result', img_JPS)
        cv2.imshow('JPS* result 2', img_JPS2)
        cv2.imshow('JPS* result 3', img_JPS3)'''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if TEST != 1:
    cap.release()
cv2.destroyAllWindows()
