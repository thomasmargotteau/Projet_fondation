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

# Function to draw a 20x20 square centered at a given point with a specified color
def draw_square(img, center, color):
    half_size = 18
    top_left = (center[0] - half_size, center[1] - half_size)
    bottom_right = (center[0] + half_size, center[1] + half_size)
    cv2.rectangle(img, top_left, bottom_right, color, -1)

def color_boxes_with_masks(grid_img, masks):

    img_with_colored_boxes = grid_img.copy()

    # Define colors for each mask type
    colors = {
        'blue': (255, 0, 0),
        'red': (0, 0, 255),
        'grey': (100, 100, 100),
        'white': (145, 5, 83),
        'black': (0, 0, 0)
    }

    # Iterate over each grid cell
    square_size = 10  # Size of each square in pixels
    for y in range(0, img_with_colored_boxes.shape[0], square_size):
        for x in range(0, img_with_colored_boxes.shape[1], square_size):
            # Check if any mask has a non-zero value within the current grid cell
            mask_color = (255, 255, 255)  # Default color if no mask is present
            for mask_name, mask in masks.items():
                if np.any(mask[y:y+square_size, x:x+square_size]):
                    # If any mask has a non-zero value, color the box with the corresponding color
                    mask_color = colors[mask_name]
                    break
            # Color each box
            cv2.rectangle(img_with_colored_boxes, (x, y), (x + square_size, y + square_size), mask_color, -1)

    return img_with_colored_boxes

def remove_small_color_groups(img_with_colored_boxes):
    square_size = 10  # Size of each square in pixels

    # Copy the image
    img_with_filtered_color_groups = img_with_colored_boxes.copy()

    # Check neighbors and change color to white if less than 3 neighbors of the same color
    for y in range(square_size, img_with_colored_boxes.shape[0] - square_size, square_size):
        for x in range(square_size, img_with_colored_boxes.shape[1] - square_size, square_size):
            current_color = img_with_colored_boxes[y, x]

            # Count number of neighboring boxes with the same color
            same_color_neighbors = 0
            for dy in range(-square_size, square_size + 1, square_size):
                for dx in range(-square_size, square_size + 1, square_size):
                    if (dy != 0 or dx != 0) and np.all(img_with_colored_boxes[y+dy, x+dx] == current_color):
                        same_color_neighbors += 1

            # If less than 3 neighbors of the same color, change color to white
            if same_color_neighbors < 5:
                cv2.rectangle(img_with_filtered_color_groups, (x, y), (x+square_size, y+square_size), (255, 255, 255), -1)

    return img_with_filtered_color_groups

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
    "Images\PhotoCam2_1.jpg",  # cpt =  0
    "Images\PhotoCarteRobot5.jpg",  # cpt =  1
    "Images\PhotoCarteRobot9.jpg",  # cpt =  2
    "Images\PhotoCarteRobot1.jpg",  # cpt =  3
    "Images\PhotoCarteRobot2.jpg",  # cpt =  4
    "Images\PhotoCarteRobot3.jpg",  # cpt =  5
    "Images\PhotoCarteRobot6.jpg",  # cpt =  6
    "Images\PhotoCarteRobot8.jpg",  # cpt =  7
    "Images\PhotoCarteRobot10.jpg",  # cpt =  8
    "Images\ConfigBlocsAvecArucoCote1.jpg",  # cpt =  9
    "Images\PhotoCarteTelephone5.jpg",  # cpt =  10
    "Images\PhotoCarteRobot11.jpg",  # cpt =  11
    "Images\ConfigBlocsAvecArucoDessusLoin1.jpg")  # cpt =  12
nbImages = len(nomImages)

frame = cv2.imread(nomImages[0])
# Define the rectangles
rectangles = [
    ((224, 605), (309, 690), (1, 37, 255)),   # Red zone
    ((806, 21), (883, 106), (255, 88, 2)),   # Blue zone
    ((806, 605), (883, 690), (23, 143, 26)),  # Green zone
    ((224, 21), (309, 106), (167, 3, 255))  # Yellow zone
]

cpt = 2

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
        cv2.imwrite("Images\Frame.jpg", frame)
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
                new_center_x = bottom_midpoint_x + int(20 * normalized_direction_vector_x)
                new_center_y = bottom_midpoint_y + int(20 * normalized_direction_vector_y)

                centre = (new_center_x, new_center_y)

                if (marker_id[0] == RB):
                    RBCoords = centre
                    cv2.circle(dst, RBCoords, tailleRB, (0, 0, 0), -1)

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

        # Save the image.
        cv2.imwrite("Images\Transformed_image.jpg", dst)
        cv2.imwrite("Images\Cache.jpg", vector_img)

        cv2.imshow("Cache", vector_img)

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

        # Draw the zones
        cv2.rectangle(cropped_image, (224, 605), (309, 690), (0, 0, 255), 3)  # Red zone
        cv2.rectangle(cropped_image, (806, 21), (883, 106), (255, 0, 0), 3)  # Blue zone
        cv2.rectangle(cropped_image, (806, 605), (883, 690), (0, 255, 0), 3)  # Green zone
        cv2.rectangle(cropped_image, (224, 21), (309, 106), (0, 242, 255), 3)  # Yellow zone

        # Save the cropped image
        cv2.imwrite("Images/cropped_frame.jpg", cropped_image)

        # Adding a Grid
        square_size = 10  # Size of each square in pixels
        grid_img = dst.copy()
        cv2.rectangle(grid_img, (0, 0), final_size, (255, 255, 255), -1)
        grid_img = Grid.add_grid(grid_img, square_size)

        # Displaying different results
        cv2.imshow('Hitbox', dst)
        cv2.imshow('Cropped image', cropped_image)
        cv2.imshow('Grid', grid_img)

        img_copy = dst.copy()

        # Define the centers of the circles
        circle_centers = [(103, 797), (140, 797), (175, 797), (215, 797), (250, 797), (760, 3)]
        circle_centers_grey = [(802, 3), (839, 3), (877, 3), (914, 3), (950, 3), (987, 3)]
        circle_radius = 15
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
        for i in range(6):
            '''print(f"Circle {i + 1}: Mean HSV = {mean_hsv_values[i][0]} / {mean_hsv_values[i][1]} / {mean_hsv_values[i][2]}")
            print(f" Grey Circle {i + 1}: Mean BGR = {mean_bgr_values[i][0]} / {mean_bgr_values[i][1]} / {mean_bgr_values[i][2]}")
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

        hsv_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        for i in range(6):
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
                    [mean_hsv_values[i][0] - 8, mean_hsv_values[i][1] - 100, 0])
                bound_upper = np.array([mean_hsv_values[i][0] + 8, 255, 255])
                magenta_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 4:
                # Red
                bound_lower = np.array(
                    [mean_hsv_values[i][0] - 20, mean_hsv_values[i][1] - 100, 0])
                bound_upper = np.array([mean_hsv_values[i][0], 255, 255])
                red_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 5:
                # White
                bound_lower = np.array(
                    [0, 0, mean_hsv_values[i][2] + 10])
                bound_upper = np.array([180, mean_hsv_values[i][2], 255])
                white_mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
            elif i == 2:
                # Grey
                bound_lower = np.array(
                    [mean_bgr_values[0][0] - 45, mean_bgr_values[0][1] - 45, mean_bgr_values[0][2] - 45])
                bound_upper = np.array(
                    [mean_bgr_values[3][0], mean_bgr_values[3][1], mean_bgr_values[3][1]])
                grey_mask = cv2.inRange(cropped_image, bound_lower, bound_upper)

        kernel = np.ones((7, 7), np.uint8)

        black_mask = cv2.inRange(cropped_image, (0, 0, 0), (1, 1, 1))

        # Definition and displaying of different masks
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

        display_black_mask = cv2.bitwise_and(cropped_image, cropped_image, mask=black_mask)

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

        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        display_white_mask = cv2.bitwise_and(cropped_image, cropped_image, mask=white_mask)
        # Making a new red mask out of magenta and red
        red_mask = cv2.bitwise_or(magenta_mask, red_mask)

        # Display the blended mask
        display_red_mask = cv2.bitwise_and(cropped_image, cropped_image, mask=red_mask)

        cv2.imshow("Blue mask", display_blue_mask)
        cv2.imshow("Red mask", display_red_mask)
        cv2.imshow("Grey mask", display_grey_mask)
        cv2.imshow("White mask", display_white_mask)

        # Combine all masks into a dictionary with mask names as keys
        all_masks = {
            'blue': blue_mask,
            'red': red_mask,
            'grey': grey_mask,
            'white': white_mask,
            'black': black_mask
        }

        # Apply the function to color the boxes
        img_with_colored_boxes = color_boxes_with_masks(cropped_image, all_masks)
        img_with_colored_boxes_corrected = remove_small_color_groups(img_with_colored_boxes)
        img_with_colored_boxes_corrected = Grid.add_grid(img_with_colored_boxes_corrected, square_size)
        for rectangle in rectangles:
            # Extract rectangle coordinates and color
            (x1, y1), (x2, y2), color = rectangle

            # Calculate the center of the rectangle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center = (center_x, center_y)

            # Draw a 20x20 square centered at the calculated center with the same color
            draw_square(img_with_colored_boxes_corrected, center, color)

        cv2.imshow("Colored grid", img_with_colored_boxes_corrected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if TEST != 1:
    cap.release()
cv2.destroyAllWindows()
