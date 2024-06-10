import cv2
import numpy as np


def add_grid(image, square_size):
    # Read the image
    img = image

    # Get image dimensions
    img_height = 800
    img_width = 1200

    # Draw vertical lines
    for x in range(square_size, img_width, square_size):
        cv2.line(img, (x, 0), (x, img_height), (0, 0, 0), 1)

    # Draw horizontal lines
    for y in range(square_size, img_height, square_size):
        cv2.line(img, (0, y), (img_width, y), (0, 0, 0), 1)
    return img


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
        'black': (0, 0, 0),
        'yellow_zone': (167, 3, 255),  # Yellow for marking 6
        'blue_zone': (255, 88, 2),  # Blue for marking 7
        'red_zone': (1, 37, 255),  # Red for marking 8
        'green_zone': (23, 143, 26)  # Green for marking 9
    }

    # Define values for each color in the grid
    color_values = {
        (255, 255, 255): 0,  # nothing
        (255, 0, 0): 1,  # blue
        (100, 100, 100): 2,  # grey
        (145, 5, 83): 3,  # white
        (0, 0, 255): 4,  # red
        (0, 0, 0): 5,  # black
        (167, 3, 255): 6,  # yellow zone
        (255, 88, 2): 7,  # blue zone
        (1, 37, 255): 8,  # red zone
        (23, 143, 26): 9  # green zone
    }

    # Create an empty grid
    grid = np.zeros((81, 121), dtype=np.uint8)

    # Iterate over each grid cell
    square_size = 10  # Size of each square in pixels
    for y in range(0, img_with_colored_boxes.shape[0], square_size):
        for x in range(0, img_with_colored_boxes.shape[1], square_size):
            # Check if any mask has a non-zero value within the current grid cell
            mask_color = (255, 255, 255)  # Default color if no mask is present
            for mask_name, mask in masks.items():
                if np.any(mask[y:y + square_size, x:x + square_size]):
                    # If any mask has a non-zero value, color the box with the corresponding color
                    mask_color = colors[mask_name]
                    break
            # Color each box
            cv2.rectangle(img_with_colored_boxes, (x, y), (x + square_size, y + square_size), mask_color, -1)

            # Fill the corresponding value in the grid
            grid_y = y // square_size
            grid_x = x // square_size

            # Check if the current box falls within the specified regions
            if 7 <= grid_y <= 14 and 27 <= grid_x <= 34:
                grid[grid_y, grid_x] = color_values[(167, 3, 255)]  # Mark as yellow zone
                cv2.rectangle(img_with_colored_boxes, (x, y), (x + square_size, y + square_size), (0, 242, 255), -1)
            elif 7 <= grid_y <= 14 and 84 <= grid_x <= 91:
                grid[grid_y, grid_x] = color_values[(255, 88, 2)]  # Mark as blue zone
                cv2.rectangle(img_with_colored_boxes, (x, y), (x + square_size, y + square_size), (232, 162, 0), -1)
            elif 65 <= grid_y <= 72 and 27 <= grid_x <= 34:
                grid[grid_y, grid_x] = color_values[(1, 37, 255)]  # Mark as red zone
                cv2.rectangle(img_with_colored_boxes, (x, y), (x + square_size, y + square_size), (36, 28, 237), -1)
            elif 65 <= grid_y <= 72 and 84 <= grid_x <= 91:
                grid[grid_y, grid_x] = color_values[(23, 143, 26)]  # Mark as green zone
                cv2.rectangle(img_with_colored_boxes, (x, y), (x + square_size, y + square_size), (76, 177, 34), -1)
            else:
                # Mark other boxes according to their colors
                grid[grid_y, grid_x] = color_values[mask_color]
            if 0 <= grid_y <= 2 and 118 <= grid_x <= 121:
                grid[grid_y, grid_x] = color_values[(100, 100, 100)]  # Mark as green zone
                cv2.rectangle(img_with_colored_boxes, (x, y), (x + square_size, y + square_size), (100, 100, 100), -1)
    return img_with_colored_boxes, grid




def remove_small_color_groups(img_with_colored_boxes, grid):
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
                    if (dy != 0 or dx != 0) and np.all(img_with_colored_boxes[y + dy, x + dx] == current_color):
                        same_color_neighbors += 1

            # If less than 3 neighbors of the same color, change color to white
            if same_color_neighbors < 2:
                cv2.rectangle(img_with_filtered_color_groups, (x, y), (x + square_size, y + square_size),
                              (255, 255, 255), -1)
                grid_y = y // square_size
                grid_x = x // square_size
                grid[grid_y, grid_x] = 0

    return img_with_filtered_color_groups, grid


