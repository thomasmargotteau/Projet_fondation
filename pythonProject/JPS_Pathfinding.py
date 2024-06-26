import numpy as np
import cv2
from heapq import heappop, heappush


def jps_algorithm(grid, start, goal, square_size, image=None):
    """
    Implement the Jump Point Search (JPS*) algorithm to find the shortest path from start to goal on the grid.

    Parameters:
        grid (numpy.ndarray): 2D grid representing the environment with obstacles marked as 1 and free cells as 0.
        start (tuple): Coordinates of the starting point (row, column).
        goal (tuple): Coordinates of the goal point (row, column).
        square_size (int): Size of each square in the grid.
        image (numpy.ndarray): Optional. Image on which the path will be drawn.

    Returns:
        path (list of tuples): List of coordinates representing the shortest path from start to goal.
    """

    # Define function to calculate Manhattan distance between two points
    def manhattan_distance(point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    # Implement JPS* algorithm here
    def is_valid_cell(row, col, parent):
        """Check if the cell (row, col) is within the grid and is a valid (free) cell."""
        if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
            return False
        if grid[row, col] == 2:  # Obstacle
            return False
        return True

    def successors(row, col, parent):
        """Generate successor cells for the given cell using JPS*."""
        successors_list = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),  # Horizontal and vertical
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonal

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if not is_valid_cell(r, c, (row, col)):
                continue
            if is_valid_cell(r, c, (row, col)):
                successors_list.append((r, c))
            else:
                jump_point = jump(r, c, (row, col))
                if jump_point:
                    successors_list.append(jump_point)
        return successors_list

    def jump(row, col, parent):
        dr = row - parent[0]
        dc = col - parent[1]
        if not is_valid_cell(row, col, parent):
            return None
        if (row, col) == goal:
            return row, col
        if dr != 0 and dc != 0:  # Diagonal move
            if (is_valid_cell(row - dr, col, (row, col)) and not is_valid_cell(row - dr, col - dc, (row, col))) or \
                    (is_valid_cell(row, col - dc, (row, col)) and not is_valid_cell(row - dr, col - dc, (row, col))):
                return row, col
        else:  # Horizontal or vertical move
            if dr != 0:  # Vertical move
                if (is_valid_cell(row, col + 1, (row, col)) and not is_valid_cell(row - dr, col + 1, (row, col))) or \
                        (is_valid_cell(row, col - 1, (row, col)) and not is_valid_cell(row - dr, col - 1, (row, col))):
                    return row, col
            else:  # Horizontal move
                if (is_valid_cell(row + 1, col, (row, col)) and not is_valid_cell(row + 1, col - dc, (row, col))) or \
                        (is_valid_cell(row - 1, col, (row, col)) and not is_valid_cell(row - 1, col - dc, (row, col))):
                    return row, col
        return jump(row + dr, col + dc, (row, col))

    open_set = [(manhattan_distance(start, goal), start)]
    closed_set = set()
    parent = {}
    g_values = {start: 0}

    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()

            # Filter the path to store one out of every three points
            filtered_path = [path[i] for i in range(0, len(path), 3)]

            if image is not None:
                draw_path(image, filtered_path, square_size)

            return filtered_path

        closed_set.add(current)
        for successor in successors(*current, current):
            if successor in closed_set:
                continue
            tentative_g = g_values[current] + 1
            if successor not in open_set or tentative_g < g_values.get(successor, float('inf')):
                parent[successor] = current
                g_values[successor] = tentative_g
                f_value = tentative_g + manhattan_distance(successor, goal)
                heappush(open_set, (f_value, successor))

    return []



def draw_path(image, path, square_size):
    """
    Draw the path on the provided image.

    Parameters:
        image (numpy.ndarray): Image on which the path will be drawn.
        path (list of tuples): List of coordinates representing the path.
        square_size (int): Size of each square in the grid.
    """
    # Define colors for drawing the path
    color = (0, 255, 0)  # Green color for the path
    arrow_color = (255, 0, 0)

    # Loop through each point in the path
    for i in range(len(path) - 1):
        # Calculate the coordinates of the current and next points in the path
        current_point = (path[i][1] * square_size - square_size // 2, path[i][0] * square_size - square_size // 2)
        next_point = (path[i + 1][1] * square_size - square_size // 2, path[i + 1][0] * square_size - square_size // 2)

        # Draw a line connecting the current and next points on the image
        cv2.line(image, current_point, next_point, color, thickness=1)
        # Calculate angle and draw arrow head
        angle = np.arctan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
        p1 = (int(next_point[0] - 6 * np.cos(angle - np.pi / 6)),
              int(next_point[1] - 6 * np.sin(angle - np.pi / 6)))
        p2 = (int(next_point[0] - 6 * np.cos(angle + np.pi / 6)),
              int(next_point[1] - 6 * np.sin(angle + np.pi / 6)))
        cv2.line(image, next_point, p1, arrow_color, 1)
        cv2.line(image, next_point, p2, arrow_color, 1)
    return image


def find_color_centers(grid, target_color):
    Color_centers = []
    visited = set()

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y][x] == target_color and (y, x) not in visited:
                # Find the cluster of red boxes
                cluster = []
                queue = [(y, x)]
                while queue:
                    node_y, node_x = queue.pop(0)
                    if grid[node_y][node_x] == target_color and (node_y, node_x) not in visited:
                        visited.add((node_y, node_x))
                        cluster.append((node_y, node_x))
                        # Check neighbors
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                if (0 <= node_y + dy < grid.shape[0] and
                                        0 <= node_x + dx < grid.shape[1] and
                                        grid[node_y + dy][node_x + dx] == target_color):
                                    queue.append((node_y + dy, node_x + dx))
                # Find the center of the cluster
                if cluster:
                    sum_y = sum(node[0] for node in cluster)
                    sum_x = sum(node[1] for node in cluster)
                    center_y = (sum_y // len(cluster)) + 1
                    center_x = (sum_x // len(cluster)) + 1
                    Color_centers.append((center_y, center_x))

    return Color_centers

def replace_cluster_with_white(img_with_colored_boxes, grid, cluster_centers, square_size):
    img_with_white_clusters = img_with_colored_boxes.copy()
    for center_y, center_x in cluster_centers:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y = center_y + dy
                x = center_x + dx
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] in [1, 2, 3, 4]:
                    grid[y, x] = 0  # Change the value in the grid to white
                    cv2.rectangle(img_with_white_clusters, (x * square_size, y * square_size),
                                  ((x + 1) * square_size, (y + 1) * square_size), (255, 255, 255), -1)
    return img_with_white_clusters, grid


def draw_cluster_centers_and_update_grid(img_with_colored_boxes, grid, cluster_centers, square_size):
    img_with_centers = img_with_colored_boxes.copy()
    for center_y, center_x in cluster_centers:
        # Calculate the center and radius in pixels
        circle_center = ((center_x) * square_size + (square_size // 2),
                         (center_y) * square_size + (square_size // 2))
        radius = 7 * square_size

        # Draw the circle on the image
        cv2.circle(img_with_centers, circle_center, radius, (255, 3, 255), 2)

        # Update the grid and image for boxes within the circle
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                # Calculate the box center in pixels
                box_center = ((x * square_size) + (square_size // 2),
                              (y * square_size) + (square_size // 2))
                # Check if the box center is within the circle
                if (box_center[0] - circle_center[0]) ** 2 + (box_center[1] - circle_center[1]) ** 2 <= radius ** 2:
                    grid[y, x] = 2  # Set the value in the grid to grey
                    cv2.rectangle(img_with_centers, (x * square_size, y * square_size),
                                  ((x + 1) * square_size, (y + 1) * square_size), (100, 100, 100), -1)

    return img_with_centers, grid


def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def find_closest_point(points, square_size, start=None):
    if start:
        start = [coord * square_size for coord in start]
    else:
        start = (0, 0)
    closest_point = min(points, key=lambda point: manhattan_distance(point, start))
    return closest_point
