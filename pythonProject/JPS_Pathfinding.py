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
        image (numpy.ndarray): Optional. Image on which the path will be drawn.

    Returns:
        path (list of tuples): List of coordinates representing the shortest path from start to goal.
    """

    # Define function to calculate Manhattan distance between two points
    def manhattan_distance(point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    # Implement JPS* algorithm here
    def is_valid_cell(row, col):
        """Check if the cell (row, col) is within the grid and is a valid (free) cell."""
        # Check if the cell is within the grid boundaries
        if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
            return False
        # Check if the cell is a valid (free) cell
        if grid[row, col] == 2:  # Obstacle
            return False
        return True

    def successors(row, col):
        """Generate successor cells for the given cell using JPS*."""
        successors_list = []

        # Define directional offsets for horizontal, vertical, and diagonal moves
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),  # Horizontal and vertical
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonal

        # Loop through each direction
        for dr, dc in directions:
            r, c = row + dr, col + dc

            # If the cell in this direction is not valid, continue to the next direction
            if not is_valid_cell(r, c):
                continue

            # Check if the cell is a forced neighbor or a jump point
            if grid[r, c] == 0:  # Free cell
                successors_list.append((r, c))
            else:  # Obstacle or boundary
                jump_point = jump(r, c, (row, col))
                if jump_point:
                    successors_list.append(jump_point)

        return successors_list

    def jump(row, col, parent):
        """Jump to the next jump point."""
        # Calculate the direction of the jump
        dr = row - parent[0]
        dc = col - parent[1]

        # If the parent cell is not valid, return None
        if not is_valid_cell(row, col):
            return None

        # If the current cell is the goal, return it as a jump point
        if (row, col) == goal:
            return row, col

        # Check if the cell is a forced neighbor
        if dr != 0 and dc != 0:  # Diagonal move
            if (is_valid_cell(row - dr, col) and not is_valid_cell(row - dr, col - dc)) or \
                    (is_valid_cell(row, col - dc) and not is_valid_cell(row - dr, col - dc)):
                return row, col
        else:  # Horizontal or vertical move
            if dr != 0:  # Vertical move
                if (is_valid_cell(row, col + 1) and not is_valid_cell(row - dr, col + 1)) or \
                        (is_valid_cell(row, col - 1) and not is_valid_cell(row - dr, col - 1)):
                    return row, col
            else:  # Horizontal move
                if (is_valid_cell(row + 1, col) and not is_valid_cell(row + 1, col - dc)) or \
                        (is_valid_cell(row - 1, col) and not is_valid_cell(row - 1, col - dc)):
                    return row, col

        # Recursively search for jump points in the direction of the jump
        return jump(row + dr, col + dc, (row, col))

    # Initialize the open set with the start node
    open_set = [(manhattan_distance(start, goal), start)]
    # Initialize the closed set
    closed_set = set()

    # Initialize dictionaries to store parent and g-values
    parent = {}
    g_values = {start: 0}

    # Run the main loop of the algorithm
    while open_set:
        # Select the node with the lowest f-value from the open set
        _, current = heappop(open_set)

        # Check if the goal is reached
        if current == goal:
            # Reconstruct the path
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()

            # If an image is provided, draw the path on the image
            if image is not None:
                draw_path(image, path, square_size)

            return path

        # Remove the current node from the open set and add it to the closed set
        closed_set.add(current)

        # Generate successors for the current node
        for successor in successors(*current):
            # Check if the successor is in the closed set
            if successor in closed_set:
                continue

            # Calculate the tentative g-value for the successor
            tentative_g = g_values[current] + 1

            # Check if the successor is not in the open set or has a lower g-value
            if successor not in open_set or tentative_g < g_values.get(successor, float('inf')):
                # Update the parent and g-value for the successor
                parent[successor] = current
                g_values[successor] = tentative_g

                # Calculate the f-value (f = g + h)
                f_value = tentative_g + manhattan_distance(successor, goal)

                # Add the successor to the open set with its f-value
                heappush(open_set, (f_value, successor))

    # If no path is found, return an empty list
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
        p1 = (int(next_point[0] - 8 * np.cos(angle - np.pi / 6)),
              int(next_point[1] - 8 * np.sin(angle - np.pi / 6)))
        p2 = (int(next_point[0] - 8 * np.cos(angle + np.pi / 6)),
              int(next_point[1] - 8 * np.sin(angle + np.pi / 6)))
        cv2.line(image, next_point, p1, arrow_color, 2)
        cv2.line(image, next_point, p2, arrow_color, 2)
    return image



def find_color_centers(grid, target_color):
    red_centers = []
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
                    red_centers.append((center_y, center_x))

    return red_centers

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def find_closest_point(points, square_size, start=None):
    if start:
        start = [coord * square_size for coord in start]
    else:
        start = (0, 0)
    closest_point = min(points, key=lambda point: manhattan_distance(point, start))
    return closest_point