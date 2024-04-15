import cv2
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
