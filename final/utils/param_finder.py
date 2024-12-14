import cv2
import numpy as np
import json
import os

def load_homography_matrix(file_path):
    """Load homography matrix from a JSON file."""
    with open(file_path, 'r') as f:
        matrix_list = json.load(f)
    return np.array(matrix_list)

def reverse_transform_image_with_homography(transformed_image, homography_matrix):
    """Reverse the homography transformation on an image."""
    inverse_homography_matrix = np.linalg.inv(homography_matrix)
    h, w = transformed_image.shape[:2]
    original_image = cv2.warpPerspective(transformed_image, inverse_homography_matrix, (w, h))
    return original_image

def color_thresholding(image, lower_bounds, upper_bounds):
    """Perform color thresholding to segment the globe."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        mask |= cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

def update_hough_circles(*args):
    """
    Update Hough Circle detection with current slider parameters.
    """
    dp = max(cv2.getTrackbarPos('dp', 'HoughCircles') / 10.0, 0.1)  # Ensure dp > 0
    minDist = max(cv2.getTrackbarPos('minDist', 'HoughCircles'), 1)  # Ensure minDist > 0
    param1 = max(cv2.getTrackbarPos('param1', 'HoughCircles'), 1)    # Ensure param1 > 0
    param2 = max(cv2.getTrackbarPos('param2', 'HoughCircles') / 100.0, 0.01)  # Ensure param2 > 0
    minRadius = max(cv2.getTrackbarPos('minRadius', 'HoughCircles'), 0)  # Ensure minRadius >= 0
    maxRadius = max(cv2.getTrackbarPos('maxRadius', 'HoughCircles'), minRadius)  # Ensure maxRadius >= minRadius


    # Convert to grayscale and blur
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_blurred = cv2.blur(gray_image, (3, 3))

    # Detect circles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT_ALT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    # Clone the original image for visualization
    output_image = image.copy()

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, radius) in circles:
            cv2.circle(output_image, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)

    # Display the output
    cv2.imshow('HoughCircles', output_image)

# Setup paths
image_path = './final/calibrated-images/000000.png'  # Replace with actual path
homography_path = './final/calibrated-images/000000.json'  # Replace with the correct path

# Bounds for color thresholding
lower_bounds = [np.array([100, 50, 50])]
upper_bounds = [np.array([140, 255, 255])]

# Load the input image and homography matrix
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

homography_matrix = load_homography_matrix(homography_path)

# Create a window and sliders
cv2.namedWindow('HoughCircles')

# Add sliders
cv2.createTrackbar('dp', 'HoughCircles', 12, 50, update_hough_circles)  # dp slider scaled directly
cv2.createTrackbar('minDist', 'HoughCircles', 50, 500, update_hough_circles)  # minDist slider
cv2.createTrackbar('param1', 'HoughCircles', 100, 500, update_hough_circles)  # param1 slider
cv2.createTrackbar('param2', 'HoughCircles', 70, 100, update_hough_circles)   # param2 slider (scaled by 100)
cv2.createTrackbar('minRadius', 'HoughCircles', 0, 200, update_hough_circles)  # minRadius slider
cv2.createTrackbar('maxRadius', 'HoughCircles', 100, 500, update_hough_circles)  # maxRadius slider

# Initial update
update_hough_circles()

# Wait for user interaction
cv2.waitKey(0)
cv2.destroyAllWindows()
