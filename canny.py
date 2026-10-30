"""Canny edge detection, implemented from scratch (owner: B).

Pipeline: gaussian_blur -> gradient -> non_maximum_suppression
          -> double_threshold -> hysteresis
"""
import numpy as np


# Step 1: smooth the image with a Gaussian kernel to reduce noise.
def gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    # img: grayscale image (2D array)
    # kernel_size: width/height of the Gaussian kernel (odd number); bigger = more blur
    # sigma: spread of the Gaussian; bigger = stronger smoothing, fewer noisy edges
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    image = img.astype(np.float64)
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1)
    x_coordinates, y_coordinates = np.meshgrid(coordinates, coordinates)

    kernel = np.exp(
        -(x_coordinates ** 2 + y_coordinates ** 2) / (2 * sigma ** 2)
    )
    kernel = kernel / kernel.sum()

    padded_image = np.pad(image, radius, mode="reflect")
    blurred_image = np.zeros_like(image)

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            local_region = padded_image[
                row:row + kernel_size,
                column:column + kernel_size
            ]
            blurred_image[row, column] = np.sum(local_region * kernel)

    return blurred_image



# Step 2: compute gradient magnitude and direction (e.g. with Sobel kernels).
def gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # img: blurred grayscale image
    # Returns: (magnitude, direction) — direction in radians
    image = img.astype(np.float64)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    padded_image = np.pad(image, 1, mode="reflect")
    gradient_x = np.zeros_like(image)
    gradient_y = np.zeros_like(image)

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            local_region = padded_image[row:row + 3, column:column + 3]
            gradient_x[row, column] = np.sum(local_region * sobel_x)
            gradient_y[row, column] = np.sum(local_region * sobel_y)

    magnitude = np.hypot(gradient_x, gradient_y)
    direction = np.arctan2(gradient_y, gradient_x)
    return magnitude, direction




    


# Step 3: thin edges to 1 pixel wide by keeping only local maxima along the gradient.
def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    # magnitude: gradient strength per pixel
    # direction: gradient angle per pixel (radians)
    angle_degrees = np.rad2deg(direction) % 180
    thinned_image = np.zeros_like(magnitude)
    height, width = magnitude.shape

    for row in range(1, height - 1):
        for column in range(1, width - 1):
            angle = angle_degrees[row, column]

            if angle < 22.5 or angle >= 157.5:
                first_neighbor = magnitude[row, column - 1]
                second_neighbor = magnitude[row, column + 1]
            elif angle < 67.5:
                first_neighbor = magnitude[row - 1, column + 1]
                second_neighbor = magnitude[row + 1, column - 1]
            elif angle < 112.5:
                first_neighbor = magnitude[row - 1, column]
                second_neighbor = magnitude[row + 1, column]
            else:
                first_neighbor = magnitude[row - 1, column - 1]
                second_neighbor = magnitude[row + 1, column + 1]

            if magnitude[row, column] >= first_neighbor and magnitude[row, column] >= second_neighbor:
                thinned_image[row, column] = magnitude[row, column]

    return thinned_image


# Step 4: classify each pixel as strong, weak, or suppressed.
def double_threshold(img: np.ndarray, low: float, high: float) -> np.ndarray:
    # img: thinned magnitude image from NMS
    # low: below this a pixel is suppressed; lower = more weak edges kept
    # high: above this a pixel is a strong edge; lower = more strong edges
    if low < 0 or high < 0 or low >= high:
        raise ValueError("Thresholds must satisfy 0 <= low < high")

    weak_value = 75
    strong_value = 255
    thresholded_image = np.zeros_like(img, dtype=np.uint8)
    strong_pixels = img >= high
    weak_pixels = (img >= low) & (img < high)

    thresholded_image[strong_pixels] = strong_value
    thresholded_image[weak_pixels] = weak_value
    return thresholded_image


# Step 5: keep weak edges only if they connect to a strong edge.
def hysteresis(img: np.ndarray) -> np.ndarray:
    # img: output of double_threshold (strong / weak / zero pixels)
    weak_value = 75
    strong_value = 255
    edge_image = img.copy()
    height, width = edge_image.shape
    strong_rows, strong_columns = np.where(edge_image == strong_value)
    pixels_to_visit = list(zip(strong_rows, strong_columns))

    while pixels_to_visit:
        row, column = pixels_to_visit.pop()
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                inside_image = 0 <= neighbor_row < height and 0 <= neighbor_column < width
                if inside_image and edge_image[neighbor_row, neighbor_column] == weak_value:
                    edge_image[neighbor_row, neighbor_column] = strong_value
                    pixels_to_visit.append((neighbor_row, neighbor_column))

    edge_image[edge_image != strong_value] = 0
    return edge_image


# Runs all 5 Canny steps in order and returns the final binary edge image.
def run_canny(img: np.ndarray, kernel_size: int, sigma: float, low: float, high: float) -> np.ndarray:
    # img: input image (RGB or grayscale)
    # kernel_size, sigma: passed to gaussian_blur
    # low, high: percentages of the strongest post-NMS gradient
    # This is the only function the UI calls.
    if img.ndim == 3 and img.shape[2] >= 3:
        grayscale_image = (
            0.299 * img[:, :, 0]
            + 0.587 * img[:, :, 1]
            + 0.114 * img[:, :, 2]
        )
    elif img.ndim == 2:
        grayscale_image = img
    else:
        raise ValueError("img must be a grayscale or RGB image")

    blurred_image = gaussian_blur(grayscale_image, kernel_size, sigma)
    magnitude, direction = gradient(blurred_image)
    thinned_image = non_maximum_suppression(magnitude, direction)
    if low < 0 or high > 100 or low >= high:
        raise ValueError("Threshold percentages must satisfy 0 <= low < high <= 100")

    maximum_gradient = thinned_image.max()
    if maximum_gradient == 0:
        return np.zeros_like(thinned_image, dtype=np.uint8)

    low_threshold = maximum_gradient * low / 100
    high_threshold = maximum_gradient * high / 100
    thresholded_image = double_threshold(thinned_image, low_threshold, high_threshold)
    return hysteresis(thresholded_image)
