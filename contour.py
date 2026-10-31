"""Active contour (greedy snake), chain code, perimeter, area (owner: D)."""
import numpy as np

import canny

# Chain code directions in image coordinates (x to the right, y DOWN):
#   3 2 1
#   4 . 0
#   5 6 7
# Key = (step_x, step_y), value = direction number 0-7.
DIRECTION_FROM_STEP = {
    (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
    (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7,
}

# Smallest gradient^2 range used to normalise the image term (stops noise being amplified).
IMAGE_RANGE_FLOOR = 0.01

# Stop the greedy loop when all points together moved less than this many pixels.
CONVERGENCE_THRESHOLD = 0.5

# Gradient of the last image used, so repeated "Run snake" clicks do not recompute it.
_gradient_cache = {"key": None, "magnitude": None}


# Creates a circle of n_points evenly spaced around the center.
def initialize_contour(center: tuple, radius: int, n_points: int) -> np.ndarray:
    # center: (x, y) pixel the user clicked
    # radius: radius of the starting circle (pixels)
    # n_points: number of snake points; more = finer contour, slower
    # Returns array of shape (n_points, 2) — each row is (x, y)
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    if radius <= 0:
        raise ValueError("radius must be positive")
    center_x, center_y = center
    angles = 2 * np.pi * np.arange(n_points) / n_points
    x_coordinates = center_x + radius * np.cos(angles)
    y_coordinates = center_y + radius * np.sin(angles)
    return np.column_stack((x_coordinates, y_coordinates))


# Converts an RGB (or grayscale) image to a float grayscale image.
def _to_grayscale(img: np.ndarray) -> np.ndarray:
    # img: RGB image (H, W, 3) or grayscale image (H, W)
    if img.ndim == 3 and img.shape[2] >= 3:
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    if img.ndim == 2:
        return img.astype(np.float64)
    raise ValueError("img must be a grayscale or RGB image")


# Builds the gradient magnitude image (scaled so its maximum is 1) using B's canny functions.
def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    # img: image as loaded by the UI; the result is cached for repeated runs on the same image
    key = (img.shape, hash(img.tobytes()))
    if _gradient_cache["key"] == key:
        return _gradient_cache["magnitude"]
    blurred_image = canny.gaussian_blur(_to_grayscale(img), 5, 1.4)
    magnitude, _ = canny.gradient(blurred_image)
    maximum = magnitude.max()
    magnitude = magnitude / maximum if maximum > 0 else magnitude
    _gradient_cache["key"] = key
    _gradient_cache["magnitude"] = magnitude
    return magnitude


# Computes the three raw energy terms for each candidate position of a point.
def _energy_terms(positions: np.ndarray, previous_points: np.ndarray,
                  next_points: np.ndarray, gradient_magnitude: np.ndarray):
    # positions: (k, 2) positions (x, y) being evaluated for the point v_i
    # previous_points: v_{i-1}, shape (2,) or (k, 2)
    # next_points: v_{i+1}, shape (2,) or (k, 2)
    # gradient_magnitude: image whose bright pixels are edges
    height, width = gradient_magnitude.shape
    columns = np.clip(np.round(positions[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(positions[:, 1]).astype(int), 0, height - 1)
    image_energy = -(gradient_magnitude[rows, columns] ** 2)
    elastic_energy = np.sum((positions - previous_points) ** 2, axis=1)
    smooth_energy = np.sum((previous_points - 2 * positions + next_points) ** 2, axis=1)
    return image_energy, elastic_energy, smooth_energy


# Computes total energy of the contour (image energy + internal energy).
def compute_energy(contour: np.ndarray, img: np.ndarray, alpha: float, beta: float) -> float:
    # contour: (n_points, 2) array of (x, y)
    # img: gradient magnitude image (edges = bright), NOT the raw image
    # alpha: elasticity weight; higher = points stay evenly spaced, contour shrinks
    # beta: smoothness weight; higher = fewer sharp corners
    previous_points = np.roll(contour, 1, axis=0)
    next_points = np.roll(contour, -1, axis=0)
    image_energy, elastic_energy, smooth_energy = _energy_terms(
        contour, previous_points, next_points, img)
    return float(np.sum(image_energy + alpha * elastic_energy + beta * smooth_energy))


# Rescales values to 0..1 (lowest value -> 0); a tiny range stays small instead of being stretched.
def _normalize(values: np.ndarray, minimum_range: float = 0.0) -> np.ndarray:
    # values: energies of all candidate positions of one point
    # minimum_range: ranges smaller than this are treated as this size
    value_range = max(values.max() - values.min(), minimum_range)
    if value_range <= 0:
        return np.zeros_like(values)
    return (values - values.min()) / value_range


# Lists every position in the window around a point, clipped to the image.
def _window_positions(point: np.ndarray, window_size: int, height: int, width: int) -> np.ndarray:
    # point: current (x, y) of the snake point
    # window_size: side of the square search window (odd)
    half = window_size // 2
    offsets = np.arange(-half, half + 1)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    positions = np.column_stack((point[0] + offset_x.ravel(), point[1] + offset_y.ravel()))
    positions[:, 0] = np.clip(positions[:, 0], 0, width - 1)
    positions[:, 1] = np.clip(positions[:, 1], 0, height - 1)
    return positions


# Moves one point to the lowest-energy position in its window and returns that position.
def _best_position(contour: np.ndarray, index: int, gradient_magnitude: np.ndarray,
                   alpha: float, beta: float, window_size: int) -> np.ndarray:
    # contour: current snake; index: which point to move
    # gradient_magnitude, alpha, beta, window_size: see evolve_contour
    height, width = gradient_magnitude.shape
    point_count = len(contour)
    previous_point = contour[(index - 1) % point_count]   # modular indexing: the loop wraps around
    next_point = contour[(index + 1) % point_count]
    candidates = _window_positions(contour[index], window_size, height, width)
    image_energy, elastic_energy, smooth_energy = _energy_terms(
        candidates, previous_point, next_point, gradient_magnitude)
    total_energy = (_normalize(image_energy, IMAGE_RANGE_FLOOR)
                    + alpha * _normalize(elastic_energy)
                    + beta * _normalize(smooth_energy))
    return candidates[np.argmin(total_energy)]


# Runs the greedy algorithm: moves each point to minimize energy.
def evolve_contour(contour: np.ndarray, img: np.ndarray, alpha: float, beta: float,
                   n_iterations: int, window_size: int) -> np.ndarray:
    # contour: initial (n_points, 2) array
    # img: RGB image as loaded by the UI (convert to grayscale inside)
    # alpha, beta: see compute_energy
    # n_iterations: maximum number of greedy passes
    # window_size: side of the search neighbourhood per point (odd); bigger = moves farther per step
    # Returns the final evolved contour
    if window_size % 2 == 0:
        window_size += 1
    gradient_magnitude = _gradient_magnitude(img)
    snake = np.array(contour, dtype=np.float64)
    for _ in range(n_iterations):
        total_movement = 0.0
        for index in range(len(snake)):
            new_position = _best_position(snake, index, gradient_magnitude,
                                          alpha, beta, window_size)
            total_movement += np.linalg.norm(new_position - snake[index])
            snake[index] = new_position
        if total_movement < CONVERGENCE_THRESHOLD:
            break
    return snake


# Converts contour points to 8-direction chain code (0–7).
def to_chain_code(contour: np.ndarray) -> list[int]:
    # contour: (n_points, 2) array of (x, y); points may be several pixels apart,
    #          so step between neighbours one pixel at a time
    pixel_points = np.round(contour).astype(int)
    chain_code = []
    for index in range(len(pixel_points)):
        current_x, current_y = pixel_points[index]
        target_x, target_y = pixel_points[(index + 1) % len(pixel_points)]  # last point links to first
        while (current_x, current_y) != (target_x, target_y):
            step_x = int(np.sign(target_x - current_x))  # normalise dx to -1, 0 or 1
            step_y = int(np.sign(target_y - current_y))  # normalise dy to -1, 0 or 1
            chain_code.append(DIRECTION_FROM_STEP[(step_x, step_y)])
            current_x += step_x
            current_y += step_y
    return chain_code


# Computes perimeter from chain code (even codes = 1, odd codes = sqrt(2)).
def compute_perimeter(chain_code: list[int]) -> float:
    # chain_code: list of directions 0–7
    odd_count = sum(1 for direction in chain_code if direction % 2 == 1)
    even_count = len(chain_code) - odd_count
    return even_count * 1.0 + odd_count * float(np.sqrt(2))


# Computes area enclosed by the contour using the shoelace formula.
def compute_area(contour: np.ndarray) -> float:
    # contour: (n_points, 2) array of (x, y)
    x_coordinates = contour[:, 0]
    y_coordinates = contour[:, 1]
    next_x = np.roll(x_coordinates, -1)  # wrap: last point connects back to the first
    next_y = np.roll(y_coordinates, -1)
    signed_area = 0.5 * np.sum(x_coordinates * next_y - next_x * y_coordinates)
    return float(abs(signed_area))