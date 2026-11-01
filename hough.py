"""Hough transform (owner: C).

Lines, circles (2D accumulator per radius), ellipses (Xie & Ji 2002).
All voting is vectorised with NumPy (np.bincount) instead of Python loops.
draw_shapes is the only drawing function in the project.
"""
import numpy as np
import cv2


# Max number of (edge point x candidate) votes built in memory at once.
VOTE_CHUNK = 2_000_000

# Most angles each radius votes with; a full circle is estimated from them (faster for big radii).
MAX_VOTE_ANGLES = 120

# A circle needs at least this fraction of its outline present, so short arcs are not circles.
MIN_CIRCLE_COVERAGE = 0.4

# Peaks only need this fraction of the required votes to be checked exactly (cheap pre-filter).
SCREENING_FRACTION = 0.5


# Finds peaks in a 2D accumulator that exceed threshold, suppressing nearby duplicates.
def _find_peaks(accumulator, threshold, nms_window=9):
    # accumulator: 2D vote array
    # threshold: minimum votes to qualify as a peak
    # nms_window: side of the local-maximum window (odd); peaks within this area are merged
    half = nms_window // 2
    peaks = []
    # Only cells above the threshold are checked (much faster than scanning every cell)
    for row, column in np.argwhere(accumulator >= threshold):
        region = accumulator[max(0, row - half):row + half + 1,
                             max(0, column - half):column + half + 1]
        if accumulator[row, column] == region.max():
            peaks.append((int(row), int(column)))
    return peaks


# --- 1. Lines -------------------------------------------------------------

# Detects straight lines and returns them as (rho, theta) pairs.
def hough_lines(edge_img: np.ndarray, threshold: int) -> list[tuple]:
    # edge_img: binary edge image from Canny
    # threshold: minimum accumulator votes to accept a line; lower = more lines
    height, width = edge_img.shape[:2]
    diagonal = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

    theta_count = 180
    thetas = np.linspace(0, np.pi, theta_count, endpoint=False)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    theta_indices = np.arange(theta_count)

    # Accumulator: rows = rho bins (-diagonal ... +diagonal), columns = theta bins
    rho_bins = 2 * diagonal + 1
    flat_accumulator = np.zeros(rho_bins * theta_count, dtype=np.int64)

    edge_points = np.argwhere(edge_img > 0)  # each row is (y, x)
    chunk_size = max(1, VOTE_CHUNK // theta_count)

    # All edge pixels in a chunk vote for all thetas at once
    for start in range(0, len(edge_points), chunk_size):
        chunk = edge_points[start:start + chunk_size]
        rho_values = chunk[:, 1:2] * cos_thetas + chunk[:, 0:1] * sin_thetas  # (N, 180)
        rho_indices = np.round(rho_values).astype(np.int64) + diagonal
        flat_indices = rho_indices * theta_count + theta_indices
        flat_accumulator += np.bincount(flat_indices.ravel(), minlength=flat_accumulator.size)

    accumulator = flat_accumulator.reshape(rho_bins, theta_count)
    lines = []
    for rho_index, theta_index in _find_peaks(accumulator, threshold):
        lines.append((rho_index - diagonal, thetas[theta_index]))
    return lines


# --- 2. Circles -----------------------------------------------------------

# Builds the 2D centre accumulator for one radius (every edge pixel votes at once).
def _circle_accumulator(edge_points, radius, height, width):
    # edge_points: array of (y, x) edge pixels
    # radius: circle radius being tested (pixels)
    # height, width: image size (centres outside it are ignored)
    # Returns (accumulator, pixels_per_vote, circle_pixels): a vote counts as pixels_per_vote
    # edge pixels, and a full circle of this radius has circle_pixels pixels
    full_angles = max(8, int(round(2 * np.pi * radius)))
    circle_pixels = len(_circle_offsets(radius, full_angles))
    offsets = _circle_offsets(radius, min(full_angles, MAX_VOTE_ANGLES))
    pixels_per_vote = circle_pixels / len(offsets)
    offset_x, offset_y = offsets[:, 0], offsets[:, 1]

    flat_accumulator = np.zeros(height * width, dtype=np.int64)
    chunk_size = max(1, VOTE_CHUNK // len(offsets))
    for start in range(0, len(edge_points), chunk_size):
        chunk = edge_points[start:start + chunk_size]
        center_x = chunk[:, 1:2] - offset_x  # (N, offsets)
        center_y = chunk[:, 0:1] - offset_y
        inside = ((center_x >= 0) & (center_x < width) &
                  (center_y >= 0) & (center_y < height))
        flat_indices = center_y[inside] * width + center_x[inside]
        flat_accumulator += np.bincount(flat_indices, minlength=flat_accumulator.size)
    return flat_accumulator.reshape(height, width), pixels_per_vote, circle_pixels


# Lists the distinct pixel offsets of a circle, sampled at angle_count angles.
def _circle_offsets(radius, angle_count):
    # radius: circle radius (pixels)
    # angle_count: how many angles around the circle to sample
    thetas = np.linspace(0, 2 * np.pi, angle_count, endpoint=False)
    offsets = np.stack([np.round(radius * np.cos(thetas)),
                        np.round(radius * np.sin(thetas))], axis=1).astype(int)
    return np.unique(offsets, axis=0)  # drop repeated pixels so nobody votes twice


# Keeps only the strongest circle among neighbours with similar centre and radius.
def _merge_circles(circles_with_votes, center_tolerance=10, radius_tolerance=5):
    # circles_with_votes: list of (x_center, y_center, radius, votes)
    # center_tolerance: centres closer than this (pixels) count as the same circle
    # radius_tolerance: radii closer than this count as the same circle (5% of the radius at least)
    kept = []
    for x, y, radius, votes in sorted(circles_with_votes, key=lambda item: -item[3]):
        is_duplicate = any(
            abs(x - kx) <= center_tolerance and abs(y - ky) <= center_tolerance
            and abs(radius - kr) <= max(radius_tolerance, 0.05 * kr)
            for kx, ky, kr, _ in kept)
        if not is_duplicate:
            kept.append((x, y, radius, votes))
    return [(x, y, radius) for x, y, radius, _ in kept]


# Widens edges by one pixel so a circle outline still counts when it is off by a pixel or two.
def _thicken_edges(edge_img):
    # edge_img: binary edge image from Canny
    # Returns a boolean image, True where an edge pixel is within 1 pixel
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate((edge_img > 0).astype(np.uint8), kernel) > 0


# Counts how many pixels of a candidate circle's outline lie on an edge.
def _outline_pixels(thick_edges, center_x, center_y, radius, circle_pixels):
    # thick_edges: boolean image from _thicken_edges
    # center_x, center_y, radius: the candidate circle
    # circle_pixels: number of pixels in a full circle of this radius (the maximum possible)
    # Returns the number of outline pixels that are on an edge (outline parts outside the image count as missing)
    height, width = thick_edges.shape
    full_angles = max(8, int(round(2 * np.pi * radius)))
    offsets = _circle_offsets(radius, full_angles)
    outline_x = center_x + offsets[:, 0]
    outline_y = center_y + offsets[:, 1]
    inside = (outline_x >= 0) & (outline_x < width) & (outline_y >= 0) & (outline_y < height)
    return int(thick_edges[outline_y[inside], outline_x[inside]].sum())


# Detects circles by looping over radii, keeping the accumulator 2D.
def hough_circles(edge_img: np.ndarray, r_min: int, r_max: int, threshold: int) -> list[tuple]:
    # edge_img: binary edge image from Canny
    # r_min: smallest radius to search (pixels)
    # r_max: largest radius to search (pixels); wider range = slower
    # threshold: minimum edge pixels on the circle to accept it; lower = more circles
    # (a circle also needs at least MIN_CIRCLE_COVERAGE of its outline present)
    # Returns list of (x_center, y_center, radius)
    height, width = edge_img.shape[:2]
    edge_points = np.argwhere(edge_img > 0)  # each row is (y, x)
    if len(edge_points) == 0:
        return []

    found = []
    thick_edges = _thicken_edges(edge_img)
    for radius in range(r_min, r_max + 1):
        accumulator, pixels_per_vote, circle_pixels = _circle_accumulator(
            edge_points, radius, height, width)
        required_pixels = max(threshold, MIN_CIRCLE_COVERAGE * circle_pixels)
        # Cheap screening with a lower bar, then an exact check of each candidate's outline
        screening_votes = SCREENING_FRACTION * required_pixels / pixels_per_vote
        for peak_y, peak_x in _find_peaks(accumulator, screening_votes):
            pixels = _outline_pixels(thick_edges, peak_x, peak_y, radius, circle_pixels)
            if pixels >= required_pixels:
                found.append((peak_x, peak_y, radius, pixels))

    return _merge_circles(found)


# --- 3. Ellipses -----------------------------------------

# Removes near-duplicate ellipses; the list must be sorted best-first.
def _deduplicate_ellipses(ellipses, merge_distance=20):
    # ellipses: list of (x_center, y_center, a, b, angle), strongest first
    # merge_distance: centres and semi-major axes closer than this are the same ellipse
    unique = []
    for candidate in ellipses:
        is_duplicate = any(
            np.hypot(candidate[0] - existing[0], candidate[1] - existing[1]) < merge_distance
            and abs(candidate[2] - existing[2]) < merge_distance
            for existing in unique)
        if not is_duplicate:
            unique.append(candidate)
    return unique


# Scores every pair (anchor point, later point) at once; returns accepted ellipses for this anchor.
def _ellipses_for_anchor(anchor_index, x_all, y_all, threshold, min_major, max_major):
    # anchor_index: index of P1; x_all, y_all: coordinates of all sampled edge points
    # threshold: minimum votes for the semi-minor axis
    # min_major, max_major: allowed distance between P1 and P2 (pixels)
    # Returns list of (x_center, y_center, a, b, angle, votes)
    x1, y1 = x_all[anchor_index], y_all[anchor_index]
    x2, y2 = x_all[anchor_index + 1:], y_all[anchor_index + 1:]
    pair_distance = np.hypot(x2 - x1, y2 - y1)
    keep = (pair_distance >= min_major) & (pair_distance <= max_major)
    if not keep.any():
        return []
    x2, y2, pair_distance = x2[keep], y2[keep], pair_distance[keep]

    # Midpoint = centre, half distance = semi-major a, segment direction = orientation
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    semi_major = pair_distance / 2
    cos_a, sin_a = (x2 - x1) / pair_distance, (y2 - y1) / pair_distance

    # Every other point P3 votes for b (rows = pairs, columns = P3 points)
    dx = x_all[None, :] - center_x[:, None]
    dy = y_all[None, :] - center_y[:, None]
    d_sq = dx * dx + dy * dy
    a_sq = (semi_major ** 2)[:, None]
    proj_sq = (dx * cos_a[:, None] + dy * sin_a[:, None]) ** 2  # (d * cos tau)^2
    denominator = a_sq - proj_sq
    with np.errstate(divide="ignore", invalid="ignore"):
        b_sq = a_sq * (d_sq - proj_sq) / denominator
    valid = ((d_sq >= 1) & (d_sq < 0.999 * a_sq) & (denominator > 0)
             & (b_sq > 0) & (b_sq <= a_sq))

    # One 1D accumulator per pair, filled together with a single bincount
    b_bins = int(np.ceil(max_major / 2)) + 2
    b_values = np.rint(np.sqrt(np.where(valid, b_sq, 0))).astype(np.int64)
    pair_rows = np.broadcast_to(np.arange(len(semi_major))[:, None], b_values.shape)
    flat_indices = pair_rows[valid] * b_bins + b_values[valid]
    votes = np.bincount(flat_indices, minlength=len(semi_major) * b_bins).reshape(-1, b_bins)
    votes[:, 0] = 0  # b = 0 is not an ellipse
    best_b, best_votes = votes.argmax(axis=1), votes.max(axis=1)

    angles = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return [(float(center_x[r]), float(center_y[r]), float(semi_major[r]),
             float(best_b[r]), float(angles[r]), int(best_votes[r]))
            for r in np.nonzero(best_votes >= threshold)[0]]


# Detects ellipses using the Xie & Ji (2002) method.
def hough_ellipses(edge_img: np.ndarray, threshold: int) -> list[tuple]:
    # edge_img: binary edge image from Canny
    # threshold: minimum votes for the minor axis; lower = more ellipses
    # Returns list of (x_center, y_center, a, b, angle)
    edge_points = np.argwhere(edge_img > 0)  # each row is (y, x)
    if len(edge_points) < 6:
        return []

    height, width = edge_img.shape[:2]
    # The major axis (distance between P1 and P2) can be as long as the image side
    max_major_distance = max(height, width)
    min_major_distance = 10

    # Subsample edge points for speed; fixed seed so results are repeatable
    max_points = 300
    if len(edge_points) > max_points:
        rng = np.random.default_rng(0)
        edge_points = edge_points[rng.choice(len(edge_points), max_points, replace=False)]
    x_all = edge_points[:, 1].astype(float)
    y_all = edge_points[:, 0].astype(float)

    candidates = []
    for anchor_index in range(len(edge_points)):
        candidates += _ellipses_for_anchor(anchor_index, x_all, y_all, threshold,
                                           min_major_distance, max_major_distance)

    # Strongest first, then merge near-duplicates (keeps the best of each group)
    candidates.sort(key=lambda item: -item[5])
    return _deduplicate_ellipses([item[:5] for item in candidates])


# --- 4. Drawing -----------------------------------------------------------

# Draws all detected shapes on a copy of the image and returns it.
def draw_shapes(img: np.ndarray, lines: list, circles: list, ellipses: list) -> np.ndarray:
    # img: original RGB image (not modified)
    # lines: list of (rho, theta)
    # circles: list of (x_center, y_center, radius)
    # ellipses: list of (x_center, y_center, a, b, angle)
    canvas = img.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, round(max(height, width) / 400))

    # Lines in red — extend far beyond the image so they span the full frame
    line_length = height + width
    for rho, theta in lines:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x0 = cos_t * rho
        y0 = sin_t * rho
        point1_x = int(x0 + line_length * (-sin_t))
        point1_y = int(y0 + line_length * cos_t)
        point2_x = int(x0 - line_length * (-sin_t))
        point2_y = int(y0 - line_length * cos_t)
        cv2.line(canvas, (point1_x, point1_y), (point2_x, point2_y),
                 (255, 0, 0), thickness, cv2.LINE_AA)

    # Circles in green
    for center_x, center_y, radius in circles:
        cv2.circle(canvas, (int(center_x), int(center_y)), int(radius),
                   (0, 255, 0), thickness, cv2.LINE_AA)

    # Ellipses in blue
    for center_x, center_y, semi_major, semi_minor, angle_degrees in ellipses:
        cv2.ellipse(
            canvas,
            (int(center_x), int(center_y)),
            (int(semi_major), int(semi_minor)),
            angle_degrees, 0, 360,
            (0, 0, 255), thickness, cv2.LINE_AA
        )

    return canvas
