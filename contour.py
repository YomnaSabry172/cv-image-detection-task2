"""Active contour (greedy snake), chain code, perimeter, area (owner: D)."""
import numpy as np


# Creates a circle of n_points evenly spaced around the center.
def initialize_contour(center: tuple, radius: int, n_points: int) -> np.ndarray:
    # center: (x, y) pixel the user clicked
    # radius: radius of the starting circle (pixels)
    # n_points: number of snake points; more = finer contour, slower
    # Returns array of shape (n_points, 2) — each row is (x, y)
    raise NotImplementedError("TODO (D): initialize contour")


# Computes total energy of the contour (image energy + internal energy).
def compute_energy(contour: np.ndarray, img: np.ndarray, alpha: float, beta: float) -> float:
    # contour: (n_points, 2) array of (x, y)
    # img: grayscale image (or its gradient)
    # alpha: elasticity weight; higher = points stay evenly spaced, contour shrinks
    # beta: smoothness weight; higher = fewer sharp corners
    raise NotImplementedError("TODO (D): compute energy")


# Runs the greedy algorithm: moves each point to minimize energy.
def evolve_contour(contour: np.ndarray, img: np.ndarray, alpha: float, beta: float,
                   n_iterations: int, window_size: int) -> np.ndarray:
    # contour: initial (n_points, 2) array
    # img: RGB image as loaded by the UI (convert to grayscale inside)
    # alpha, beta: see compute_energy
    # n_iterations: maximum number of greedy passes
    # window_size: side of the search neighbourhood per point (odd); bigger = moves farther per step
    # Returns the final evolved contour
    raise NotImplementedError("TODO (D): evolve contour (greedy)")


# Converts contour points to 8-direction chain code (0–7).
def to_chain_code(contour: np.ndarray) -> list[int]:
    # contour: (n_points, 2) array of (x, y); points may be several pixels apart,
    #          so step between neighbours one pixel at a time
    raise NotImplementedError("TODO (D): chain code")


# Computes perimeter from chain code (even codes = 1, odd codes = sqrt(2)).
def compute_perimeter(chain_code: list[int]) -> float:
    # chain_code: list of directions 0–7
    raise NotImplementedError("TODO (D): perimeter")


# Computes area enclosed by the contour using the shoelace formula.
def compute_area(contour: np.ndarray) -> float:
    # contour: (n_points, 2) array of (x, y)
    raise NotImplementedError("TODO (D): area")
