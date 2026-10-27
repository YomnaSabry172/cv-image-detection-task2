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
    raise NotImplementedError("TODO (B): Gaussian blur")


# Step 2: compute gradient magnitude and direction (e.g. with Sobel kernels).
def gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # img: blurred grayscale image
    # Returns: (magnitude, direction) — direction in radians
    raise NotImplementedError("TODO (B): gradient magnitude and direction")


# Step 3: thin edges to 1 pixel wide by keeping only local maxima along the gradient.
def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    # magnitude: gradient strength per pixel
    # direction: gradient angle per pixel (radians)
    raise NotImplementedError("TODO (B): non-maximum suppression")


# Step 4: classify each pixel as strong, weak, or suppressed.
def double_threshold(img: np.ndarray, low: float, high: float) -> np.ndarray:
    # img: thinned magnitude image from NMS
    # low: below this a pixel is suppressed; lower = more weak edges kept
    # high: above this a pixel is a strong edge; lower = more strong edges
    raise NotImplementedError("TODO (B): double threshold")


# Step 5: keep weak edges only if they connect to a strong edge.
def hysteresis(img: np.ndarray) -> np.ndarray:
    # img: output of double_threshold (strong / weak / zero pixels)
    raise NotImplementedError("TODO (B): hysteresis edge tracking")


# Runs all 5 Canny steps in order and returns the final binary edge image.
def run_canny(img: np.ndarray, kernel_size: int, sigma: float, low: float, high: float) -> np.ndarray:
    # img: input image (RGB or grayscale)
    # kernel_size, sigma: passed to gaussian_blur
    # low, high: passed to double_threshold
    # This is the only function the UI calls.
    raise NotImplementedError("TODO (B): run all Canny steps")
