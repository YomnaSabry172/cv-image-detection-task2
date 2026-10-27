"""Hough transform shape detection, implemented from scratch (owner: C).

Lines, circles (2D accumulator per radius), ellipses (Xie & Ji 2002).
draw_shapes is the only drawing function in the project.
"""
import numpy as np


# Detects straight lines and returns them as (rho, theta) pairs.
def hough_lines(edge_img: np.ndarray, threshold: int) -> list[tuple]:
    # edge_img: binary edge image from Canny
    # threshold: minimum accumulator votes to accept a line; lower = more lines
    raise NotImplementedError("TODO (C): Hough lines")


# Detects circles by looping over radii, keeping the accumulator 2D.
def hough_circles(edge_img: np.ndarray, r_min: int, r_max: int, threshold: int) -> list[tuple]:
    # edge_img: binary edge image from Canny
    # r_min: smallest radius to search (pixels)
    # r_max: largest radius to search (pixels); wider range = slower
    # threshold: minimum votes to accept a circle; lower = more circles
    # Returns list of (x_center, y_center, radius)
    raise NotImplementedError("TODO (C): Hough circles")


# Detects ellipses using the Xie & Ji (2002) method.
def hough_ellipses(edge_img: np.ndarray, threshold: int) -> list[tuple]:
    # edge_img: binary edge image from Canny
    # threshold: minimum votes for the minor axis; lower = more ellipses
    # Returns list of (x_center, y_center, a, b, angle)
    raise NotImplementedError("TODO (C): Hough ellipses (Xie & Ji)")


# Draws all detected shapes on a copy of the image and returns it.
def draw_shapes(img: np.ndarray, lines: list, circles: list, ellipses: list) -> np.ndarray:
    # img: original RGB image (not modified)
    # lines: list of (rho, theta)
    # circles: list of (x_center, y_center, radius)
    # ellipses: list of (x_center, y_center, a, b, angle)
    raise NotImplementedError("TODO (C): draw detected shapes")
