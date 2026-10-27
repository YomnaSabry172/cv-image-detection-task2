"""Image loading, saving and conversion for display (owner: Lead)."""
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap


# Reads an image file from disk and returns it as an RGB numpy array.
def load_image(path: str) -> np.ndarray:
    # path: file path to read (np.fromfile handles non-ASCII Windows paths)
    file_bytes = np.fromfile(path, dtype=np.uint8)
    bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


# Saves an RGB or grayscale numpy array to disk.
def save_image(img: np.ndarray, path: str) -> None:
    # img: image to save (RGB, grayscale, bool or float)
    # path: destination; the extension (.png, .jpg) picks the format
    image_uint8 = _to_uint8(img)
    if image_uint8.ndim == 3:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    extension = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    success, encoded = cv2.imencode(extension, image_uint8)
    if not success:
        raise ValueError(f"Could not save image: {path}")
    encoded.tofile(path)


# Converts a numpy image (RGB or grayscale) to a QPixmap for the PyQt5 window.
def to_display(img: np.ndarray) -> QPixmap:
    # img: image to show (RGB, grayscale, bool or float)
    image_uint8 = np.ascontiguousarray(_to_uint8(img))
    height, width = image_uint8.shape[:2]
    if image_uint8.ndim == 2:
        qt_image = QImage(image_uint8.data, width, height, width, QImage.Format_Grayscale8)
    else:
        qt_image = QImage(image_uint8.data, width, height, 3 * width, QImage.Format_RGB888)
    # copy() so the QImage owns its memory after the numpy array is freed
    return QPixmap.fromImage(qt_image.copy())


# Scales any image (bool, float, int) into the 0–255 uint8 range.
def _to_uint8(img: np.ndarray) -> np.ndarray:
    # img: image of any dtype
    if img.dtype == np.uint8:
        return img
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    image_float = img.astype(np.float64)
    max_value = image_float.max() if image_float.size else 0
    if max_value <= 1.0:
        image_float = image_float * 255.0
    elif max_value > 255.0:
        image_float = image_float / max_value * 255.0
    return np.clip(image_float, 0, 255).astype(np.uint8)
