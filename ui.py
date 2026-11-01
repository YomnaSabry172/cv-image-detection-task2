"""All PyQt5 UI: theme, main window, two tabs, controls and image panels (owner: Lead).

The UI never contains algorithm logic. It reads controls, calls the functions
in canny.py / hough.py / contour.py, and displays what they return.
"""
import os
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QColor, QFont, QPainter, QPalette, QPen, QRegExpValidator
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QTabWidget, QVBoxLayout, QWidget,
)

import canny
import contour
import hough
from image_io import load_image, save_image, to_display

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"

# Colors used when drawing the snake on the image (RGB, because images are RGB)
CONTOUR_LINE_RGB = (45, 212, 191)
CONTOUR_POINT_RGB = (251, 191, 36)
CENTER_MARKER_RGB = (244, 114, 182)

STYLESHEET = """
QWidget#root { background: #0f1420; }
QLabel#appTitle { font-size: 18pt; font-weight: 600; color: #f1f4f9; }
QLabel#appSubtitle { color: #8b93a7; }
QLabel#accentDot { background: #2dd4bf; border-radius: 5px; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: #171d2b; color: #8b93a7; padding: 9px 22px; margin-right: 8px; min-width: 150px;
               border: 1px solid #252e41; border-radius: 9px; }
QTabBar::tab:selected { background: #2dd4bf; color: #04211d; border-color: #2dd4bf; font-weight: 600; }
QTabBar::tab:hover:!selected { color: #e6e9ef; border-color: #3a4560; }
QFrame#card { background: #171d2b; border: 1px solid #252e41; border-radius: 14px; }
QLabel#cardTitle { font-size: 11pt; font-weight: 600; color: #f1f4f9; }
QLabel#cardHint, QLabel#caption { color: #8b93a7; font-size: 9pt; }
QLabel#paramName { color: #c3c9d6; }
QLineEdit#paramValue { background: #0f1420; color: #2dd4bf; border: 1px solid #252e41;
                       border-radius: 6px; padding: 1px 6px; font-family: Consolas;
                       selection-background-color: #2dd4bf; selection-color: #04211d; }
QLineEdit#paramValue:hover { border-color: #3a4560; }
QLineEdit#paramValue:focus { border-color: #2dd4bf; background: #0b0f19; }
QPushButton { background: #1f2737; color: #e6e9ef; border: 1px solid #2f3a50;
              border-radius: 9px; padding: 8px 14px; }
QPushButton:hover { background: #26304a; border-color: #3d4a66; }
QPushButton:pressed { background: #1a2131; }
QPushButton:checked { background: #134e4a; border-color: #2dd4bf; color: #5eead4; }
QPushButton#primary { background: #2dd4bf; color: #04211d; border: none; font-weight: 600; }
QPushButton#primary:hover { background: #5eead4; }
QPushButton#primary:pressed { background: #14b8a6; }
QSlider::groove:horizontal { height: 4px; background: #2a3346; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #2dd4bf; border-radius: 2px; }
QSlider::handle:horizontal { background: #e6e9ef; width: 14px; height: 14px;
                             margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #5eead4; }
QCheckBox { spacing: 8px; color: #e6e9ef; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 5px;
                       border: 1px solid #3a4560; background: #0f1420; }
QCheckBox::indicator:checked { background: #2dd4bf; border: 3px solid #0f1420;
                               outline: none; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: #2a3346; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QPlainTextEdit { background: #0b0f19; color: #5eead4; border: 1px solid #252e41;
                 border-radius: 9px; padding: 6px; font-family: Consolas; font-size: 10pt; }
QFrame#stat { background: #0f1420; border: 1px solid #252e41; border-radius: 11px; }
QLabel#statValue { font-size: 17pt; font-weight: 600; color: #2dd4bf; }
QStatusBar { background: #0b0f19; color: #8b93a7; border-top: 1px solid #1c2333; }
QToolTip { background: #1f2737; color: #e6e9ef; border: 1px solid #2dd4bf; padding: 6px; }
QMessageBox { background: #171d2b; }
QMessageBox QLabel { color: #e6e9ef; }
"""


# Applies the dark theme (Fusion style + palette + stylesheet) to the whole app.
def apply_theme(app: QApplication) -> None:
    # app: the running QApplication
    app.setStyle("Fusion")
    palette = QPalette()
    colors = {
        QPalette.Window: "#0f1420", QPalette.WindowText: "#e6e9ef",
        QPalette.Base: "#0b0f19", QPalette.AlternateBase: "#171d2b",
        QPalette.Text: "#e6e9ef", QPalette.Button: "#1f2737",
        QPalette.ButtonText: "#e6e9ef", QPalette.Highlight: "#2dd4bf",
        QPalette.HighlightedText: "#04211d", QPalette.ToolTipBase: "#1f2737",
        QPalette.ToolTipText: "#e6e9ef",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)


# Calls an algorithm function and shows a friendly dialog if it is missing or crashes.
def run_algorithm(parent: QWidget, function, *args) -> tuple[bool, object]:
    # parent: widget that owns the dialog
    # function: algorithm to call, e.g. canny.run_canny
    # args: arguments passed straight to the function
    # Returns (succeeded, result)
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        return True, function(*args)
    except NotImplementedError as error:
        title, text = "Not implemented yet", f"{function.__module__}.{function.__name__}()\n\n{error}"
    except Exception as error:
        traceback.print_exc()
        title, text = "Algorithm error", f"{function.__module__}.{function.__name__}() failed:\n\n{type(error).__name__}: {error}"
    finally:
        QApplication.restoreOverrideCursor()
    QMessageBox.information(parent, title, text)
    return False, None


# Opens a file dialog and returns the chosen image path ("" if cancelled).
def ask_open_path(parent: QWidget) -> str:
    # parent: widget that owns the dialog
    path, _ = QFileDialog.getOpenFileName(parent, "Upload image", "", IMAGE_FILTER)
    return path


# Opens a save dialog, writes the image, and reports the result in the status bar.
def save_with_dialog(parent: QWidget, img: np.ndarray, status_signal) -> None:
    # parent: widget that owns the dialog
    # img: image to save (None means nothing to save yet)
    # status_signal: signal used to print the outcome in the status bar
    if img is None:
        QMessageBox.information(parent, "Nothing to save", "Run an algorithm first, then save its result.")
        return
    path, _ = QFileDialog.getSaveFileName(parent, "Save result", "result.png", "PNG (*.png);;JPEG (*.jpg)")
    if not path:
        return
    save_image(img, path)
    status_signal.emit(f"Saved {os.path.basename(path)}")


# Builds a rounded card frame with a bold title and an optional grey hint line.
def make_card(title: str, hint: str = "") -> tuple[QFrame, QVBoxLayout]:
    # title: heading at the top of the card
    # hint: smaller grey explanation under the title
    card = QFrame(objectName="card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(10)
    layout.addWidget(QLabel(title, objectName="cardTitle"))
    if hint:
        hint_label = QLabel(hint, objectName="cardHint")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)
    return card, layout


# Builds a push button; primary buttons are filled with the accent color.
def make_button(text: str, on_click, primary: bool = False) -> QPushButton:
    # text: button label
    # on_click: function called when the button is clicked
    # primary: True for the main action of a card
    button = QPushButton(text)
    if primary:
        button.setObjectName("primary")
    button.setCursor(Qt.PointingHandCursor)
    button.clicked.connect(lambda: on_click())
    return button


# Stacks control cards in a fixed-width, scrollable left column.
def make_sidebar(cards: list[QFrame]) -> QScrollArea:
    # cards: card frames shown top to bottom
    column = QWidget()
    layout = QVBoxLayout(column)
    layout.setContentsMargins(0, 0, 6, 0)
    layout.setSpacing(14)
    for card in cards:
        layout.addWidget(card)
    layout.addStretch()
    scroll = QScrollArea()
    scroll.setWidget(column)
    scroll.setWidgetResizable(True)
    scroll.setFixedWidth(340)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return scroll


# Wraps an image canvas in a card with a title and a caption on the right.
def make_canvas_card(title: str, canvas: QWidget) -> tuple[QFrame, QLabel]:
    # title: heading above the image
    # canvas: the ImageCanvas to show
    card = QFrame(objectName="card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(14, 12, 14, 14)
    header = QHBoxLayout()
    header.addWidget(QLabel(title, objectName="cardTitle"))
    header.addStretch()
    caption = QLabel("", objectName="caption")
    header.addWidget(caption)
    layout.addLayout(header)
    layout.addWidget(canvas, 1)
    return card, caption


# Builds a small tile showing one big number with a caption under it.
def make_stat_tile(caption: str) -> tuple[QFrame, QLabel]:
    # caption: what the number means, e.g. "Perimeter (px)"
    tile = QFrame(objectName="stat")
    layout = QVBoxLayout(tile)
    layout.setContentsMargins(14, 10, 14, 10)
    layout.setSpacing(2)
    value_label = QLabel("—", objectName="statValue")
    layout.addWidget(value_label)
    layout.addWidget(QLabel(caption, objectName="caption"))
    return tile, value_label


# Draws the snake (closed line + points) and its center marker on a copy of the image.
def draw_contour(img: np.ndarray, contour_points, center) -> np.ndarray:
    # img: RGB image (not modified)
    # contour_points: (n_points, 2) array of (x, y), or None
    # center: (x, y) the user clicked, or None
    canvas = img.copy()
    thickness = max(1, round(max(img.shape[:2]) / 400))
    if contour_points is not None and len(contour_points) > 1:
        polygon = np.round(contour_points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, CONTOUR_LINE_RGB, thickness, cv2.LINE_AA)
        for x, y in polygon[:, 0]:
            cv2.circle(canvas, (int(x), int(y)), thickness + 1, CONTOUR_POINT_RGB, -1, cv2.LINE_AA)
    if center is not None:
        marker_size = 8 * thickness + 6
        cv2.drawMarker(canvas, (int(center[0]), int(center[1])), CENTER_MARKER_RGB,
                       cv2.MARKER_CROSS, marker_size, thickness, cv2.LINE_AA)
    return canvas


class ImageCanvas(QWidget):
    """Shows a numpy image scaled to fit, reports clicks in image pixels, accepts dropped files."""

    clicked = pyqtSignal(int, int)  # (x, y) in image pixel coordinates
    file_dropped = pyqtSignal(str)  # path of an image file dropped on the canvas

    # Creates an empty canvas that shows placeholder text until an image is set.
    def __init__(self, placeholder: str):
        # placeholder: text shown while no image is loaded
        super().__init__()
        self.placeholder = placeholder
        self.pixmap = None
        self.setMinimumSize(320, 260)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # Replaces the shown image; None clears the canvas.
    def set_image(self, img) -> None:
        # img: numpy image or None
        self.pixmap = to_display(img) if img is not None else None
        self.update()

    # Returns where the image is drawn inside the widget, keeping its aspect ratio.
    def _image_rect(self) -> QRectF:
        area = QRectF(self.rect()).adjusted(10, 10, -10, -10)
        scale = min(area.width() / self.pixmap.width(), area.height() / self.pixmap.height())
        width, height = self.pixmap.width() * scale, self.pixmap.height() * scale
        left = area.x() + (area.width() - width) / 2
        top = area.y() + (area.height() - height) / 2
        return QRectF(left, top, width, height)

    # Paints the dark background, then the image or the placeholder text.
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        border_style = Qt.SolidLine if self.pixmap else Qt.DashLine
        painter.setPen(QPen(QColor("#2a3346"), 1.2, border_style))
        painter.setBrush(QColor("#0b0f19"))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 10, 10)
        if self.pixmap is None:
            painter.setPen(QColor("#5b6378"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        painter.drawPixmap(self._image_rect(), self.pixmap, QRectF(self.pixmap.rect()))

    # Converts a left click on the widget into image pixel coordinates.
    def mousePressEvent(self, event):
        if self.pixmap is None or event.button() != Qt.LeftButton:
            return
        image_rect = self._image_rect()
        if not image_rect.contains(event.localPos()):
            return
        scale = image_rect.width() / self.pixmap.width()
        x = int((event.localPos().x() - image_rect.x()) / scale)
        y = int((event.localPos().y() - image_rect.y()) / scale)
        self.clicked.emit(x, y)

    # Accepts drags that carry files.
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    # Emits the path of the first dropped file.
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())


class LabeledSlider(QWidget):
    """A slider with its name on the left and an editable value box on the right."""

    value_changed = pyqtSignal(float)

    # Creates the slider; step can be fractional (0.1) or 2 with an odd minimum for odd-only values.
    def __init__(self, name: str, minimum: float, maximum: float, default: float,
                 step: float = 1, tooltip: str = ""):
        # name: label shown above the slider
        # minimum, maximum: allowed value range
        # default: starting value
        # step: distance between two slider positions
        # tooltip: plain-English explanation shown on hover
        super().__init__()
        self.minimum, self.maximum, self.step = minimum, maximum, step
        self.is_integer = float(step).is_integer() and float(minimum).is_integer()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, round((maximum - minimum) / step))
        self.slider.setValue(round((default - minimum) / step))
        self.slider.setCursor(Qt.PointingHandCursor)
        self.value_input = self._build_value_input()
        header = QHBoxLayout()
        header.addWidget(QLabel(name, objectName="paramName"))
        header.addStretch()
        header.addWidget(self.value_input)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.slider)
        self.setToolTip(tooltip)
        self.slider.valueChanged.connect(self._on_slider_moved)
        self._on_slider_moved()

    # Builds the small value box; a validator blocks letters while typing.
    def _build_value_input(self) -> QLineEdit:
        value_input = QLineEdit(objectName="paramValue")
        value_input.setFixedWidth(64)
        value_input.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # only digits (and one "." for decimal sliders); range is handled in _on_text_entered
        allowed_pattern = r"\d*" if self.is_integer else r"\d*\.?\d*"
        value_input.setValidator(QRegExpValidator(QRegExp(allowed_pattern)))
        value_input.editingFinished.connect(self._on_text_entered)
        return value_input

    # Applies a typed value: clamps to the range, snaps to the step, moves the slider.
    def _on_text_entered(self) -> None:
        try:
            typed_value = float(self.value_input.text())
        except ValueError:
            self._show_value()  # empty or half-typed: restore the current value
            return
        self.set_value(min(max(typed_value, self.minimum), self.maximum))
        self._show_value()  # slider may not move (same step), so refresh the box anyway

    # Writes the current value into the box.
    def _show_value(self) -> None:
        current = self.value()
        self.value_input.setText(str(current) if self.is_integer else f"{current:.2f}")

    # Returns the current value (int for integer sliders, float otherwise).
    def value(self):
        raw_value = self.minimum + self.slider.value() * self.step
        return int(round(raw_value)) if self.is_integer else round(raw_value, 4)

    # Moves the slider to a new value.
    def set_value(self, new_value: float) -> None:
        # new_value: value to show (snapped to the nearest step)
        self.slider.setValue(round((new_value - self.minimum) / self.step))

    # Updates the value box and re-emits the new value.
    def _on_slider_moved(self, *_):
        self._show_value()
        self.value_changed.emit(float(self.value()))


# Keeps two sliders ordered: lower slider never above upper slider.
def link_min_max(lower: LabeledSlider, upper: LabeledSlider) -> None:
    # lower: slider that must stay <= upper (e.g. low threshold, r_min)
    # upper: slider that must stay >= lower (e.g. high threshold, r_max)
    lower.value_changed.connect(lambda v: upper.set_value(v) if v > upper.value() else None)
    upper.value_changed.connect(lambda v: lower.set_value(v) if v < lower.value() else None)


class ShapeDetectionTab(QWidget):
    """Tab 1: Canny edges, then Hough lines / circles / ellipses on those edges."""

    status_message = pyqtSignal(str)

    # Builds the controls sidebar and the original / result image panels.
    def __init__(self):
        super().__init__()
        self.original_image = None  # RGB image the user uploaded
        self.edge_image = None      # latest Canny output, used as Hough input
        self.result_image = None    # image currently shown on the right (what Save writes)
        self.original_canvas = ImageCanvas("Click “Upload image” or drop a file here")
        self.result_canvas = ImageCanvas("Results appear here")
        self.original_canvas.file_dropped.connect(self.upload_image)
        original_card, self.original_caption = make_canvas_card("Original", self.original_canvas)
        result_card, self.result_caption = make_canvas_card("Result", self.result_canvas)
        sidebar = make_sidebar([self._build_image_card(), self._build_canny_card(), self._build_hough_card()])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(sidebar)
        layout.addWidget(original_card, 1)
        layout.addWidget(result_card, 1)

    # Card 1: upload and save buttons.
    def _build_image_card(self) -> QFrame:
        card, layout = make_card("Image", "PNG, JPG, BMP or TIFF. You can also drag a file onto the left panel.")
        layout.addWidget(make_button("Upload image", self.upload_image))
        layout.addWidget(make_button("Save result", self.save_result))
        return card

    # Card 2: Canny sliders and the Run Canny button.
    def _build_canny_card(self) -> QFrame:
        card, layout = make_card("Canny edge detection", "Run this first — Hough uses these edges.")
        self.kernel_slider = LabeledSlider("Kernel size", 3, 15, 5, step=2,
                                           tooltip="Gaussian kernel width (odd). Bigger = more blur, fewer noisy edges.")
        self.sigma_slider = LabeledSlider("Sigma", 0.1, 5.0, 1.4, step=0.1,
                                          tooltip="Gaussian spread. Bigger = smoother image, only strong edges survive.")
        self.low_slider = LabeledSlider("Low threshold (%)", 0, 100, 25,
                                        tooltip="Percentage of the strongest gradient. Lower = more weak edges kept.")
        self.high_slider = LabeledSlider("High threshold (%)", 0, 100, 75,
                                         tooltip="Percentage of the strongest gradient. Lower = more strong edges.")
        link_min_max(self.low_slider, self.high_slider)
        for slider in (self.kernel_slider, self.sigma_slider, self.low_slider, self.high_slider):
            layout.addWidget(slider)
        layout.addWidget(make_button("Run Canny", self.run_canny, primary=True))
        return card

    # Card 3: shape checkboxes, Hough sliders and the Run Hough button.
    def _build_hough_card(self) -> QFrame:
        card, layout = make_card("Hough transform", "Pick one or more shapes to detect.")
        self.lines_check = QCheckBox("Lines")
        self.circles_check = QCheckBox("Circles")
        self.ellipses_check = QCheckBox("Ellipses")
        self.lines_check.setChecked(True)
        checkbox_row = QHBoxLayout()
        for checkbox in (self.lines_check, self.circles_check, self.ellipses_check):
            checkbox.setCursor(Qt.PointingHandCursor)
            checkbox_row.addWidget(checkbox)
        layout.addLayout(checkbox_row)
        self.lines_threshold_slider = LabeledSlider(
            "Lines threshold (votes)", 1, 500, 100,
            tooltip="Minimum edge pixels on a line. Lower = more (and more false) lines.")
        self.circles_threshold_slider = LabeledSlider(
            "Circles threshold (votes)", 1, 500, 60,
            tooltip="Minimum edge pixels on a circle. At least 40% of the outline must also be present. Lower = more circles.")
        self.r_min_slider = LabeledSlider("Min radius", 5, 300, 10,
                                          tooltip="Smallest circle radius searched, in pixels.")
        self.r_max_slider = LabeledSlider("Max radius", 5, 300, 60,
                                          tooltip="Largest circle radius searched. Wider range = slower.")
        link_min_max(self.r_min_slider, self.r_max_slider)
        self.ellipses_threshold_slider = LabeledSlider(
            "Ellipses threshold (votes)", 1, 300, 50,
            tooltip="Minimum sampled edge points that must fit the ellipse (max 300). Lower = more (and more false) ellipses.")
        # Each shape shows its own settings only while its checkbox is ticked
        option_groups = (
            (self.lines_check, (self.lines_threshold_slider,)),
            (self.circles_check, (self.circles_threshold_slider, self.r_min_slider, self.r_max_slider)),
            (self.ellipses_check, (self.ellipses_threshold_slider,)),
        )
        for checkbox, sliders in option_groups:
            layout.addWidget(self._make_option_group(checkbox, sliders))
        layout.addWidget(make_button("Run Hough", self.run_hough, primary=True))
        return card

    # Wraps sliders in a widget that is only visible while its shape checkbox is ticked.
    def _make_option_group(self, checkbox: QCheckBox, sliders: tuple) -> QWidget:
        # checkbox: the shape checkbox that shows / hides the group
        # sliders: the controls that belong to that shape
        group = QWidget()
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(10)
        for slider in sliders:
            group_layout.addWidget(slider)
        group.setVisible(checkbox.isChecked())
        checkbox.toggled.connect(group.setVisible)
        return group

    # Loads an image from a dialog (or a dropped path) and resets old results.
    def upload_image(self, path: str = "") -> None:
        # path: file to load; empty = ask with a file dialog
        path = path or ask_open_path(self)
        if not path:
            return
        try:
            image = load_image(path)
        except ValueError as error:
            QMessageBox.warning(self, "Could not open image", str(error))
            return
        self.original_image, self.edge_image, self.result_image = image, None, None
        self.original_canvas.set_image(image)
        self.result_canvas.set_image(None)
        self.original_caption.setText(f"{image.shape[1]} × {image.shape[0]} px")
        self.result_caption.setText("")
        self.status_message.emit(f"Loaded {os.path.basename(path)}")

    # Shows a message and returns False if no image is loaded yet.
    def _require_image(self) -> bool:
        if self.original_image is None:
            QMessageBox.information(self, "No image", "Upload an image first.")
            return False
        return True

    # Reads the Canny sliders, runs canny.run_canny and shows the edges.
    def run_canny(self) -> None:
        if not self._require_image():
            return
        succeeded, edges = run_algorithm(
            self, canny.run_canny, self.original_image, self.kernel_slider.value(),
            self.sigma_slider.value(), self.low_slider.value(), self.high_slider.value())
        if not succeeded:
            return
        self.edge_image = edges
        self._show_result(edges, "Canny edges")

    # Runs the selected Hough detectors on the Canny edges and draws the shapes.
    def run_hough(self) -> None:
        if not self._require_image():
            return
        if self.edge_image is None:
            QMessageBox.information(self, "Run Canny first", "Hough works on the Canny edge image. Click “Run Canny” first.")
            return
        if not (self.lines_check.isChecked() or self.circles_check.isChecked() or self.ellipses_check.isChecked()):
            QMessageBox.information(self, "No shape selected", "Tick at least one of Lines, Circles or Ellipses.")
            return
        detections = self._detect_selected_shapes()
        if detections is None:
            return
        lines, circles, ellipses = detections
        succeeded, drawn = run_algorithm(self, hough.draw_shapes, self.original_image, lines, circles, ellipses)
        if succeeded:
            self._show_result(drawn, f"{len(lines)} lines · {len(circles)} circles · {len(ellipses)} ellipses")

    # Calls each ticked Hough detector; returns (lines, circles, ellipses) or None on failure.
    def _detect_selected_shapes(self):
        detections = {"lines": [], "circles": [], "ellipses": []}
        jobs = [
            (self.lines_check, "lines", hough.hough_lines,
             (self.edge_image, self.lines_threshold_slider.value())),
            (self.circles_check, "circles", hough.hough_circles,
             (self.edge_image, self.r_min_slider.value(), self.r_max_slider.value(),
              self.circles_threshold_slider.value())),
            (self.ellipses_check, "ellipses", hough.hough_ellipses,
             (self.edge_image, self.ellipses_threshold_slider.value())),
        ]
        for checkbox, key, function, args in jobs:
            if not checkbox.isChecked():
                continue
            succeeded, found = run_algorithm(self, function, *args)
            if not succeeded:
                return None
            detections[key] = list(found)
        return detections["lines"], detections["circles"], detections["ellipses"]

    # Puts an image on the right panel and remembers it for saving.
    def _show_result(self, img: np.ndarray, caption: str) -> None:
        # img: image to show
        # caption: short description shown above the panel
        self.result_image = img
        self.result_canvas.set_image(img)
        self.result_caption.setText(caption)
        self.status_message.emit(caption)

    # Saves the image currently shown on the right panel.
    def save_result(self) -> None:
        save_with_dialog(self, self.result_image, self.status_message)


class ActiveContourTab(QWidget):
    """Tab 2: place a circular snake, evolve it, and show chain code, perimeter and area."""

    status_message = pyqtSignal(str)

    # Builds the controls sidebar, the image panel and the results card.
    def __init__(self):
        super().__init__()
        self.image = None           # RGB image the user uploaded
        self.center = None          # (x, y) the user clicked
        self.contour_points = None  # current snake, (n_points, 2) array of (x, y)
        self.shown_image = None     # image with the snake drawn on it (what Save writes)
        self.canvas = ImageCanvas("Click “Upload image” or drop a coin image here")
        self.canvas.clicked.connect(self.on_canvas_clicked)
        self.canvas.file_dropped.connect(self.upload_image)
        canvas_card, self.canvas_caption = make_canvas_card("Image", self.canvas)
        sidebar = make_sidebar([self._build_image_card(), self._build_contour_card(), self._build_snake_card()])
        right_column = QVBoxLayout()
        right_column.setSpacing(16)
        right_column.addWidget(canvas_card, 1)
        right_column.addWidget(self._build_results_card())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(sidebar)
        layout.addLayout(right_column, 1)

    # Card 1: upload and save buttons.
    def _build_image_card(self) -> QFrame:
        card, layout = make_card("Image", "A coin image works best: one clear object on a plain background.")
        layout.addWidget(make_button("Upload image", self.upload_image))
        layout.addWidget(make_button("Save result", self.save_result))
        return card

    # Card 2: starting circle size/resolution and the Draw contour button.
    def _build_contour_card(self) -> QFrame:
        card, layout = make_card("Initial contour", "Click “Draw contour”, then click the center of the object.")
        self.n_points_slider = LabeledSlider("Number of points", 10, 300, 60,
                                             tooltip="Points on the snake. More = finer boundary, slower.")
        self.radius_slider = LabeledSlider("Radius", 5, 500, 80,
                                           tooltip="Radius of the starting circle in pixels. Should enclose the object.")
        for slider in (self.n_points_slider, self.radius_slider):
            slider.value_changed.connect(self._on_initial_shape_changed)
            layout.addWidget(slider)
        self.draw_button = QPushButton("Draw contour")
        self.draw_button.setCheckable(True)
        self.draw_button.setCursor(Qt.PointingHandCursor)
        self.draw_button.toggled.connect(self._on_draw_toggled)
        layout.addWidget(self.draw_button)
        return card

    # Card 3: snake energy weights, iterations, window size and Run snake.
    def _build_snake_card(self) -> QFrame:
        card, layout = make_card("Snake parameters", "Greedy algorithm: each point moves to the lowest-energy pixel nearby.")
        self.alpha_slider = LabeledSlider("Alpha (elasticity)", 0.0, 5.0, 1.0, step=0.05,
                                          tooltip="Keeps points evenly spaced. Higher = contour shrinks / tightens.")
        self.beta_slider = LabeledSlider("Beta (smoothness)", 0.0, 5.0, 1.0, step=0.05,
                                         tooltip="Penalises sharp bends. Higher = rounder contour.")
        self.iterations_slider = LabeledSlider("Iterations", 1, 1000, 200,
                                               tooltip="Maximum greedy passes over all points.")
        self.window_slider = LabeledSlider("Window size", 3, 15, 5, step=2,
                                           tooltip="Neighbourhood searched per point (odd). Bigger = moves farther per step.")
        for slider in (self.alpha_slider, self.beta_slider, self.iterations_slider, self.window_slider):
            layout.addWidget(slider)
        layout.addWidget(make_button("Run snake", self.run_snake, primary=True))
        return card

    # Card under the image: perimeter / area / points tiles and the chain code box.
    def _build_results_card(self) -> QFrame:
        card, layout = make_card("Results")
        tiles_row = QHBoxLayout()
        tiles_row.setSpacing(12)
        perimeter_tile, self.perimeter_value = make_stat_tile("Perimeter (px)")
        area_tile, self.area_value = make_stat_tile("Area (px²)")
        length_tile, self.chain_length_value = make_stat_tile("Chain code length")
        for tile in (perimeter_tile, area_tile, length_tile):
            tiles_row.addWidget(tile)
        layout.addLayout(tiles_row)
        chain_header = QHBoxLayout()
        chain_header.addWidget(QLabel("Chain code (0–7)", objectName="paramName"))
        chain_header.addStretch()
        chain_header.addWidget(make_button("Copy", self._copy_chain_code))
        layout.addLayout(chain_header)
        self.chain_code_box = QPlainTextEdit()
        self.chain_code_box.setReadOnly(True)
        self.chain_code_box.setFixedHeight(70)
        self.chain_code_box.setPlaceholderText("Run the snake to see the chain code")
        layout.addWidget(self.chain_code_box)
        return card

    # Loads an image from a dialog (or a dropped path) and clears the old snake.
    def upload_image(self, path: str = "") -> None:
        # path: file to load; empty = ask with a file dialog
        path = path or ask_open_path(self)
        if not path:
            return
        try:
            image = load_image(path)
        except ValueError as error:
            QMessageBox.warning(self, "Could not open image", str(error))
            return
        self.image, self.center, self.contour_points = image, None, None
        self.shown_image = image
        self.canvas.set_image(image)
        self.canvas_caption.setText(f"{image.shape[1]} × {image.shape[0]} px")
        self._clear_results()
        self.status_message.emit(f"Loaded {os.path.basename(path)} — click “Draw contour” to place the snake")

    # Turns click-to-place mode on or off when the Draw contour button is toggled.
    def _on_draw_toggled(self, checked: bool) -> None:
        # checked: True when the button is pressed in
        if checked and self.image is None:
            self.draw_button.setChecked(False)
            QMessageBox.information(self, "No image", "Upload an image first.")
            return
        if checked:
            self.canvas.setCursor(Qt.CrossCursor)
            self.status_message.emit("Click the center of the object on the image")
        else:
            self.canvas.unsetCursor()

    # Sets the snake center from a click (only while Draw contour is active).
    def on_canvas_clicked(self, x: int, y: int) -> None:
        # x, y: clicked pixel in image coordinates
        if not self.draw_button.isChecked():
            return
        self.center = (x, y)
        self.draw_button.setChecked(False)
        self._initialize_contour()

    # Rebuilds the starting circle when points/radius change, if one is already placed.
    def _on_initial_shape_changed(self, _value: float) -> None:
        if self.center is not None and self.contour_points is not None:
            self._initialize_contour()

    # Calls contour.initialize_contour with the current controls and draws the circle.
    def _initialize_contour(self) -> None:
        succeeded, points = run_algorithm(
            self, contour.initialize_contour, self.center,
            self.radius_slider.value(), self.n_points_slider.value())
        if not succeeded:
            self._redraw("Center set")
            return
        self.contour_points = np.asarray(points, dtype=float)
        self._clear_results()
        self._redraw(f"Initial contour · {len(self.contour_points)} points")

    # Evolves the snake, then computes chain code, perimeter and area.
    def run_snake(self) -> None:
        if self.image is None:
            QMessageBox.information(self, "No image", "Upload an image first.")
            return
        if self.contour_points is None:
            QMessageBox.information(self, "No contour", "Click “Draw contour”, then click on the image to place the snake.")
            return
        succeeded, evolved = run_algorithm(
            self, contour.evolve_contour, self.contour_points.copy(), self.image,
            self.alpha_slider.value(), self.beta_slider.value(),
            self.iterations_slider.value(), self.window_slider.value())
        if not succeeded:
            return
        self.contour_points = np.asarray(evolved, dtype=float)
        self._redraw("Evolved contour")
        self._compute_measurements()

    # Calls chain code, perimeter and area in order, stopping at the first failure.
    def _compute_measurements(self) -> None:
        succeeded, chain_code = run_algorithm(self, contour.to_chain_code, self.contour_points)
        if not succeeded:
            return
        succeeded, perimeter = run_algorithm(self, contour.compute_perimeter, chain_code)
        if not succeeded:
            return
        succeeded, area = run_algorithm(self, contour.compute_area, self.contour_points)
        if not succeeded:
            return
        self.perimeter_value.setText(f"{perimeter:.2f}")
        self.area_value.setText(f"{area:.1f}")
        self.chain_length_value.setText(str(len(chain_code)))
        self.chain_code_box.setPlainText("".join(str(direction) for direction in chain_code))
        self.status_message.emit(f"Snake done — perimeter {perimeter:.2f} px, area {area:.1f} px²")

    # Redraws the image with the current snake and center on it.
    def _redraw(self, caption: str) -> None:
        # caption: short description shown above the image
        self.shown_image = draw_contour(self.image, self.contour_points, self.center)
        self.canvas.set_image(self.shown_image)
        self.canvas_caption.setText(caption)

    # Resets the result tiles and chain code box.
    def _clear_results(self) -> None:
        for value_label in (self.perimeter_value, self.area_value, self.chain_length_value):
            value_label.setText("—")
        self.chain_code_box.clear()

    # Copies the chain code text to the clipboard.
    def _copy_chain_code(self) -> None:
        QApplication.clipboard().setText(self.chain_code_box.toPlainText())
        self.status_message.emit("Chain code copied")

    # Saves the image with the snake drawn on it.
    def save_result(self) -> None:
        save_with_dialog(self, self.shown_image, self.status_message)


class MainWindow(QMainWindow):
    """Top-level window: header, the two tabs, and a status bar."""

    # Builds the header, adds both tabs and connects their status messages.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Task 2 — Shape Detection & Active Contour")
        self.resize(1440, 900)
        shape_tab, contour_tab = ShapeDetectionTab(), ActiveContourTab()
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.tabBar().setDrawBase(False)
        tabs.addTab(shape_tab, "Shape Detection")
        tabs.addTab(contour_tab, "Active Contour")
        for tab in (shape_tab, contour_tab):
            tab.status_message.connect(lambda text: self.statusBar().showMessage(text, 8000))
        root = QWidget(objectName="root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 10)
        layout.setSpacing(14)
        layout.addLayout(self._build_header())
        layout.addWidget(tabs, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready — upload an image to begin")

    # Builds the title block: accent dot + app name, with the subtitle underneath.
    def _build_header(self) -> QVBoxLayout:
        dot = QLabel(objectName="accentDot")
        dot.setFixedSize(10, 10)
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        title_row.addWidget(dot, 0, Qt.AlignVCenter)
        title_row.addWidget(QLabel("Shape Detection & Active Contour", objectName="appTitle"))
        title_row.addStretch()
        subtitle = QLabel("Canny edges  ·  Hough lines, circles, ellipses  ·  Greedy snake", objectName="appSubtitle")
        subtitle.setContentsMargins(22, 0, 0, 0)
        header = QVBoxLayout()
        header.setSpacing(0)
        header.addLayout(title_row)
        header.addWidget(subtitle)
        return header
