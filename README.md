# CV Task 2 — Shape Detection & Active Contour

A PyQt5 desktop app that:

1. Detects edges with **Canny** (from scratch)
2. Detects **lines, circles and ellipses** with the Hough transform and draws them on the image
3. Fits an **active contour (greedy snake)** around an object and reports its **chain code**, **perimeter** and **area**

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

- **Shape Detection tab:** upload an image (or drag it onto the left panel), click **Run Canny**, tick the shapes you want, then click **Run Hough**.
- **Active Contour tab:** upload a coin image, click **Draw contour**, click the center of the coin, then click **Run snake**.

Hover any slider to see what its parameter does. Until an algorithm is written, its button shows "Not implemented yet".

## Team

| Person | Files | Responsibility |
|--------|-------|----------------|
| Lead (Yomna) | `main.py`, `ui.py`, `image_io.py` | App launch, UI, image loading / saving / display |
| B | `canny.py` | Canny edge detection (5 steps) |
| C | `hough.py` | Hough lines, circles, ellipses (Xie & Ji 2002), drawing |
| D | `contour.py` | Greedy snake, chain code, perimeter, area |

## File map

```
main.py          entry point — creates the app and opens the window
ui.py            theme, main window, both tabs, all controls
image_io.py      load_image / save_image / to_display
canny.py         gaussian_blur → gradient → non_maximum_suppression → double_threshold → hysteresis
hough.py         hough_lines, hough_circles, hough_ellipses, draw_shapes
contour.py       initialize_contour, compute_energy, evolve_contour, to_chain_code, compute_perimeter, compute_area
report/          one report per team member
requirements.txt numpy, opencv-python, PyQt5, scipy
CLAUDE.md        project rules and function contracts — read this first
```

## Notes for teammates

- The UI passes the **RGB image as loaded** to `run_canny` and `evolve_contour`. Convert to grayscale inside your function.
- Canny thresholds arrive in the **0–255** range.
- Return any image type (bool, float, uint8). The UI scales it for display.
- If your function raises an exception, the app shows the error message and prints the traceback in the terminal. It does not crash.
