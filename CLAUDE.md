# CLAUDE.md — CV Task 2: Shape Detection & Active Contour

> Read this file completely before writing any code.
> This is the single source of truth for the project.
> Deadline: Monday. Keep everything simple and navigable.

---

## 1. What this project does

A desktop application (PyQt5) that:
1. Detects edges using **Canny** (from scratch)
2. Detects **lines, circles, and ellipses** using Hough Transform, draws them on the image
3. Lets the user initialize an **Active Contour (snake)** around an object, evolves it to fit the boundary, outputs **chain code**, perimeter, and area

---

## 2. Team & ownership

| Person | File | Responsibility |
|--------|------|----------------|
| Lead (Yomna) | `main.py`, `ui.py`, `image_io.py` | App launch, all UI windows and buttons, image loading/saving/display |
| B | `canny.py` | All 5 Canny steps from scratch |
| C | `hough.py` | Hough lines, circles (optimized), ellipses (Xie & Ji 2002) |
| D | `contour.py` | Active contour snake (greedy), chain code, perimeter, area |

**Rule:** only edit your own file. If you need something from another file, call the function — don't copy the logic.

---

## 3. File structure

```
cv-task2/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── main.py              # Lead — entry point, launches the app
├── ui.py                # Lead — all PyQt5 UI: windows, buttons, tabs, sliders
├── image_io.py          # Lead — load image, save image, convert for display
├── canny.py             # B — Canny edge detection from scratch
├── hough.py             # C — Hough transform: lines, circles, ellipses
├── contour.py           # D — Active contour, chain code, perimeter, area
└── report/
    ├── report_lead.md   # Lead fills this
    ├── report_B.md      # B fills this
    ├── report_C.md      # C fills this
    └── report_D.md      # D fills this
```

No subfolders. No helper files. If you feel the urge to create a new file, talk to the Lead first.

---

## 4. Rules everyone must follow

### 4.1 From scratch vs library
- **Canny:** every single step is from scratch. No `cv2.Canny`.
- **Hough lines and circles:** from scratch. No `cv2.HoughLines` or `cv2.HoughCircles`.
- **Ellipse (Xie & Ji):** from scratch based on the paper.
- **Active contour:** from scratch (greedy algorithm).
- **Allowed libraries:** NumPy for array math, `cv2` only for reading/writing images and displaying results, `scipy.ndimage` only if needed for Gaussian blur inside Canny, PyQt5 for UI.
- **Chain code:** from scratch.

### 4.2 Code quality rules
- Every function has a **one-line comment** above it saying what it does in plain English.
- Every parameter has a comment saying what it controls.
- No function longer than 50 lines. If it is, split it.
- No duplicated logic across files.
- Variable names must be readable: `gradient_magnitude` not `gm`, `accumulator` not `acc2`.

### 4.3 The discussion rule
Before the discussion, every person must be able to answer:
- What does each of your functions do?
- What happens if I change this parameter up or down?
- Which function calls which, and in what order?

If you cannot answer these, the code is not ready.

---

## 5. Function signatures (the contract between teammates)

These are fixed. Do not change them without telling everyone.

### `image_io.py` (Lead)
```python
def load_image(path: str) -> np.ndarray:
    # Returns RGB numpy array

def save_image(img: np.ndarray, path: str) -> None:
    # Saves image to disk

def to_display(img: np.ndarray) -> QPixmap:
    # Converts numpy array to QPixmap for showing in PyQt5 window
```

### `canny.py` (B)
```python
def gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    # Step 1: smooth the image to reduce noise

def gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Step 2: compute gradient magnitude and direction
    # Returns: (magnitude, direction)

def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    # Step 3: thin edges to 1 pixel wide

def double_threshold(img: np.ndarray, low: float, high: float) -> np.ndarray:
    # Step 4: classify pixels as strong, weak, or suppressed

def hysteresis(img: np.ndarray) -> np.ndarray:
    # Step 5: keep weak edges only if connected to strong ones

def run_canny(img: np.ndarray, kernel_size: int, sigma: float, low: float, high: float) -> np.ndarray:
    # Runs all 5 steps in order and returns the final edge image
    # This is the only function the UI calls
```

### `hough.py` (C)
```python
def hough_lines(edge_img: np.ndarray, threshold: int) -> list[tuple]:
    # Detects straight lines. Returns list of (rho, theta) pairs.

def hough_circles(edge_img: np.ndarray, r_min: int, r_max: int, threshold: int) -> list[tuple]:
    # Detects circles. Loops over radii to keep accumulator 2D.
    # Returns list of (x_center, y_center, radius)

def hough_ellipses(edge_img: np.ndarray, threshold: int) -> list[tuple]:
    # Detects ellipses using Xie & Ji (2002) method.
    # Returns list of (x_center, y_center, a, b, angle)

def draw_shapes(img: np.ndarray, lines: list, circles: list, ellipses: list) -> np.ndarray:
    # Draws all detected shapes on a copy of the image and returns it
    # This is the only drawing function — no drawing elsewhere
```

### `contour.py` (D)
```python
def initialize_contour(center: tuple, radius: int, n_points: int) -> np.ndarray:
    # Creates a circle of n_points evenly spaced around the center
    # Returns array of shape (n_points, 2) — each row is (x, y)

def compute_energy(contour: np.ndarray, img: np.ndarray, alpha: float, beta: float) -> float:
    # Computes total energy of the contour (image energy + internal energy)
    # alpha controls elasticity, beta controls smoothness

def evolve_contour(contour: np.ndarray, img: np.ndarray, alpha: float, beta: float,
                   n_iterations: int, window_size: int) -> np.ndarray:
    # Runs the greedy algorithm: moves each point to minimize energy
    # Returns the final evolved contour

def to_chain_code(contour: np.ndarray) -> list[int]:
    # Converts contour points to chain code (0–7 directions)

def compute_perimeter(chain_code: list[int]) -> float:
    # Computes perimeter from chain code

def compute_area(contour: np.ndarray) -> float:
    # Computes area enclosed by the contour using the shoelace formula
```

---

## 6. UI layout (Lead builds this)

**Two tabs:**

**Tab 1 — Shape Detection**
- Upload image button
- Run Canny button + sliders: kernel size, sigma, low threshold, high threshold
- Run Hough button + controls:
  - Checkbox: detect lines / circles / ellipses (can select multiple)
  - For circles: min radius, max radius sliders
  - Threshold slider (shared)
- Display: original image on the left, result on the right
- Save result button

**Tab 2 — Active Contour**
- Upload image button (coin image)
- Controls: number of contour points, radius, alpha, beta, iterations, window size
- "Draw contour" button — user clicks on the image to set center
- "Run snake" button — evolves the contour
- Display: image with contour drawn on it (updates each iteration or at end)
- Shows: chain code output, perimeter value, area value
- Save result button

---

## 7. How the UI calls the algorithms (the chain you must know)

**Tab 1 flow:**
```
User clicks "Run Canny"
  → ui.py reads slider values
  → calls canny.run_canny(img, kernel_size, sigma, low, high)
  → displays result

User clicks "Run Hough"
  → ui.py reads checkboxes and slider values
  → calls hough.hough_lines / hough_circles / hough_ellipses on the Canny result
  → calls hough.draw_shapes(original_img, lines, circles, ellipses)
  → displays result
```

**Tab 2 flow:**
```
User clicks on image → sets center point
  → ui.py reads n_points and radius from controls
  → calls contour.initialize_contour(center, radius, n_points)
  → draws initial contour on image

User clicks "Run snake"
  → ui.py reads alpha, beta, iterations, window_size
  → calls contour.evolve_contour(contour, img, alpha, beta, iterations, window_size)
  → calls contour.to_chain_code(final_contour)
  → calls contour.compute_perimeter(chain_code)
  → calls contour.compute_area(final_contour)
  → displays result + chain code + numbers
```

This section is what you study before the discussion. Know it by heart.

---

## 8. Report instructions (`report/`)

Each person fills in their own `report_<name>.md` before submission.

**Required sections in every report:**
1. What I implemented (brief description of each function)
2. Algorithm explanation in plain words (step by step, no copy-paste from slides)
3. Parameter effects:
   - For each parameter your feature has: what happens when you increase it? What happens when you decrease it? Show an example image.
4. Observations: anything unexpected you noticed while testing
5. Screenshots of results on at least 2 different images

**B's report must include:** effect of sigma, kernel size, low threshold, high threshold on the final edges.
**C's report must include:** effect of threshold on lines/circles/ellipses, r_min/r_max effect, comparison of results on different images.
**D's report must include:** effect of alpha, beta, n_points, window_size, iterations on the final contour shape and convergence.
**Lead's report must include:** UI decisions, integration notes, any issues encountered.

---

## 9. Git workflow

- Branch names: `canny` (B), `hough` (C), `contour` (D), `ui` (Lead)
- Commit messages: short and clear — "add gaussian blur step", "fix circle accumulator", not "update file"
- Merge to main only when your feature works and you've tested it
- Do not touch other people's files

---

## 10. Phase 0 — what to build first (Lead, run this in Claude Code)

When asked to "run Phase 0":

1. Create the full file structure above
2. Every algorithm file (`canny.py`, `hough.py`, `contour.py`) exists with:
   - All functions from Section 5 with their exact signatures
   - A one-line comment on each function
   - Body raises `NotImplementedError("TODO (B/C/D): description")`
3. `image_io.py` fully working: load, save, to_display
4. `ui.py` fully working: both tabs, all controls, all buttons wired up, shows "Not implemented yet" message when algorithm buttons are clicked
5. `main.py` working: launches the app, shows the window
6. `report/` folder with 4 empty template files, each with the section headers from Section 8
7. `requirements.txt` with: numpy, opencv-python, PyQt5, scipy
8. `README.md` with: how to install, how to run, team table, file map

**Acceptance check:** `python main.py` opens a working two-tab window. Clicking algorithm buttons shows "Not implemented yet". Image upload and display works.
