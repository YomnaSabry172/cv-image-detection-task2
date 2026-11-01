# Report — C: Hough transform (lines, circles, ellipses)

## 1. What I implemented

| Function | Description |
|----------|-------------|
| `_find_peaks(accumulator, threshold, nms_window)` | Shared helper that scans a 2D accumulator for cells above a vote threshold, suppressing nearby duplicates with a local-maximum window so we don't return many near-identical detections. |
| `hough_lines(edge_img, threshold)` | Detects straight lines using the polar Hough transform. Builds a (ρ, θ) accumulator, lets every edge pixel vote for all 180 θ values at once (vectorised), then extracts peaks. Returns a list of (rho, theta) pairs. |
| `hough_circles(edge_img, r_min, r_max, threshold)` | Detects circles by looping over each candidate radius separately, keeping the accumulator 2D (height × width) instead of a huge 3D cube. For each radius, every edge pixel votes at once (one array operation + `np.bincount`) for the centre positions on a circle of that radius, one vote per pixel of the digital circle, so the threshold means "how many edge pixels lie on the circle". Peaks from neighbouring radii are then merged (`_merge_circles`), keeping only the strongest circle of each group. Returns a list of (x_center, y_center, radius). |
| `_ellipses_for_anchor(...)` | Helper for ellipse detection. For one anchor point P1 it handles every partner P2 and every third point P3 at once as 2D arrays, using the geometric relationship from Xie & Ji (2002) to compute each candidate semi-minor axis b, and returns the ellipses whose b-vote count reaches the threshold. |
| `_deduplicate_ellipses(ellipses, merge_distance)` | Removes near-duplicate ellipse detections (centre and semi-major axis both within merge_distance pixels). The list is sorted strongest-first, so the best detection of each group is the one kept. |
| `hough_ellipses(edge_img, threshold)` | Detects ellipses using the Xie & Ji (2002) method. For every pair of edge points, treats their midpoint as a candidate centre and their half-distance as semi-major axis a. Then scores all other edge points in a 1D accumulator for the semi-minor axis b. Returns a list of (x_center, y_center, a, b, angle). |
| `draw_shapes(img, lines, circles, ellipses)` | Draws all detected shapes on a **copy** of the image. Lines in red, circles in green, ellipses in blue. Uses cv2 drawing functions only (allowed — these are not detection functions). |

## 2. Algorithm explanation

### Hough Lines

The key insight: every straight line can be described by two numbers — **ρ** (perpendicular distance from the origin) and **θ** (angle of that perpendicular). The equation is `ρ = x·cos(θ) + y·sin(θ)`. This avoids the infinite-slope problem of `y = mx + c`.

**Step by step:**

1. Compute the image diagonal `d = √(H² + W²)` — this is the maximum possible |ρ|.
2. Create a 2D accumulator with rows for ρ (from −d to +d) and 180 columns for θ (from 0° to 179°).
3. Find all white (edge) pixels in the Canny output.
4. For each edge pixel (x, y): compute `ρ = x·cos(θ) + y·sin(θ)` for every θ simultaneously (vectorised with NumPy), and add 1 vote to the corresponding accumulator cell.
5. After all pixels have voted, find peaks — cells whose vote count is above the threshold and that are the local maximum in a 9×9 window (non-maximum suppression).
6. Convert each peak's (row, col) index back to (ρ, θ) and return them.

### Hough Circles

A circle has 3 parameters: centre_x, centre_y, and radius. A 3D accumulator would be enormous, so we **loop over each radius** separately and use a clean 2D accumulator (height × width) for each one.

**Step by step:**

1. Work out the radius-specific set of offsets: the distinct pixels of a circle of radius r (about 2πr of them).
2. For each radius r in [r_min, r_max]:
   - Create a fresh 2D accumulator (same size as the image).
   - Every edge pixel (x, y) votes for the centres `(x − offset_x, y − offset_y)`; all edge pixels vote in one array operation and the votes are counted with `np.bincount`. Big radii vote with at most 120 sampled angles and the result is scaled back to "edge pixels on the full circle", which keeps big radii fast.
   - Only count votes that fall inside the image bounds.
   - Find peaks with NMS, same as for lines. A peak must reach the threshold **and** at least 40% of the circle's outline, so a short arc of a bigger circle is not reported as a small circle.
3. Collect the peaks of all radii with their vote counts, sort by votes, and drop any circle whose centre is within 10 px and radius within 5 px (or 5% of the radius, whichever is larger) of a stronger one. This also merges the inner and outer edge of a thick outline into one circle.

### Hough Ellipses (Xie & Ji 2002)

An ellipse has 5 parameters (centre_x, centre_y, semi-major a, semi-minor b, angle). A 5D accumulator is completely impractical, so Xie & Ji reduce the problem to a series of **1D accumulator** lookups.

**Step by step:**

1. Get all edge pixels. If there are more than 300, randomly subsample to 300 with a fixed seed so results are repeatable (otherwise the O(n³) runtime becomes too slow).
2. For every pair of edge points (P1, P2):
   - Compute their distance. Skip if it's too small (<10 px) or longer than the image side (this is the full major axis, so the limit has to be the whole image side, not half of it).
   - The midpoint of P1–P2 is the candidate centre; half the distance is the semi-major axis a; the angle of the P1–P2 segment is the orientation.
   - Create a 1D accumulator for candidate semi-minor axis b (from 0 to a).
   - For every other edge point P3:
     - Compute its distance d from the candidate centre.
     - If d ≥ a or d < 1, skip (P3 is outside the ellipse or too close to the centre).
     - Project the vector (P3 − centre) onto the major-axis direction to get cos(τ).
     - Use the ellipse equation: `b² = (a² · d² · sin²τ) / (a² − d² · cos²τ)`.
     - If b² is valid (positive, ≤ a²), vote for `b = √(b²)` in the accumulator.
   - If the peak of the accumulator ≥ threshold, accept this ellipse.
3. Sort the accepted ellipses by votes (strongest first) and remove near-duplicates (centres and semi-major axes within 20 px), so the best detection of each group is kept.

In the code, the "for every other edge point" in step 2 is not a Python loop: for one P1, all partners P2 and all third points P3 are processed together as 2D arrays and the b-votes of every pair are counted with a single `np.bincount`.

### Drawing

`draw_shapes` works on a **copy** of the original image:
- Lines: compute two endpoints far beyond the image edges along the direction perpendicular to the (ρ, θ) normal, then draw with `cv2.line`.
- Circles: draw with `cv2.circle`.
- Ellipses: draw with `cv2.ellipse`.

## 3. Parameter effects

### Effect of threshold on lines / circles / ellipses

_Increase:_ Only the strongest, most prominent shapes survive. On a building image, raising the threshold from 80 to 200 keeps only the longest, straightest edges of the building and drops all the small spurious lines.

_Decrease:_ Many more shapes appear, including false positives. Short line segments, small arcs that aren't really circles, and noise all start being detected. On a coin image, lowering the circle threshold too far produces dozens of overlapping green rings where there should be just one.

### Effect of r_min / r_max

_Increase r_min:_ Small circles are ignored. If r_min = 50, a coin with radius 30 won't be found. Useful to skip noisy small detections.

_Decrease r_min:_ Smaller circles can be detected, but noise circles become more common and the search takes longer because more radii are tested.

_Increase r_max:_ Larger circles can be found, but the algorithm runs slower because it iterates over more radii. If set much larger than any real circle in the image, it just wastes time without finding anything new.

_Decrease r_max:_ Large circles are missed. If the biggest coin in the image has radius 80 and r_max = 60, it won't be detected.

_Best practice:_ Set r_min and r_max to bracket the expected object sizes in the image. A tight range (e.g. 30–60 for coins) runs much faster and gives cleaner results than a wide range (5–300).

### Comparison of results on different images

**Building/road image (lines):** Hough lines work very well — strong straight edges produce tall peaks in the accumulator. The main challenge is picking the right threshold: too low and every brick edge becomes a line; too high and only 1–2 of the biggest walls are detected.

**Coin image (circles):** Circle detection is reliable when the radius range is set correctly. Multiple coins of similar size are found simultaneously. If the coins are different sizes, you need a wider r_min–r_max range, which is slower.

**Oval/egg image (ellipses):** Ellipse detection is slower and more sensitive to noise. The random subsampling means results can vary slightly between runs. Works best on images with one or two clean elliptical shapes against a plain background.

## 4. Observations

1. **Hough lines are fast** — all edge pixels vote for all 180 θ values in one array operation (`np.bincount`), so the accumulator fills in a fraction of a second. Peak finding only checks the cells above the threshold, so it is cheap too.

2. **Circle detection speed depends heavily on the radius range.** Each extra radius value adds one more vectorised voting pass over all edge pixels. Keeping r_max − r_min small (say, 20–30) makes a big difference. For circles the threshold means "edge pixels on the circle", so a larger circle can reach a higher threshold than a small one.

3. **Ellipse detection is still the slowest**, but vectorising the P2/P3 loops makes it take only a few seconds (about 2–5 s measured for 300 points). The random subsampling to 300 points (fixed seed) is still needed because memory and time grow with the square and cube of the point count. Because the sample is spread over all edges, an ellipse in a busy image only gets a few points, so use a higher threshold on clean images and expect misses in cluttered ones. A low threshold also produces extra false ellipses, because pairs of points that are not the true major axis can still reach it.

4. **NMS window matters.** Without non-maximum suppression on the accumulator, a single strong line produces a cluster of 5–10 nearby peaks. The 9×9 NMS window collapses these into one clean detection.

5. **The threshold is relative to the image.** A threshold of 100 might be perfect for a 640×480 image but far too low for a 1920×1080 image with more edge pixels. Users need to experiment.

6. **Drawing on a copy is important.** Early testing without `.copy()` caused the original image to get overwritten, which broke subsequent detections when the user re-ran Hough without re-uploading.

## 5. Screenshots

_To be added after testing with the UI — run `python main.py`, upload at least 2 different images, capture screenshots of the detected shapes._
