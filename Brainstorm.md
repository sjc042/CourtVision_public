### 03/31/2026
- After Day3 and custom model training and visual testing, detection with YOLOv8n F16 yields about 10 FPS, "made" detection is very unreliable, has very low Recall, higher Precision. Potential reason: low fps, skipped frames with ball in rim/net; small objects of ball and rim. So if "made" is present, then very likely made shot. But need to mainly rely on other info, such as extrapolated ball trajectory (sparse actual detection) and rim intersection.
    - Detection of small objects can be somewhat mitigated as such:
        - Once the court outline is configured, zoom-in/crop the court area ROI
        - This makes detecting "ball", "rim", and "made" qualitatively significantly easier.
    - Use court mapping to heuristicly find virtual ROI (square, needs to be tested) for rim detection filtering, then run a moving average (very small update ~0.01) of rim box. If multiple rim detected inside virtual ROI, compute confidence weighted bbox average.\
    - **Caveat** Need to determine which comes first and helps identify the other? rim->court? court->rim?
    - Free-throw calibration and manual court line calibration (as in HomeCourt)
        - Manual court line calibration rotates around vertical axis at rim (in HomeCourt)

- For ball arch, calculate angle by sampling points from release point to a little past peak; for made-shot detection, calculate by sampling from peak position to height plane (parallel to ground) at rim height; this is to help reduce "A physical parabola under gravity projects to a conic section (not necessarily a parabola) through a perspective camera"
- If need for Person tracking AND need for ReID emerges, instead of using deep learning based methods, consider using simpler image compression methods such as eigen value computing, pHash similar method (but not gray scale), etc.
    - pHash (perceptual hash) actually understands the image. The steps:
        1. Shrink — resize to a tiny fixed square (e.g. 32×32 for hash_size=16). This discards high-frequency noise and makes the hash resolution-independent.
        2. Grayscale — convert to grayscale. Color is irrelevant for structural similarity.
        3. DCT (Discrete Cosine Transform) — apply a 2D DCT to the pixel grid. This is the same transform used inside JPEG compression. It converts the image from pixel space into frequency space — low frequencies (overall structure, gradients) in the top-left, high frequencies (fine detail) in the bottom-right.
        4. Keep only low frequencies — take just the top-left hash_size × hash_size block (e.g. 16×16 = 256 values). This is the structural "fingerprint" of the image, ignoring texture and noise.
        5. Binarize — compute the mean of those 256 values, then for each value: 1 if above mean, 0 if below. This gives you a 256-bit hash.
---