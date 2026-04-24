# Canny Edge Detection from Scratch

This repository provides a step-by-step implementation of the Canny Edge Detection algorithm. Unlike library-based solutions, this project builds every stage of the pipeline manually to demonstrate a deep understanding of image processing fundamentals.

## Pipeline Overview
1. **Noise Reduction:** Gaussian blur is applied to remove image noise that could cause false edge detection.
2. **Gradient Calculation:** Using Sobel operators to find intensity changes in horizontal and vertical directions.
3. **Non-Maximum Suppression:** Thinning out the edges by suppressing pixels that are not local maxima in the gradient direction.
4. **Double Thresholding:** Identifying strong, weak, and non-relevant pixels based on intensity thresholds.
5. **Hysteresis Tracking:** Finalizing the edge map by keeping weak edges only if they are connected to strong ones.

## Requirements
- `numpy`, `scipy`, `scikit-image`, `matplotlib`

To install dependencies, run: `pip install -r requirements.txt`