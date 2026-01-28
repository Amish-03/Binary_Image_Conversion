# Binary Image Conversion Visualization

A real-time educational tool to demonstrate how **binary thresholding** works using computer vision. This tool visualizes the intensity histogram of a webcam feed, detects local minima/peaks, and automatically selects an optimal threshold to separate foreground from background.

## Features

- **Real-time Visualization**: Displays the original feed, binary output, and intensity histogram in a single integrated window.
- **Histogram Analysis**:
  - **Smoothed Distribution**: Gaussian smoothing to reduce noise.
  - **Minima Detection (Red Dots)**: Identifies potential threshold candidates (valleys).
  - **Peak Detection (Green Dots)**: Identifies dominant intensity clusters.
- **Auto Thresholding (Hysteresis)**:
  - Automatically selects the best threshold based on the deepest valley between peaks.
  - Implements "sticky" logic to prevent flickering when multiple valleys are present.
- **Interactive Manual Mode**:
  - Click anywhere on the histogram to manually set the threshold.
  - Toggle between **Auto** and **Manual** modes via a button or keyboard shortcut.
- **Performance**: Optimized for smooth ≥20 FPS performance using `matplotlib` backend integration.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Amish-03/Binary_Image_Conversion.git
    cd Binary_Image_Conversion
    ```

2.  Install dependencies:
    ```bash
    pip install opencv-python numpy matplotlib scipy
    ```

## Usage

Run the script:

```bash
python thresholding_tool.py
```

### Controls

-   **Click on Graph**: Set manual threshold.
-   **Click 'AUTO/MANUAL' Button**: Toggle mode.
-   **`m`**: Toggle Auto/Manual Mode via keyboard.
-   **`q`**: Quit application.

## How it Works

The tool computes the grayscale intensity histogram (0-255). It smooths this data to find significant "valleys" (local minima). In **Auto Mode**, it selects the most significant valley to separate the dark and light pixels, effectively segmenting the image.