# Dynamic View Synthesis with 3D Point Cloud Reconstruction

This project demonstrates the application of deep learning for depth estimation to simulate changing viewpoints from fisheye images using the MiDaS model from Intel. It employs OpenCV and PyQt for handling image operations and user interface interactions. 

A significant feature is the transformation of estimated depth maps into 3D point clouds, which are then manipulated to simulate a change in viewpoint based on user-defined parameters. 

## Setup Instructions

1. Clone this repository to your local machine.
2. Ensure you have Python 3.8 or newer installed.
3. Install Open3D, PyTorch, OpenCV, and PyQt5 by running `pip install -r requirements.txt`.
4. Execute `python main.py` to launch the application. Use the GUI prompts to load a fisheye image and simulate a change in viewpoint.

Please note: This project involves 3D point cloud generation and transformation to emulate different camera angles, going beyond simple depth map visualization.
