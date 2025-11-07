Head-Tracking Mouse Control System

A real-time system that maps head movement to mouse cursor position using OpenCV and facial detection. Designed for accessibility use cases, enabling hands-free computer interaction.

Purpose
Demonstrate real-time computer vision and human-computer interaction principles.
Provide a practical solution for users with limited hand mobility.
Practice video stream processing, facial landmark detection, and system-level automation.

Features
Real-time face detection using OpenCV's Haar Cascade classifier.
Smooth mouse cursor movement with configurable sensitivity and interpolation.
Dynamic sensitivity adjustment (`+` / `-` keys) and edge-boosting algorithm for precise control.
On-screen display of FPS, coordinates, and sensitivity settings.
Performance optimizations (resolution reduction, buffer management) for stable 60 FPS.
Leverages AI-assisted development (e.g., ChatGPT, Qwen) for efficient code generation and debugging.

Requirements
To run this project, you need Python 3.8 or higher.

How to Run
Clone this repository or navigate to the project directory.
Ensure all requirements from requirements.txt are installed.
Run the script:
python cv2_fullmouse_control.py

Allow camera access if prompted. The system will start tracking your head movement.
Press ESC to quit the application.

Notes:
Requires a working webcam.
Performance may vary depending on your computer's CPU and camera capabilities.
Ensure adequate lighting for optimal face detection.

Acknowledgements
Libraries: OpenCV , PyAutoGUI
