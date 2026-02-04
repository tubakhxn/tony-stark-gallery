# Futuristic 3D Holographic Photo Gallery

A real-time hand gesture-controlled 3D holographic photo gallery using OpenCV, MediaPipe, and PyOpenGL.
Dev/Creator= tubakhxn
## Features

- Real-time hand tracking using MediaPipe
- 3D holographic rendering with PyOpenGL
- Gesture controls:
  - Move hand left/right → rotate gallery
  - Move hand up/down → zoom in/out
  - Pinch → select and highlight image
  - Twist hand → spin carousel
- Dark starfield background with neon effects
- Smooth cinematic transitions
- UI overlay showing selected image

## Requirements

- Python 3.8+
- Webcam

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your images to the `gallery/` folder (supports JPG, PNG, GIF)

3. Run the application:
```bash
python holographic_gallery.py
```

## Controls

- **Hand Movement**: Move your hand in front of the camera
- **Rotate Gallery**: Move hand left/right
- **Zoom**: Move hand up/down
- **Select Image**: Pinch with thumb and index finger
- **Spin Carousel**: Rotate thumb and index finger
- **Exit**: Press 'q' or ESC

## Project Structure

```
gallery/
├── holographic_gallery.py    # Main application
├── requirements.txt          # Dependencies
├── README.md                # This file
└── gallery/                 # Image folder
    ├── sample1.jpg
    ├── sample2.jpg
    └── ...
```