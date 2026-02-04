#!/usr/bin/env python3
"""
Futuristic 3D Holographic Photo Gallery with Hand Gesture Controls

A real-time interactive 3D photo gallery that responds to hand gestures using OpenCV, 
MediaPipe, and PyOpenGL. Features holographic visual effects, smooth transitions, 
and intuitive gesture controls.

Author: GitHub Copilot
Date: September 2025
"""

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import OpenGL.GL as gl
import OpenGL.GLU as glu
import pygame
from pygame.locals import *
import math
import os
import glob
import time
from typing import List, Tuple, Optional


class HandGestureDetector:
    """Detects and processes hand gestures using MediaPipe."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Enable two hands!
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture state variables
        self.hand_position = (0.5, 0.5)  # Normalized position (0-1)
        self.hand2_position = (0.5, 0.5)  # Second hand position
        self.is_pinching = False
        self.is_pinching2 = False
        self.twist_angle = 0.0
        self.zoom_level = 1.0
        self.two_hands_detected = False
        self.hands_distance = 0.0
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Process video frame and extract hand gestures."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture_data = {
            'hand_detected': False,
            'position': self.hand_position,
            'hand2_position': self.hand2_position,
            'is_pinching': False,
            'is_pinching2': False,
            'twist_angle': 0.0,
            'zoom_level': 1.0,
            'two_hands_detected': False,
            'hands_distance': 0.0,
            'gesture_mode': 'normal'
        }
        
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                hand_landmarks_list.append(hand_landmarks.landmark)
            
            # Process first hand
            if len(hand_landmarks_list) >= 1:
                landmarks = hand_landmarks_list[0]
                
                # Palm center (approximate)
                palm_x = landmarks[9].x
                palm_y = landmarks[9].y
                
                # Thumb tip and index tip
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                
                # Calculate hand position (normalized)
                self.hand_position = (palm_x, palm_y)
                
                # Pinch detection (distance between thumb and index)
                pinch_distance = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 + 
                    (thumb_tip.y - index_tip.y) ** 2
                )
                self.is_pinching = pinch_distance < 0.05
                
                # Twist detection (angle between thumb and index)
                angle = math.atan2(
                    index_tip.y - thumb_tip.y,
                    index_tip.x - thumb_tip.x
                )
                self.twist_angle = math.degrees(angle)
                
                # Zoom based on hand height
                self.zoom_level = 2.0 - palm_y  # Higher = zoom in
                
                gesture_data.update({
                    'hand_detected': True,
                    'position': self.hand_position,
                    'is_pinching': self.is_pinching,
                    'twist_angle': self.twist_angle,
                    'zoom_level': self.zoom_level
                })
            
            # Process second hand if available
            if len(hand_landmarks_list) >= 2:
                landmarks2 = hand_landmarks_list[1]
                
                # Second hand palm center
                palm2_x = landmarks2[9].x
                palm2_y = landmarks2[9].y
                self.hand2_position = (palm2_x, palm2_y)
                
                # Second hand pinch detection
                thumb_tip2 = landmarks2[4]
                index_tip2 = landmarks2[8]
                pinch_distance2 = math.sqrt(
                    (thumb_tip2.x - index_tip2.x) ** 2 + 
                    (thumb_tip2.y - index_tip2.y) ** 2
                )
                self.is_pinching2 = pinch_distance2 < 0.05
                
                # Distance between hands
                self.hands_distance = math.sqrt(
                    (palm_x - palm2_x) ** 2 + (palm_y - palm2_y) ** 2
                )
                
                self.two_hands_detected = True
                
                # Determine gesture mode based on hand positions
                gesture_mode = 'normal'
                if self.is_pinching and self.is_pinching2:
                    gesture_mode = 'shape_control'  # Both hands pinching = shape mode
                elif self.hands_distance > 0.3:
                    gesture_mode = 'spread_formation'  # Hands far apart = spread mode
                elif abs(palm_y - palm2_y) < 0.1 and abs(palm_x - palm2_x) > 0.2:
                    gesture_mode = 'line_formation'  # Hands horizontal = line mode
                elif abs(palm_x - palm2_x) < 0.1 and abs(palm_y - palm2_y) > 0.2:
                    gesture_mode = 'vertical_formation'  # Hands vertical = tower mode
                
                gesture_data.update({
                    'hand2_position': self.hand2_position,
                    'is_pinching2': self.is_pinching2,
                    'two_hands_detected': True,
                    'hands_distance': self.hands_distance,
                    'gesture_mode': gesture_mode
                })
            else:
                self.two_hands_detected = False
        
        return frame, gesture_data


class HolographicGallery:
    """3D holographic photo gallery with OpenGL rendering."""
    
    def __init__(self, image_folder: str = "gallery"):
        self.image_folder = image_folder
        self.images = []
        self.textures = []
        self.image_names = []
        
        # Gallery state
        self.rotation_y = 0.0
        self.rotation_x = 0.0
        self.zoom = 5.0
        self.selected_image = 0
        self.carousel_rotation = 0.0
        
        # Formation modes
        self.formation_mode = 'spiral'  # spiral, grid, circle, line, tower, heart, star
        self.target_formation = 'spiral'
        
        # Animation smoothing
        self.target_rotation_y = 0.0
        self.target_zoom = 5.0
        self.target_carousel = 0.0
        self.formation_transition = 0.0  # 0-1 transition between formations
        
        # Visual effects
        self.time = 0.0
        self.stars = self._generate_stars(200)
        
        # Initialize gesture detector
        self.gesture_detector = HandGestureDetector()
        
        # Load images
        self._load_images()
        
    def _generate_stars(self, count: int) -> List[Tuple[float, float, float]]:
        """Generate random star positions for background."""
        stars = []
        for _ in range(count):
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-50, 50)
            z = np.random.uniform(-50, -10)
            stars.append((x, y, z))
        return stars
        
    def _load_images(self):
        """Load all images from the gallery folder."""
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
            
        # Supported image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(self.image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.image_folder, ext.upper())))
            
        if not image_files:
            print(f"No images found in {self.image_folder}/")
            print("Add some images to the gallery folder and restart.")
            return
            
        # Load and create textures
        for img_path in image_files[:20]:  # Limit to 20 images for performance
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                
                # Resize to power of 2 for OpenGL compatibility
                size = 512
                img = img.resize((size, size), Image.Resampling.LANCZOS)
                
                img_data = np.array(img)
                
                # Create OpenGL texture
                texture_id = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
                
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, 0, gl.GL_RGB, size, size, 0,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_data
                )
                
                self.textures.append(texture_id)
                self.images.append(img_data)
                self.image_names.append(os.path.basename(img_path))
                
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                
        print(f"Loaded {len(self.images)} images")
                
    def _get_formation_position(self, index: int, total: int, formation: str) -> Tuple[float, float, float]:
        """Calculate position based on formation type."""
        if formation == 'spiral':
            # Original spiral galaxy
            angle = (index / total) * 2 * math.pi
            spiral_factor = index / total
            radius = 3.0 + spiral_factor * 2.0
            height_offset = math.sin(spiral_factor * math.pi * 2) * 0.5
            x = radius * math.cos(angle)
            y = height_offset
            z = radius * math.sin(angle)
            
        elif formation == 'grid':
            # 3D Grid formation
            grid_size = int(math.ceil(math.sqrt(total)))
            row = index // grid_size
            col = index % grid_size
            x = (col - grid_size/2) * 2.5
            y = (row - grid_size/2) * 2.5
            z = math.sin(index * 0.5) * 1.0
            
        elif formation == 'circle':
            # Perfect circle
            angle = (index / total) * 2 * math.pi
            radius = 4.0
            x = radius * math.cos(angle)
            y = math.sin(angle * 3) * 0.5  # Wave effect
            z = radius * math.sin(angle)
            
        elif formation == 'line':
            # Straight line
            x = (index - total/2) * 2.0
            y = math.sin(index * 0.3) * 0.5
            z = 0
            
        elif formation == 'tower':
            # Vertical tower
            x = math.cos(index * 0.5) * 1.0
            y = index * 1.5 - total * 0.75
            z = math.sin(index * 0.5) * 1.0
            
        elif formation == 'heart':
            # Heart shape
            t = (index / total) * 2 * math.pi
            x = 3 * (16 * math.sin(t)**3)
            y = 3 * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            z = math.sin(t * 2) * 0.5
            x *= 0.2  # Scale down
            y *= 0.2
            
        elif formation == 'star':
            # Star formation
            angle = (index / total) * 2 * math.pi
            if index % 2 == 0:
                radius = 4.0  # Outer points
            else:
                radius = 2.0  # Inner points
            x = radius * math.cos(angle)
            y = math.sin(angle * 5) * 0.5
            z = radius * math.sin(angle)
            
        else:  # Default to spiral
            angle = (index / total) * 2 * math.pi
            radius = 3.0 + (index / total) * 2.0
            x = radius * math.cos(angle)
            y = 0
            z = radius * math.sin(angle)
            
        return (x, y, z)
        
    def _draw_starfield(self):
        """Draw animated starfield background."""
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_TEXTURE_2D)
        
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(0.8, 0.9, 1.0)  # Blue-white stars
        
        for star in self.stars:
            # Animated twinkling
            brightness = 0.5 + 0.5 * math.sin(self.time * 2 + star[0] * 0.1)
            gl.glColor3f(brightness, brightness, brightness)
            gl.glVertex3f(star[0], star[1], star[2])
            
        gl.glEnd()
        
        # Draw grid lines
        gl.glLineWidth(1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(0.0, 0.8, 1.0, 0.3)  # Cyan grid
        
        grid_size = 20
        grid_spacing = 2
        
        for i in range(-grid_size, grid_size + 1, grid_spacing):
            # Horizontal lines
            gl.glVertex3f(-grid_size, i, -20)
            gl.glVertex3f(grid_size, i, -20)
            
            # Vertical lines
            gl.glVertex3f(i, -grid_size, -20)
            gl.glVertex3f(i, grid_size, -20)
            
        gl.glEnd()
        
    def _draw_image_frame(self, texture_id: int, position: Tuple[float, float, float], 
                         scale: float = 1.0, selected: bool = False):
        """Draw a single image with holographic frame effect."""
        gl.glPushMatrix()
        
        gl.glTranslatef(position[0], position[1], position[2])
        gl.glScalef(scale, scale, scale)
        
        # Neon glow effect for selected image
        if selected:
            # Draw glow
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
            
            glow_size = 1.2
            glow_alpha = 0.5 + 0.3 * math.sin(self.time * 3)
            
            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(0.0, 1.0, 1.0, glow_alpha)  # Cyan glow
            gl.glVertex3f(-glow_size, -glow_size, 0.01)
            gl.glVertex3f(glow_size, -glow_size, 0.01)
            gl.glVertex3f(glow_size, glow_size, 0.01)
            gl.glVertex3f(-glow_size, glow_size, 0.01)
            gl.glEnd()
            
            gl.glDisable(gl.GL_BLEND)
        
        # Draw the image
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glColor4f(1.0, 1.0, 1.0, 0.9)  # Slightly transparent
        
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 1)
        gl.glVertex3f(-1, -1, 0)
        gl.glTexCoord2f(1, 1)
        gl.glVertex3f(1, -1, 0)
        gl.glTexCoord2f(1, 0)
        gl.glVertex3f(1, 1, 0)
        gl.glTexCoord2f(0, 0)
        gl.glVertex3f(-1, 1, 0)
        gl.glEnd()
        
        # Draw neon border
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glLineWidth(2.0)
        gl.glBegin(gl.GL_LINE_LOOP)
        
        if selected:
            gl.glColor4f(0.0, 1.0, 1.0, 1.0)  # Bright cyan for selected
        else:
            gl.glColor4f(0.5, 0.8, 1.0, 0.6)  # Dim blue for others
            
        gl.glVertex3f(-1.05, -1.05, 0.01)
        gl.glVertex3f(1.05, -1.05, 0.01)
        gl.glVertex3f(1.05, 1.05, 0.01)
        gl.glVertex3f(-1.05, 1.05, 0.01)
        gl.glEnd()
        
        gl.glPopMatrix()
        
    def _update_animations(self, dt: float):
        """Update smooth animations with interpolation."""
        lerp_speed = 5.0 * dt
        
        # Smooth rotation
        self.rotation_y += (self.target_rotation_y - self.rotation_y) * lerp_speed
        
        # Smooth zoom
        self.zoom += (self.target_zoom - self.zoom) * lerp_speed
        
        # Smooth carousel rotation
        self.carousel_rotation += (self.target_carousel - self.carousel_rotation) * lerp_speed
        
        # Formation transition
        if self.formation_mode != self.target_formation:
            self.formation_transition += dt * 2.0  # 2 second transition
            if self.formation_transition >= 1.0:
                self.formation_mode = self.target_formation
                self.formation_transition = 0.0
        
    def process_gestures(self, gesture_data: dict):
        """Process hand gestures and update gallery state."""
        if gesture_data['hand_detected']:
            hand_x, hand_y = gesture_data['position']
            
            # Two hands mode - Advanced controls!
            if gesture_data['two_hands_detected']:
                gesture_mode = gesture_data['gesture_mode']
                
                if gesture_mode == 'shape_control':
                    # Both hands pinching = cycle through formations
                    formation_list = ['spiral', 'grid', 'circle', 'line', 'tower', 'heart', 'star']
                    current_index = formation_list.index(self.target_formation) if self.target_formation in formation_list else 0
                    # Change formation based on hand movement
                    if abs(hand_x - gesture_data['hand2_position'][0]) > 0.3:
                        self.target_formation = formation_list[(current_index + 1) % len(formation_list)]
                        
                elif gesture_mode == 'spread_formation':
                    # Hands spread apart = grid formation
                    self.target_formation = 'grid'
                    distance = gesture_data['hands_distance']
                    self.target_zoom = 3.0 + distance * 8.0  # Zoom based on hand distance
                    
                elif gesture_mode == 'line_formation':
                    # Hands horizontal = line formation
                    self.target_formation = 'line'
                    self.target_rotation_y = (hand_x - 0.5) * 180
                    
                elif gesture_mode == 'vertical_formation':
                    # Hands vertical = tower formation
                    self.target_formation = 'tower'
                    avg_y = (hand_y + gesture_data['hand2_position'][1]) / 2
                    self.target_zoom = 3.0 + (1.0 - avg_y) * 5.0
                    
                else:
                    # Two hands detected but no special gesture = circle formation
                    self.target_formation = 'circle'
                    # Rotation based on average hand position
                    avg_x = (hand_x + gesture_data['hand2_position'][0]) / 2
                    self.target_rotation_y = (avg_x - 0.5) * 360
            
            else:
                # Single hand mode - Original controls + new formations
                
                # Rotation based on hand horizontal movement
                self.target_rotation_y = (hand_x - 0.5) * 360
                
                # Zoom based on hand vertical movement
                zoom_factor = gesture_data['zoom_level']
                self.target_zoom = 3.0 + zoom_factor * 4.0
                
                # Pinch gesture for image selection and formation change
                if gesture_data['is_pinching']:
                    # Select nearest image based on rotation
                    if self.images:
                        angle_per_image = 360.0 / len(self.images)
                        normalized_rotation = (self.rotation_y % 360) / angle_per_image
                        self.selected_image = int(normalized_rotation) % len(self.images)
                        
                        # Change formation based on hand height when pinching
                        if hand_y < 0.3:  # High = heart
                            self.target_formation = 'heart'
                        elif hand_y > 0.7:  # Low = star
                            self.target_formation = 'star'
                        else:  # Middle = spiral
                            self.target_formation = 'spiral'
                
                # Twist gesture for carousel spin
                twist_angle = gesture_data['twist_angle']
                self.target_carousel = twist_angle * 2.0
        
    def render(self, width: int, height: int):
        """Render the complete 3D holographic gallery."""
        if not self.images:
            return
            
        # Clear screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Set up perspective
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45, width / height, 0.1, 100.0)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        # Camera position
        glu.gluLookAt(0, 0, self.zoom, 0, 0, 0, 0, 1, 0)
        
        # Apply rotations
        gl.glRotatef(self.rotation_x, 1, 0, 0)
        gl.glRotatef(self.rotation_y + self.carousel_rotation, 0, 1, 0)
        
        # Draw starfield background
        self._draw_starfield()
        
        # Draw images in dynamic formations
        num_images = len(self.images)
        
        for i, texture_id in enumerate(self.textures):
            # Get position based on current formation
            if self.formation_transition > 0 and self.formation_mode != self.target_formation:
                # Transitioning between formations
                pos1 = self._get_formation_position(i, num_images, self.formation_mode)
                pos2 = self._get_formation_position(i, num_images, self.target_formation)
                
                # Lerp between positions
                t = min(self.formation_transition, 1.0)
                x = pos1[0] * (1-t) + pos2[0] * t
                y = pos1[1] * (1-t) + pos2[1] * t
                z = pos1[2] * (1-t) + pos2[2] * t
            else:
                # Use current formation
                x, y, z = self._get_formation_position(i, num_images, self.formation_mode)
            
            # Add some animation based on time
            time_offset = self.time + i * 0.1
            if self.formation_mode == 'heart':
                y += math.sin(time_offset) * 0.1  # Gentle heartbeat
            elif self.formation_mode == 'star':
                # Twinkling effect
                scale_mod = 1.0 + math.sin(time_offset * 3) * 0.1
            elif self.formation_mode == 'grid':
                z += math.sin(time_offset) * 0.2  # Floating effect
            
            # Scale selected image
            base_scale = 1.0
            if self.formation_mode == 'star' and 'scale_mod' in locals():
                base_scale *= scale_mod
                
            scale = base_scale * (1.5 if i == self.selected_image else 1.0)
            selected = i == self.selected_image
            
            self._draw_image_frame(texture_id, (x, y, z), scale, selected)
            
    def get_current_mode_info(self) -> str:
        """Get information about current formation and gesture mode."""
        if hasattr(self, 'last_gesture_data') and self.last_gesture_data.get('two_hands_detected'):
            gesture_mode = self.last_gesture_data.get('gesture_mode', 'normal')
            return f"Formation: {self.formation_mode.upper()} | Mode: TWO HANDS - {gesture_mode.upper()}"
        else:
            return f"Formation: {self.formation_mode.upper()} | Mode: SINGLE HAND"


class HolographicGalleryApp:
    """Main application class."""
    
    def __init__(self):
        self.width = 1200
        self.height = 800
        self.running = True
        
        # Initialize Pygame
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Futuristic 3D Holographic Photo Gallery")
        
        # Initialize OpenGL
        self._setup_opengl()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            self.running = False
            return
            
        # Initialize gallery
        self.gallery = HolographicGallery()
        
        # Font for UI overlay
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        
        # Timing
        self.clock = pygame.time.Clock()
        self.last_time = time.time()
        
    def _setup_opengl(self):
        """Configure OpenGL settings for holographic effects."""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Dark space background
        gl.glClearColor(0.02, 0.02, 0.1, 1.0)
        
        # Enable anti-aliasing
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        
    def _draw_ui_overlay(self):
        """Draw UI overlay with selected image info."""
        # Switch to 2D rendering
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width, 0, self.height, -1, 1)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_TEXTURE_2D)
        
        # Draw semi-transparent background for text
        gl.glEnable(gl.GL_BLEND)
        gl.glColor4f(0.0, 0.0, 0.0, 0.7)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(10, self.height - 80)
        gl.glVertex2f(400, self.height - 80)
        gl.glVertex2f(400, self.height - 10)
        gl.glVertex2f(10, self.height - 10)
        gl.glEnd()
        
        # Text would be rendered here with a proper text rendering system
        # For now, we'll use console output
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        # Restore matrices
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
    def run(self):
        """Main application loop."""
        print("Starting Futuristic 3D Holographic Photo Gallery")
        print("🎮 SINGLE HAND CONTROLS:")
        print("- Move hand left/right → rotate gallery")
        print("- Move hand up/down → zoom")
        print("- Pinch (thumb + index) → select image")
        print("- Pinch HIGH → Heart formation")
        print("- Pinch LOW → Star formation") 
        print("- Pinch MIDDLE → Spiral formation")
        print("- Twist hand → spin carousel")
        print()
        print("🎮 TWO HANDS CONTROLS:")
        print("- Both hands pinch → Cycle formations")
        print("- Spread hands apart → Grid + zoom control")
        print("- Hands horizontal → Line formation")
        print("- Hands vertical → Tower formation")
        print("- Two hands normal → Circle formation")
        print()
        print("✨ FORMATIONS: Spiral, Grid, Circle, Line, Tower, Heart, Star")
        print("- Press 'q' or ESC to exit")
        
        while self.running:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Capture and process camera frame
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror the image
                processed_frame, gesture_data = self.gallery.gesture_detector.process_frame(frame)
                
                # Process gestures
                self.gallery.last_gesture_data = gesture_data  # Store for mode info
                self.gallery.process_gestures(gesture_data)
                
                # Show camera feed (optional, for debugging)
                cv2.imshow('Hand Tracking', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            
            # Update gallery animations
            self.gallery.time = current_time
            self.gallery._update_animations(dt)
            
            # Render 3D gallery
            self.gallery.render(self.width, self.height)
            
            # Draw UI overlay
            self._draw_ui_overlay()
            
            # Display current selection and mode in console
            if self.gallery.image_names and self.gallery.selected_image < len(self.gallery.image_names):
                selected_name = self.gallery.image_names[self.gallery.selected_image]
            else:
                selected_name = "No image selected"
                
            mode_info = self.gallery.get_current_mode_info()
            
            current_info = f"{selected_name} | {mode_info}"
            if hasattr(self, '_last_info') and self._last_info != current_info:
                print(current_info)
            self._last_info = current_info
            
            # Swap buffers
            pygame.display.flip()
            
            # Maintain 60 FPS
            self.clock.tick(60)
        
        # Cleanup
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


def main():
    """Entry point of the application."""
    try:
        app = HolographicGalleryApp()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A working camera")
        print("2. All required packages installed (run: pip install -r requirements.txt)")
        print("3. Images in the gallery/ folder")


if __name__ == "__main__":
    main()