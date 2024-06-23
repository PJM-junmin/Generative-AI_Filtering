import random
import numpy as np
import math
from PIL import Image
import cv2

def apply_glitch(image):
    image_np = np.array(image)
    rows, cols, _ = image_np.shape

    # Number of glitch lines
    num_lines = random.randint(20, 80)  # Increase the number of glitch lines for a stronger effect
    for _ in range(num_lines):
        start_row = random.randint(0, rows - 1)
        height = random.randint(5, 40)  # Increase the height range for stronger distortion
        start_col = random.randint(0, cols // 2)
        width = random.randint(cols // 4, cols // 2)
        
        glitch_section = image_np[start_row:start_row + height, start_col:start_col + width].copy()
        shift = random.randint(-50, 50)  # Increase the shift range for more noticeable glitches

        # Calculate the new start and end columns with clipping to avoid boundary issues
        new_start_col = np.clip(start_col + shift, 0, cols - width)
        new_end_col = np.clip(new_start_col + width, 0, cols)

        # Ensure the new section fits within image bounds
        if new_end_col > new_start_col:
            # Apply random opacity to the glitch section
            opacity = random.uniform(0.5, 1.0)
            temp_section = cv2.addWeighted(glitch_section[:, :new_end_col - new_start_col], opacity, image_np[start_row:start_row + height, new_start_col:new_end_col], 1 - opacity, 0)
            image_np[start_row:start_row + height, new_start_col:new_end_col] = temp_section

    # Adding colored rectangles with random opacity
    num_rectangles = random.randint(5, 20)  # Number of rectangles
    for _ in range(num_rectangles):
        start_row = random.randint(0, rows - 50)
        start_col = random.randint(0, cols - 40)
        height = random.randint(20, 100)  # Increase the height for elongated rectangles
        width = random.randint(20, 100)  # Increase the width for elongated rectangles
        color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])  # Red, Green, Blue
        opacity = random.uniform(0.2, 0.7)  # Random opacity

        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[:, :] = color

        # Ensure the overlay fits within image bounds
        end_row = min(start_row + height, rows)
        end_col = min(start_col + width, cols)
        overlay_height = end_row - start_row
        overlay_width = end_col - start_col

        if overlay_height > 0 and overlay_width > 0:
            overlay = overlay[:overlay_height, :overlay_width]
            image_np[start_row:end_row, start_col:end_col] = cv2.addWeighted(image_np[start_row:end_row, start_col:end_col], 1 - opacity, overlay, opacity, 0)

    # Adding gray filter to some sections
    num_gray_sections = random.randint(5, 30)  # Number of gray sections
    for _ in range(num_gray_sections):
        start_row = random.randint(0, rows - 50)
        start_col = random.randint(0, cols - 40)
        height = random.randint(20, 100)
        width = random.randint(20, 100)

        # Ensure the section fits within image bounds
        end_row = min(start_row + height, rows)
        end_col = min(start_col + width, cols)
        section_height = end_row - start_row
        section_width = end_col - start_col

        if section_height > 0 and section_width > 0:
            gray_section = cv2.cvtColor(image_np[start_row:end_row, start_col:end_col], cv2.COLOR_BGR2GRAY)
            gray_section = cv2.cvtColor(gray_section, cv2.COLOR_GRAY2BGR)
            image_np[start_row:end_row, start_col:end_col] = gray_section

    return Image.fromarray(image_np)