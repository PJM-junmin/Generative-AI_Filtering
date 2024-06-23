import math
import random

def apply_dissolve(image):
    image_np = np.array(image)
    rows, cols, _ = image_np.shape

    effect = random.choice(['W', 'S', 'spiral'])

    if effect == 'W':
        amplitude = random.randint(50, 100)  # Significantly increase the amplitude range for more distortion
        for i in range(0, rows, 10):
            shift = int(amplitude * math.sin(2.0 * math.pi * i / rows))
            if shift < 0:
                shift = max(shift, -cols)  # Ensure shift does not exceed image bounds
            else:
                shift = min(shift, cols)
            image_np[i:i+10, :] = np.roll(image_np[i:i+10, :], shift, axis=1)

    elif effect == 'S':
        amplitude = random.randint(50, 100)  # Significantly increase the amplitude range for more distortion
        for i in range(0, cols, 10):
            shift = int(amplitude * math.sin(2.0 * math.pi * i / cols))
            if shift < 0:
                shift = max(shift, -rows)  # Ensure shift does not exceed image bounds
            else:
                shift = min(shift, rows)
            image_np[:, i:i+10] = np.roll(image_np[:, i:i+10], shift, axis=0)

    elif effect == 'spiral':
        center_x, center_y = cols // 2, rows // 2
        max_shift = random.randint(30, 60)  # Increase max_shift for stronger effect
        for y in range(rows):
            for x in range(cols):
                angle = math.atan2(y - center_y, x - center_x)
                radius = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                shift_x = int(max_shift * math.cos(angle + radius / 20.0))
                shift_y = int(max_shift * math.sin(angle + radius / 20.0))
                new_x, new_y = min(max(x + shift_x, 0), cols - 1), min(max(y + shift_y, 0), rows - 1)
                image_np[y, x] = image_np[new_y, new_x]

    return Image.fromarray(image_np)