from flask import Flask, request, render_template, redirect, url_for, jsonify
from googletrans import Translator
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline
import cv2
from PIL import Image
import numpy as np
import os
import random
import math
import time

# Initialize Flask app
app = Flask(__name__)

# Ensure 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Define AI configuration class
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    cartoon_model_id = "instruction-tuning-sd/scratch-cartoonizer"

def load_models():
    # Load the image generation model
    image_gen_model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id, torch_dtype=torch.float16 if CFG.device == "cuda" else torch.float32,
        revision="fp16" if CFG.device == "cuda" else None
    )
    image_gen_model = image_gen_model.to(CFG.device)

    # Load the cartoonization model with fp16 support if available, 일단 시도
    try:
        cartoon_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            CFG.cartoon_model_id, torch_dtype=torch.float16 if CFG.device == "cuda" else torch.float32,
            revision="fp16" if CFG.device == "cuda" else None
        ).to(CFG.device)
    except:
        # Fall back to the default loading method if fp16 revision is not supported, 없으면 이걸로
        cartoon_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            CFG.cartoon_model_id, torch_dtype=torch.float16 if CFG.device == "cuda" else torch.float32
        ).to(CFG.device)

    return image_gen_model, cartoon_model

image_gen_model, cartoon_model = load_models()

# Translation function, 영어로 Prompt 번역
def get_translation(text, dest_lang="en"):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text

# Image generation function, 이미지 생성
def generate_image(prompt, model, seed, steps, size, guidance_scale):
    generator = torch.Generator(device=CFG.device).manual_seed(seed)
    image = model(
        prompt, num_inference_steps=steps,
        generator=generator,
        guidance_scale=guidance_scale
    ).images[0]
    
    image = image.resize(size)
    return image

# Glitch Filter
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



# Dissolve Filter
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
        max_shift = random.randint(5, 60)  # Increase max_shift for stronger effect
        for y in range(rows):
            for x in range(cols):
                angle = math.atan2(y - center_y, x - center_x)
                radius = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                shift_x = int(max_shift * math.cos(angle + radius / 20.0))
                shift_y = int(max_shift * math.sin(angle + radius / 20.0))
                new_x, new_y = min(max(x + shift_x, 0), cols - 1), min(max(y + shift_y, 0), rows - 1)
                image_np[y, x] = image_np[new_y, new_x]

    return Image.fromarray(image_np)


# Filter application function, 필터 적용
def apply_filter(image, filter_type, file_name):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    filtered_file_name = f"{os.path.splitext(file_name)[0]}_{filter_type}{os.path.splitext(file_name)[1]}"
    
    if filter_type == "cartoon":
        if os.path.exists(f'static/{filtered_file_name}'):
            return Image.open(f'static/{filtered_file_name}')
        else:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cartoon_image = cartoon_model("Cartoonize the following image", image=image_pil).images[0]
            cartoon_image.save(f'static/{filtered_file_name}')
            return cartoon_image
    
    elif filter_type == "outline":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        Image.fromarray(edges).save(f'static/{filtered_file_name}')
        return Image.fromarray(edges)
    
    elif filter_type == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Image.fromarray(gray).save(f'static/{filtered_file_name}')
        return Image.fromarray(gray)
    
    elif filter_type == "glitch":
        filtered_image = apply_glitch(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        filtered_image.save(f'static/{filtered_file_name}')
        return filtered_image
    
    elif filter_type == "dissolve":
        filtered_image = apply_dissolve(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        filtered_image.save(f'static/{filtered_file_name}')
        return filtered_image
    
    elif filter_type == "none":
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(f'static/{filtered_file_name}')
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    else:
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(f'static/{filtered_file_name}')
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# Route for the main page, 메인 화면
@app.route("/")
def index():
    return render_template("text_to_image.html")

# Route for generating the image and displaying the new page
@app.route("/generate", methods=["POST"]) # Post해서, 값을 보내는 방식, Generated Image button 클릭시 작동
def generate():
    user_prompt = request.form.get("text")
    file_name = request.form.get("file_name") + ".png"
    steps = int(request.form.get("steps"))
    size = request.form.get("size").split('x')
    size = (int(size[0]), int(size[1]))
    guidance_scale = float(request.form.get("guidance_scale"))
    seed = int(request.form.get("seed"))
    
    # Translate the prompt
    translation = get_translation(user_prompt, "en")
    
    # Generate the image
    generated_image = generate_image(translation, image_gen_model, seed, steps, size, guidance_scale)
    
    # Save the generated image to a temporary file
    file_path = os.path.join('static', file_name)
    generated_image.save(file_path)
    
    return redirect(url_for('set_filter_image', file_name=file_name, prompt=user_prompt))



# Route for setting filter on the generated image, GET하고, Post해서 보내는 방식, Filter 적용시 작동
@app.route("/set_filter_image", methods=["GET", "POST"])
def set_filter_image():
    file_name = request.args.get("file_name")
    prompt = request.args.get("prompt")
    filtered_file_name = None
    
    if request.method == "POST":
        filter_type = request.form.get("filter_type")
        image = Image.open(os.path.join('static', file_name))
        filtered_image = apply_filter(image, filter_type, file_name)
        filtered_file_name = f"{os.path.splitext(file_name)[0]}_{filter_type}{os.path.splitext(file_name)[1]}"
        return render_template("set_filter_image.html", file_name=file_name, filtered_file_name=filtered_file_name, prompt=prompt)
    
    return render_template("set_filter_image.html", file_name=file_name, filtered_file_name=filtered_file_name, prompt=prompt)

if __name__ == "__main__":
    app.run(debug=True)
