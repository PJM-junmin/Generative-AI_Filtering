from googletrans import Translator
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline #stable Diffusion
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

def get_translation(text, dest_lang):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text

class CFG:
    device = "cuda"
    seed = 20
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 40
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (1280, 720)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 300

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(CFG.image_gen_size)
    return image

def apply_filter(image, filter_type):
    # OpenCV는 RGB가 아닌 BGR로 이미지를 다룸
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if filter_type == "cartoon":
        # 카툰 필터 적용
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    elif filter_type == "outline":
        # OutLine 필터 적용
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Convert back to RGB
        return Image.fromarray(edges)
    elif filter_type == "gray":
        # 흑백 필터 적용
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(gray)
    else:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 사용자로부터 프롬프트 입력받기
user_prompt = input("이미지를 생성할 프롬프트를 입력하세요: ")

# 사용자로부터 파일명 입력받기
file_name = input("저장할 파일명을 입력하세요 (확장자 제외): ")
file_name = file_name.strip() + ".png"

# 사용자로부터 필터 타입 입력받기
print("적용할 필터를 선택하세요: cartoon, outline, gray")
filter_type = input("필터 타입을 입력하세요: ").strip().lower()

# 프롬프트 번역
translation = get_translation(user_prompt, "en")
generated_image = generate_image(translation, image_gen_model)

# 선택된 필터 적용
filtered_image = apply_filter(generated_image, filter_type)

# 현재 작업 디렉토리를 확인하고 출력
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

# 이미지 저장 경로 설정
output_path = os.path.join(current_directory, file_name)
filtered_image.save(output_path)
print(f"Image saved to {output_path}")
