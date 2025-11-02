import os
import json
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random

input_dir = r"/home/chaima/ocr_arabic/number/val/images"
output_dir = r"/home/chaima/ocr_arabic/number/val/images"
labels_path = r"/home/chaima/ocr_arabic/number/val/labels.json"

with open(labels_path, 'r', encoding='utf-8') as f:
    labels = json.load(f)
 
def add_gaussian_noise(image, mean=0, std=10):
    img_array = np.array(image)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.uint8)
    noisy_img = np.clip(img_array + noise, 0, 255)
    return Image.fromarray(noisy_img)

def apply_gaussian_blur(image, radius=1):
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def adjust_brightness_contrast(image, brightness_factor=1.1, contrast_factor=1.1):
    img_array = np.array(image).astype(np.float32)
    img_array = img_array * brightness_factor
    img_array = (img_array - 128) * contrast_factor + 128
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def invert_colors(image):
    return Image.eval(image, lambda x: 255 - x)

def apply_shear(image, shear_factor=0.1):
    img_array = np.array(image)
    shear_matrix = (1, shear_factor, 0, 0, 1, 0)
    return Image.fromarray(img_array).transform(
        image.size, Image.AFFINE, shear_matrix, resample=Image.BICUBIC, fillcolor="white"
    )

def apply_scale(image, scale_factor=1.2):
    original_size = image.size
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    scaled_img = image.resize(new_size, resample=Image.BICUBIC)
    if scale_factor > 1:
        left = (new_size[0] - original_size[0]) // 2
        top = (new_size[1] - original_size[1]) // 2
        scaled_img = scaled_img.crop((left, top, left + original_size[0], top + original_size[1]))
    else:
        new_img = Image.new("RGB", original_size, "white")
        left = (original_size[0] - new_size[0]) // 2
        top = (original_size[1] - new_size[1]) // 2
        new_img.paste(scaled_img, (left, top))
        scaled_img = new_img
    return scaled_img

def apply_translation(image, dx=10, dy=10):
    return image.transform(
        image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BICUBIC, fillcolor="white"
    )

def add_salt_pepper_noise(image, prob=0.01):
    img_array = np.array(image)
    noise = np.random.random(img_array.shape[:2])
    img_array[noise < prob / 2] = 0
    img_array[noise > 1 - prob / 2] = 255
    return Image.fromarray(img_array)

def random_crop_and_pad(image, crop_ratio=0.8):
    original_size = image.size
    crop_width = int(original_size[0] * crop_ratio)
    crop_height = int(original_size[1] * crop_ratio)
    left = random.randint(0, original_size[0] - crop_width)
    top = random.randint(0, original_size[1] - crop_height)
    cropped_img = image.crop((left, top, left + crop_width, top + crop_height))
    new_img = Image.new("RGB", original_size, "white")
    new_img.paste(cropped_img.resize((crop_width, crop_height), Image.BICUBIC), (left, top))
    return new_img

def adjust_color_jitter(image, hue_factor=0.1):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1 + random.uniform(-hue_factor, hue_factor))

def apply_dropout(image, prob=0.05):
    img_array = np.array(image)
    mask = np.random.random(img_array.shape[:2]) < prob
    img_array[mask] = 255
    return Image.fromarray(img_array)

transformations = [
 #  ("rotate_5", lambda img: img.rotate(5, resample=Image.BICUBIC, fillcolor="white")),
    ("rotate_-5", lambda img: img.rotate(-5, resample=Image.BICUBIC, fillcolor="white")),
  # ("rotate_10", lambda img: img.rotate(10, resample=Image.BICUBIC, fillcolor="white")),
    ("blur", lambda img: apply_gaussian_blur(img, radius=1)),
   #("invert", lambda img: invert_colors(img)),
   # ("shear_0.1", lambda img: apply_shear(img, shear_factor=0.1)),
    #("shear_-0.1", lambda img: apply_shear(img, shear_factor=-0.1)),
    #("scale_up", lambda img: apply_scale(img, scale_factor=1.2)),
    #("scale_down", lambda img: apply_scale(img, scale_factor=0.8)),
    #("salt_pepper", lambda img: add_salt_pepper_noise(img, prob=0.01)),
   ("crop_pad", lambda img: random_crop_and_pad(img, crop_ratio=0.6)),
    ("color_jitter", lambda img: adjust_color_jitter(img, hue_factor=0.1)),
    ("dropout", lambda img: apply_dropout(img, prob=0.05)),
]

for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    base_name, ext = os.path.splitext(img_name)
    if not base_name.isdigit():
        continue  

    img_path = os.path.join(input_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    for transform_name, transform_func in transformations:
        transformed_img = transform_func(image)
        new_img_name = f"{base_name}_{transform_name}{ext}"
        new_img_path = os.path.join(output_dir, new_img_name)
        transformed_img.save(new_img_path)
        if img_name in labels:
            labels[new_img_name] = labels[img_name]

with open(labels_path, 'w', encoding='utf-8') as f:
    json.dump(labels, f, ensure_ascii=False, indent=4)

print(f"Augmentation terminée. Nouvelles images ajoutées dans {output_dir}")
