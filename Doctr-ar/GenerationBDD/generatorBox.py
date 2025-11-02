import os
import json
import hashlib
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split

PHRASE_FILE = "/mnt/shared/rania/rania_stage_ocr_ar/codes/a.txt"
FONTS_DIR = "/mnt/shared/rania/rania_stage_ocr_ar/font"
OUTPUT_DIR = "/mnt/shared/rania/rania_stage_ocr_ar/detection"
FONT_SIZE = 20
TOP_BOTTOM_MARGIN = 60
SIDE_MARGIN = 60
TRAIN_RATIO = 0.8
LINE_SPACING = 2.0  # Facteur d'espacement entre les lignes 

def get_fonts(font_dir):
    return [
        os.path.join(font_dir, f)
        for f in os.listdir(font_dir)
        if f.lower().endswith(('.ttf', '.otf'))
    ]

def make_dirs(base):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base, split, "images"), exist_ok=True)

def compute_image_hash(img):
    img_bytes = img.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:16]

def calc_line_width(draw, font, text):
    """Retourne la largeur de la ligne en tenant compte des espaces"""
    words = text.split(' ')
    space_width = draw.textbbox((0, 0), " ", font=font)[2]
    word_widths = [draw.textbbox((0, 0), word, font=font)[2] for word in words]
    return sum(word_widths) + space_width * (len(words) - 1)

def calc_multiline_bbox(draw, font, text):
    """Calcule les dimensions d'un texte multi-lignes avec interligne augmenté"""
    lines = text.split('\n')
    widths = [calc_line_width(draw, font, l) for l in lines]
    bbox_sample = draw.textbbox((0, 0), "Ay", font=font)
    font_height = bbox_sample[3] - bbox_sample[1]
    
    # Calculer la hauteur totale avec l'interligne souhaité
    if len(lines) > 1:
        # Premier ligne sans interligne supplémentaire + les autres lignes avec interligne
        total_height = font_height + (len(lines) - 1) * (font_height * LINE_SPACING)
    else:
        total_height = font_height
        
    return max(widths), total_height, font_height

def draw_text_and_get_polygons(draw, font, text, start_y, img_width):
    """
    Dessine chaque mot séparément, retourne les polygones
    """
    words = text.split(' ')
    space_width = draw.textbbox((0, 0), " ", font=font)[2]
    word_widths = [draw.textbbox((0, 0), word, font=font)[2] for word in words]
    total_text_width = sum(word_widths) + space_width * (len(words) - 1)
    x = (img_width - total_text_width) // 2
    y = start_y

    polygons = []
    for word, word_width in zip(words, word_widths):
        # Draw each word at its X
        draw.text((x, y), word, font=font, fill="black")
        bbox = draw.textbbox((x, y), word, font=font)
        poly = [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ]
        polygons.append(poly)
        x += word_width + space_width
    return polygons

def draw_multiline_and_get_polygons(draw, font, text, start_y, img_width):
    """Dessine un texte multi-lignes avec interligne augmenté et retourne les polygones"""
    polygons = []
    lines = text.split('\n')
    y = start_y
    bbox_sample = draw.textbbox((0, 0), "Ay", font=font)
    font_height = bbox_sample[3] - bbox_sample[1]
    
    for line in lines:
        # Centrer chaque ligne horizontalement
        line_polygons = draw_text_and_get_polygons(draw, font, line, y, img_width)
        polygons.extend(line_polygons)
        
        # Passer à la ligne suivante avec l'interligne augmenté
        y += font_height * LINE_SPACING
    
    return polygons

def group_lines(lines, n):
    """Groupe les lignes par n et les joint avec des retours à la ligne"""
    return ['\n'.join(lines[i:i+n]) for i in range(0, len(lines), n) if len(lines[i:i+n]) == n]

def main():
    fonts = get_fonts(FONTS_DIR)
    if not fonts:
        print("No fonts found in directory:", FONTS_DIR)
        return

    with open(PHRASE_FILE, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Grouper les lignes par 3 avec des retours à la ligne
    grouped_lines = group_lines(lines, 3)
    all_samples = []
    for multi_line in grouped_lines:
        for font_path in fonts:
            all_samples.append((multi_line, font_path))

    # Shuffle and split train/val
    indices = list(range(len(all_samples)))
    train_indices, val_indices = train_test_split(indices, train_size=TRAIN_RATIO, random_state=42, shuffle=True)
    splits = {'train': train_indices, 'val': val_indices}
    make_dirs(OUTPUT_DIR)

    img_idx = 1
    label_dicts = {'train': {}, 'val': {}}

    for split, idxs in splits.items():
        images_folder = os.path.join(OUTPUT_DIR, split, "images")
        label_dict = label_dicts[split]
        for idx in idxs:
            multi_line, font_path = all_samples[idx]
            font = ImageFont.truetype(font_path, FONT_SIZE)

            # Dummy draw to determine sizes
            dummy_img = Image.new("RGB", (1, 1))
            draw_dummy = ImageDraw.Draw(dummy_img)
            
            # Calculer les dimensions pour le texte multi-lignes
            line_width, total_text_height, font_height = calc_multiline_bbox(draw_dummy, font, multi_line)
            
            # Convertir en entiers pour éviter l'erreur "float object cannot be interpreted as an integer"
            img_width = int(line_width + 2 * SIDE_MARGIN)
            img_height = int(total_text_height + 2 * TOP_BOTTOM_MARGIN)

            img = Image.new("RGB", (img_width, img_height), color="white")
            draw = ImageDraw.Draw(img)

            # Centrer verticalement le texte multi-lignes
            start_y = (img_height - total_text_height) // 2

            # Dessiner les lignes et obtenir les polygones
            polygons = draw_multiline_and_get_polygons(draw, font, multi_line, start_y, img_width)

            img_name = f"sample_img_{img_idx:05d}.png"
            img_path = os.path.join(images_folder, img_name)
            img.save(img_path)

            img_hash = compute_image_hash(img)
            label_dict[img_name] = {
                "img_dimensions": [img_width, img_height],
                "img_hash": img_hash,
                "polygons": polygons
            }
            img_idx += 1

        # Save the JSON for this split
        json_path = os.path.join(OUTPUT_DIR, split, "labels.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(label_dict, jf, indent=2, ensure_ascii=False)
        print(f"Saved {len(label_dict)} samples for split {split} in {images_folder}")

if __name__ == "__main__":
    main()