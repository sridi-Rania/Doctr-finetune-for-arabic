import os
import json
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
import random

TEXT_FILE = "/mnt/shared/rania/rania_stage_ocr_ar/codes/َtash"
FONTS_DIR = "/mnt/shared/rania/rania_stage_ocr_ar/font"
OUTPUT_DIR = "arab/mnt/shared/rania/rania_stage_ocr_ar/arab"
TEXT_COLORS = ["black" ]
BACKGROUND_COLORS = ["white"]
FONT_SIZES = [50 ]
TOP_BOTTOM_MARGIN_RATIO = 0.12
TRAIN_RATIO = 0.8

def get_fonts(font_dir):
    """Retourne une liste de chemins de polices avec leur index"""
    font_files = [
        os.path.join(font_dir, f)
        for f in os.listdir(font_dir)
        if f.lower().endswith(('.ttf', '.otf'))
    ]
    # Trie les polices pour assurer un ordre cohérent
    font_files.sort()
    # Retourne la liste des polices avec leur index
    return [(i, font_path) for i, font_path in enumerate(font_files)]

def make_dirs(base):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base, split, "images"), exist_ok=True)

def draw_text_image(text, font_path, font_size, bg_color, txt_color):
    font = ImageFont.truetype(font_path, font_size)
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_width = max(1, text_width)
    text_height = max(1, text_height)
    margin = int(text_height * TOP_BOTTOM_MARGIN_RATIO)
    img_width = text_width + 2 * margin
    img_height = text_height + 2 * margin
    img_width = max(1, img_width)
    img_height = max(1, img_height)
    image = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(image)
    x = (img_width - text_width) // 2 - bbox[0]
    y = (img_height - text_height) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=txt_color, align="center")
    return image

def load_existing_labels(json_path):
    """Charge les métadonnées JSON existantes si elles existent."""
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Erreur lors de la lecture du fichier JSON {json_path}, création d'un nouveau fichier.")
    return {}

def main():
    # Fixe une graine aléatoire pour garantir la cohérence de la division train/val
    random.seed(42)
    
    fonts = get_fonts(FONTS_DIR)
    if not fonts:
        print("No fonts found in directory:", FONTS_DIR)
        return
        
    with open(TEXT_FILE, encoding="utf-8") as f:
        # Saute les lignes vides
        lines = [line.strip() for line in f if line.strip()]

    # Créer toutes les combinaisons possibles
    all_combinations = []
    for line_idx, text in enumerate(lines):
        for font_idx, font_path in fonts:
            for font_size in FONT_SIZES:
                for bg_color in BACKGROUND_COLORS:
                    for txt_color in TEXT_COLORS:
                        all_combinations.append({
                            "text": text,
                            "font_path": font_path,
                            "font_size": font_size,
                            "bg_color": bg_color,
                            "txt_color": txt_color,
                            "line_idx": line_idx,
                            "font_idx": font_idx
                        })
    
    # Diviser toutes les combinaisons en train/val
    train_combinations, val_combinations = train_test_split(
        all_combinations,
        train_size=TRAIN_RATIO,
        random_state=42
    )
    
    # Vérifie combien de combinaisons par ensemble
    print(f"Nombre total de combinaisons: {len(all_combinations)}")
    print(f"Combinaisons train: {len(train_combinations)} ({len(train_combinations)/len(all_combinations)*100:.1f}%)")
    print(f"Combinaisons val: {len(val_combinations)} ({len(val_combinations)/len(all_combinations)*100:.1f}%)")
    
    make_dirs(OUTPUT_DIR)
    
    # Génère les images pour chaque ensemble
    for split, combinations in [('train', train_combinations), ('val', val_combinations)]:
        images_folder = os.path.join(OUTPUT_DIR, split, "images")
        json_path = os.path.join(OUTPUT_DIR, split, "labels.json")
        
        # Charge les métadonnées existantes
        meta = load_existing_labels(json_path)
        
        # Commence la numérotation à 1
        next_image_number = 1
        
        # Garde une trace des nouvelles images ajoutées
        new_images_count = 0
        
        # Boucle sur toutes les combinaisons
        for info in combinations:
            text = info["text"]
            font_path = info["font_path"]
            font_size = info["font_size"]
            bg_color = info["bg_color"]
            txt_color = info["txt_color"]
            
            # Génère un nom pour l'image en incrémentant le compteur
            # jusqu'à trouver un nom qui n'existe pas
            while True:
                img_name = f"img_{next_image_number}.jpg"
                img_path = os.path.join(images_folder, img_name)
                next_image_number += 1
                
                # Si ce nom de fichier n'existe pas déjà, on peut l'utiliser
                if not os.path.exists(img_path) and img_name not in meta:
                    break
            
            # Génère et sauvegarde l'image
            img = draw_text_image(text, font_path, font_size, bg_color, txt_color)
            img.save(img_path, format="JPEG")
            
            # Ajoute le texte dans les métadonnées
            meta[img_name] = text
            
            # Incrémente le compteur d'images ajoutées
            new_images_count += 1
            
            # Affiche la progression tous les 100 fichiers
            
            
        # Sauvegarde les métadonnées mises à jour
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, ensure_ascii=False, indent=2)
            
        print(f"Added {new_images_count} new images for '{split}' in {images_folder}")
        print(f"Total images for '{split}': {len(meta)}")

if __name__ == "__main__":
    main()