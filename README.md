[README.md](https://github.com/user-attachments/files/23292274/README.md)
# 🧠 OCR arabe basé sur l'IA

Ce projet a été réalisé dans le cadre d’un stage de fin d’études au sein de la startup **Data2Innov**.  
Il s’agit d’un système de reconnaissance optique de caractères (OCR) conçu pour extraire automatiquement du texte en **arabe** à partir d’images.

Le modèle **DocTr** ne prenant pas en charge la langue arabe par défaut, nous avons entrepris de le **fine-tuner** pour cette tâche spécifique.

Le processus de fine-tuning a été divisé en deux étapes principales :

- 📍 **Détection** du texte dans les images  
- 📍 **Reconnaissance** du contenu textuel détecté

Ce projet fournit une base claire et structurée pour adapter et entraîner ces deux modèles, tout en facilitant leur compréhension et leur utilisation.

---

## ⚙️ Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # sous Windows : venv\Scripts\activate

# Cloner le dépôt
git clone https://github.com/rafabdel/rania_stage_ocr_ar.git
cd rania_stage_ocr_ar

# Installer les dépendances
pip install -r requirements.txt
```
## 🔧 Démarrage du Fine-Tuning - Détection

Après l’installation, la première étape consiste à **fine-tuner le modèle de détection**.

Ce modèle nécessite :

- 📁 Un dossier contenant les **images**
- 🗂️ Un fichier `.json` associant à chaque image les **coordonnées des polygones** entourant les mots

> ⚠️ **Important** : DocTr effectue la détection **mot par mot**.  
## 🧾 Génération des données d'entraînement

Pour faciliter la préparation des données dans le format attendu par le modèle, un script nommé `generatorBox` a été développé.  
Ce script vous permet de générer automatiquement les fichiers d'entraînement (`train.json`) et de validation (`val.json`) sans vous soucier du formatage.

### ✅ Prérequis
- Fournir un fichier `.txt` contenant une liste de **phrases ou de mots**,  par ligne.

### ▶️ Lancer le générateur

```bash
# Exécuter le script generatorBox
python codes/generatorBox.py
```
Aprés avoir génerer les données vous pouver lancer le train apartir de la commande 
```bash
venv/bin/python3 doctr/references/detection/train_pytorch.py db_resnet50  --train_path path --val_path path--name OCRD  --epochs 100   --batch_size 8   --input_size 512   --lr 0.0001  --optim adam  --workers 0 --pretrained   --output_dir output  --wb --amp 
```
🧰 Annotation réelle avec CRAFT
Pour des données réelles, nous avons utilisé CRAFT pour générer automatiquement les bounding boxes.
CRAFT permet une labellisation fine et efficace des zones textuelles.
## 🔧 Démarrage du Fine-Tuning - Recognition
🔧 Étape 2 : Fine-tuning - Reconnaissance
🧾 Contraintes à respecter
Une seule image par mot

Taille d’image proche de 128×64 pixels

Un script dédié permet de générer automatiquement ces images à partir d’un simple fichier texte.
### ▶️ Lancer le générateur

```bash
# Exécuter le script genWords
python codes/genWords.py
```
Le script genWords.py prend un fichier .txt contenant une liste de mots, un par ligne, et génère les images dans le format requis.
▶️ Lancer l'entraînement du modèle de reconnaissance
```bash
venv/bin/python3 doctr/references/recognition/train_pytorch.py parseq \
    --train_path arab/train \
    --val_path arab/val \
    --vocab 'multilingual' \
    --epochs 20 \
    --early-stop \
    --batch_size 16 \
    --lr 0.0001 \
    --optim adam \
    --workers 0 \
    --font "font/AThuluth Regular.ttf,font/AeCortoba-wPVz.ttf,font/AgaCordobaV2QrtbBold-W0vV.ttf,font/AgaRasheeqV2Rshyq-pqJy.ttf,font/AligarhArabicFREEPERSONALUSE-Black.otf,font/AligarhArabicFREEPERSONALUSE-Light.otf,font/AligarhArabicFREEPERSONALUSE-Regular.otf,font/AligarhArabicFREEPERSONALUSE-Thin.otf,font/ayman24.ttf,font/Bahij_Myriad_Arabic-Bold.ttf,font/BelalBoldBold-BWrl8.ttf,font/DejaVuSansCondensed-Bold.ttf,font/DejaVuSansCondensed.ttf,font/Hoba-GOwzg.ttf,font/LAXR.otf,font/NAZANIN.TTF,font/TufuliArabicDEMO-Bold.otf,font/TufuliArabicDEMO-Regular.otf" \
    --output_dir output \
    --wb

```

