import asyncio
import platform
import logging
import traceback
import tempfile
import shutil
import os
import io
import string
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from doctr.io import DocumentFile
from doctr.models.detection.differentiable_binarization.pytorch import db_resnet50
from doctr.models.recognition import parseq
from doctr.models.zoo import ocr_predictor
import wave as wav_lib
from tts_arabic import tts


VOCABS: dict[str, str] = {
    # Arabic & Persian
    "arabic_diacritics": "ًٌٍَُِّْ",
    "arabic_digits": "٠١٢٣٤٥٦٧٨٩",
    "arabic_letters": "- ء آ أ ؤ إ ئ ا ٪ ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك  ٰیٕ٪ ل م ن ه ة و ي پ چ ڢ ڤ گ ﻻ ﻷ ﻹ ﻵ ﺀ ﺁ ﺃ ﺅ ﺇ ﺉ ﺍ ﺏ ﺕ ﺙ ﺝ ﺡ ﺥ ﺩ ﺫ ﺭ ﺯ ﺱ ﺵ ﺹ ﺽ ﻁ ﻅ ﻉ ﻍ ﻑ ﻕ ﻙ ﻝ ﻡ ﻥ ﻩ ﻩ ﻭ ﻱ ﺑ ﺗ ﺛ ﺟ ﺣ ﺧ ﺳ ﺷ ﺻ ﺿ ﻃ ﻇ ﻋ ﻏ ﻓ ﻗ ﻛ ﻟ ﻣ ﻧ ﻫ ﻳ ﺒ ﺘ ﺜ ﺠ ﺤ ﺨ ﺴ ﺸ ﺼ ﺾ ﻄ ﻈ ﻌ ﻐ ﻔ ﻘ ﻜ ﻠ ﻤ ﻨ ﻬ ﻴ ﺎ ﺐ ﺖ ﺚ ﺞ ﺢ ﺦ ﺪ ﺬ ﺮ ﺰ ﺲ ﺶ ﺺ ﺾ ﻂ ﻆ ﻊ ﻎ ﻒ ﻖ ﻚ ﻞ ﻢ ﻦ ﻪ ﺔ ﺓﺋ ﺓﺋ ى ﻼوفرّٕ  ﺊ ﻯ ﻀ ﻯ ﻼ ﺋ ﺊﺓى ﻀال ص ح x ـ ـوx  ﻰ ﻮ ﻲ ً ٌ  ؟ ؛ « » — !  # $ % & ' ( ) * + , - . / : ; < = > ? @ [ ] ^ _  { | } ~",
    "arabic_punctuation": "؟؛«»—",
    "persian_letters": "پچڢڤگ",
    "digits": string.digits,
    "punctuation": string.punctuation,
}
VOCABS["arabic"] = (
    VOCABS["digits"]
    + VOCABS["arabic_digits"]
    + VOCABS["arabic_letters"]
    + VOCABS["persian_letters"]
    + VOCABS["arabic_diacritics"]
    + VOCABS["arabic_punctuation"]
    + VOCABS["punctuation"]
)
arabic = VOCABS["arabic"]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if model files exist
det_model_path = "/mnt/shared/rania/rania_stage_ocr_ar/output/detectionn.pt"
reco_model_path = "/mnt/shared/rania/rania_stage_ocr_ar/output/Rbest.pt"

logger.info(f"Checking for model at: {det_model_path}")
if not os.path.exists(det_model_path):
    error_msg = f"Detection model not found at {det_model_path}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

logger.info(f"Checking for model at: {reco_model_path}")
if not os.path.exists(reco_model_path):
    error_msg = f"Recognition model not found at {reco_model_path}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

# Determine device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    # Load detection model
    logger.info("Loading detection model...")
    det_model = db_resnet50(pretrained=True, pretrained_backbone=True)
    det_model.load_state_dict(torch.load(det_model_path, map_location=device))
    det_model.to(device)
    det_model.eval()
    
    # Load recognition model
    logger.info("Loading recognition model...")
    reco_model = parseq(pretrained=True, pretrained_backbone=True, vocab=arabic)
    state_dict = torch.load(reco_model_path, map_location=device)
    reco_model.load_state_dict(state_dict, strict=False)
    reco_model.to(device)
    reco_model.eval()
    # Create OCR predictor
    logger.info("Creating OCR predictor...")
    predictor = ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
        pretrained=True,
        assume_straight_pages=True,
        straighten_pages=True,
        export_as_straight_boxes=True,
        detect_orientation=True,
        disable_page_orientation=True,
        disable_crop_orientation=True
    )
    logger.info("Models loaded successfully")
    
except Exception as e:
    error_msg = f"Error loading models: {str(e)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    raise RuntimeError(error_msg)

def process_page(page):
    """Process a single page and extract text organized by lines."""
    try:
        # Extract words with their bounding boxes
        words = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geometry = getattr(word, 'geometry', None)
                    value = getattr(word, 'value', '')
                    confidence = getattr(word, 'confidence', 0.0)
                    if geometry is None or not value:
                        logger.warning(f"Skipping invalid word: value='{value}', geometry={geometry}")
                        continue
                    words.append({
                        'geometry': geometry,  # ((x_min, y_min), (x_max, y_max))
                        'value': value,
                        'confidence': confidence
                    })
        
        # Filter valid bounding boxes
        min_width = 0.01
        min_height = 0.02
        filtered_words = []
        for word in words:
            try:
                (x_min, y_min), (x_max, y_max) = word['geometry']
                width = x_max - x_min
                height = y_max - y_min
                if width > min_width and height > min_height:
                    filtered_words.append(word)
                else:
                    logger.debug(f"Filtered out word '{word['value']}' due to size: width={width}, height={height}")
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid geometry for word '{word['value']}': {word['geometry']}, error: {str(e)}")
                continue
        
        # Organize words into lines (group by y-center with tolerance)
        y_tolerance = 0.03
        lines = {}
        for word in filtered_words:
            try:
                (x_min, y_min), (x_max, y_max) = word['geometry']
                y_center = (y_min + y_max) / 2
                
                found_line = False
                for y_key in list(lines.keys()):
                    if abs(y_center - y_key) < y_tolerance:
                        lines[y_key].append(word)
                        found_line = True
                        break
                
                if not found_line:
                    lines[y_center] = [word]
            except (TypeError, ValueError) as e:
                logger.warning(f"Error processing word geometry '{word['geometry']}': {str(e)}")
                continue
        
        # Sort lines top-to-bottom by y_center
        sorted_lines = sorted(lines.items(), key=lambda x: x[0])
        
        # Sort words within each line right-to-left (by x_max for Arabic)
        result_lines = []
        for y_center, line_words in sorted_lines:
            sorted_words = sorted(line_words, key=lambda w: -w['geometry'][1][0])  # Sort by x_max
            line_text = ' '.join([word['value'] for word in sorted_words])
            result_lines.append(line_text)
            logger.debug(f"Line at y={y_center:.4f}: {line_text} (words: {len(sorted_words)})")
        
        if not result_lines:
            logger.warning("No valid text lines extracted")
            return ["No text could be extracted from this document."]
        
        return result_lines
    
    except Exception as e:
        logger.error(f"Error processing page: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing page: {str(e)}")

class TextToSpeechRequest(BaseModel):
    text: str
    speaker: int = 1
    pace: float = 1.0
    denoise: float = 0.005
    volume: float = 0.9
    pitch_mul: float = 1.0
    pitch_add: float = 0.0
    save_to: str = "./test.wav"
    bits_per_sample: int = 32

def generate_audio_wave(
    text: str,
    speaker: int = 0,
    pace: float = 1.0,
    denoise: float = 0.005,
    volume: float = 0.9,
    pitch_mul: float = 1.0,
    pitch_add: float = 0.0,
    save_to: str = "./test.wav",
    bits_per_sample: int = 32
):
    """Generate audio waveform from text using tts_arabic with specified parameters."""
    try:
        wave = tts(
            text=text,
            speaker=speaker,
            pace=pace,
            denoise=denoise,
            volume=volume,
            play=False,
            pitch_mul=pitch_mul,
            pitch_add=pitch_add,
            vowelizer=None,
            model_id='fastpitch',
            vocoder_id='hifigan',
            cuda=None,
            save_to=save_to,
            bits_per_sample=bits_per_sample
        )
        logger.info(f"Generated audio waveform for text (length: {len(text)} chars, speaker: {speaker})")
        return wave
    except Exception as e:
        logger.error(f"Error generating audio waveform: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

def create_audio_response(wave, filename="speech.wav", bits_per_sample=32):
    """Create a StreamingResponse for audio data."""
    audio_bytes = io.BytesIO()
    with wav_lib.open(audio_bytes, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(22050)
        if bits_per_sample == 32:
            scaled_wave = (wave * 2147483647).astype(np.int32)
        elif bits_per_sample == 16:
            scaled_wave = (wave * 32767).astype(np.int16)
        else:
            scaled_wave = (wave * 127).astype(np.int8)
        wav_file.writeframes(scaled_wave.tobytes())
    audio_bytes.seek(0)
    logger.info(f"Created audio response (filename: {filename})")
    return StreamingResponse(
        audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.get("/")
async def read_root():
    return {"message": "OCR API is running", "status": "ok"}

@app.post("/ocr_process")
async def ocr_process(
    file: UploadFile = File(...),
    page_num: int = Query(0, description="Page number to process (0-based index, 0 is first page)"),
    process_all_pages: bool = Query(False, description="Whether to process all pages in a PDF")
):
    temp_file = None
    try:
        logger.info(f"Processing file: {file.filename}, content-type: {file.content_type}")
        logger.info(f"Page requested: {page_num}, Process all pages: {process_all_pages}")
        
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_file = temp.name
            content = await file.read()
            temp.write(content)
        
        logger.info(f"Saved uploaded file to temporary location: {temp_file}")
        
        is_pdf = suffix.lower() == '.pdf' or file.content_type == 'application/pdf'
        
        if is_pdf:
            logger.info("Processing PDF document...")
            doc = DocumentFile.from_pdf(temp_file)
            
            if len(doc) == 0:
                raise HTTPException(status_code=400, detail="PDF has no pages")
            
            logger.info(f"PDF has {len(doc)} pages")
            
            if page_num >= len(doc) and not process_all_pages:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid page number {page_num}. PDF has {len(doc)} pages (0-{len(doc)-1})"
                )
            
            if process_all_pages:
                logger.info("Processing all PDF pages...")
                all_pages_results = []
                for i, page_img in enumerate(doc):
                    logger.info(f"Processing page {i+1}/{len(doc)}...")
                    prediction = predictor([page_img])
                    if prediction.pages and len(prediction.pages) > 0:
                        page_lines = process_page(prediction.pages[0])
                        all_pages_results.append({
                            "page": i,
                            "lines": page_lines
                        })
                
                logger.info(f"Completed processing all {len(doc)} pages")
                return {
                    "file_type": "pdf",
                    "page_count": len(doc),
                    "all_pages": all_pages_results
                }
            else:
                logger.info(f"Processing PDF page {page_num+1}/{len(doc)}...")
                image = doc[page_num]
        else:
            logger.info("Processing image document...")
            image = DocumentFile.from_images(temp_file)[0]
            is_pdf = False
        
        logger.info("Document loaded successfully")
        
        logger.info("Running OCR prediction...")
        prediction = predictor([image])
        logger.info("OCR prediction completed")
        
        if not prediction.pages or len(prediction.pages) == 0:
            raise HTTPException(status_code=400, detail="No text found in the document")
        
        result_lines = process_page(prediction.pages[0])
        logger.info(f"OCR completed. Found {len(result_lines)} lines of text.")
        
        return {
            "lines": result_lines,
            "file_type": "pdf" if is_pdf else "image",
            "page_count": len(doc) if is_pdf else 1,
            "current_page": page_num if is_pdf else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during OCR processing: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Error deleting temporary file {temp_file}: {str(e)}")

@app.post("/text-to-speech/")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        wave = generate_audio_wave(
            text=request.text,
            speaker=request.speaker,
            pace=request.pace,
            denoise=request.denoise,
            volume=request.volume,
            pitch_mul=request.pitch_mul,
            pitch_add=request.pitch_add,
            save_to=request.save_to,
            bits_per_sample=request.bits_per_sample
        )
        return create_audio_response(wave, filename="speech.wav", bits_per_sample=request.bits_per_sample)
    except Exception as e:
        logger.error(f"Erreur lors de la génération vocale: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération vocale: {str(e)}")

@app.post("/download-voice/")
async def download_voice(request: TextToSpeechRequest):
    try:
        wave = generate_audio_wave(
            text=request.text,
            speaker=request.speaker,
            pace=request.pace,
            denoise=request.denoise,
            volume=request.volume,
            pitch_mul=request.pitch_mul,
            pitch_add=request.pitch_add,
            save_to=request.save_to,
            bits_per_sample=request.bits_per_sample
        )
        return create_audio_response(wave, filename="voice.wav", bits_per_sample=request.bits_per_sample)
    except Exception as e:
        logger.error(f"Erreur lors de la génération du fichier vocal: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du fichier vocal: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)