import os
import re
import torch
import pandas as pd
import logging
import threading
from PIL import Image, UnidentifiedImageError
from glob import glob
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import login
from transformers import Blip2Processor, Blip2ForConditionalGeneration

#Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(r"C:\Users\liangming.jiang21\FYP\processing.log"),
        logging.StreamHandler()
    ]
)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

class Config:
    INPUT_DIR = Path(r"C:\Users\liangming.jiang21\FYP\outputs-non-autism")
    OUTPUT_DIR = Path(r"C:\Users\liangming.jiang21\FYP\analysis_results\Autism")
    MODEL_LOCAL_PATH = r"C:\Users\liangming.jiang21\Desktop\models--Salesforce--blip2-opt-2.7b"
    MODEL_NAME = "Salesforce/blip2-opt-2.7b"
    IMAGE_SIZE = 384
    MAX_WORKERS = min(os.cpu_count(), 4)
    IMAGE_EXTS = ["png", "jpg", "jpeg", "bmp", "tiff"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Medical feature configuration
MEDICAL_FEATURES = {
    "mouth_outline": {
        "attributes": ["color", "shape", "symmetry", "vermilion_border", "lips_seal"]
    },
    "upper_mouth": {
        "attributes": ["philtrum_shape", "cupids_bow", "mucosal_color", "vertical_proportion", "surface_texture"]
    },
    "lower_mouth": {
        "attributes": ["labiomental_angle", "vermilion_thickness", "horizontal_contour", "skin_texture", "dynamic_folds"]
    },
    "mandibular_muscle": {
        "attributes": ["muscle_definition", "symmetry", "skin_coverage", "jawline_contour", "static_tension"]
    },
    "eye_outline": {
        "attributes": ["palpebral_fissure", "canthal_angles", "lid_margin", "eyelash_density", "lateral_contour"]
    },
    "iris_outline_left": {
        "attributes": ["color", "pupil_size", "limbal_ring", "crypt_pattern", "collarette_detail"]
    },
    "outer_eyelid_left": {
        "attributes": ["skin_texture", "lid_crease", "vascularity", "lash_orientation", "lateral_fold"]
    },
    "iris_outline_right": {
        "attributes": ["color", "pupil_size", "limbal_ring", "crypt_pattern", "collarette_detail"]
    },
    "outer_eyelid_right": {
        "attributes": ["skin_texture", "lid_crease", "vascularity", "lash_orientation", "lateral_fold"]
    },
    "nose_outline": {
        "attributes": ["dorsal_shape", "tip_definition", "ala_position", "columella_show", "nasolabial_angle"]
    },
    "nasal_bone": {
        "attributes": ["bridge_straightness", "root_depth", "osteocartilaginous_junction", "sidewall_contour", "midline_alignment"]
    },
    "nose_tip": {
        "attributes": ["lobule_shape", "alar_columellar_relation", "tip_rotation", "skin_thickness", "supratip_break"]
    },
    "left_ala_nasi": {
        "attributes": ["ala_shape", "nostril_show", "skin_texture", "sidewall_contour", "dynamic_mobility"]
    },
    "right_ala_nasi": {
        "attributes": ["ala_shape", "nostril_show", "skin_texture", "sidewall_contour", "dynamic_mobility"]
    },
    "forehead_right": {
        "attributes": ["skin_texture", "horizontal_rhytids", "vascular_pattern", "hairline_contour", "bony_prominence"]
    },
    "forehead_left": {
        "attributes": ["skin_texture", "horizontal_rhytids", "vascular_pattern", "hairline_contour", "bony_prominence"]
    },
    "forehead_center_outline": {
        "attributes": ["glabellar_shape", "vertical_furrows", "skin_tone", "midline_contour", "supraorbital_ridge"]
    },
    "eye_side_right": {
        "attributes": ["lateral_canthus", "crow_feet", "temporal_hollowing", "vascular_visibility", "skin_laxity"]
    },
    "eye_side_left": {
        "attributes": ["lateral_canthus", "crow_feet", "temporal_hollowing", "vascular_visibility", "skin_laxity"]
    },
    "perioral_right_up": {
        "attributes": ["nasolabial_fold", "cheek_contour", "skin_texture", "dynamic_lines", "vascular_pattern"]
    },
    "perioral_left_up": {
        "attributes": ["nasolabial_fold", "cheek_contour", "skin_texture", "dynamic_lines", "vascular_pattern"]
    },
    "chin": {
        "attributes": ["mental_protrusion", "labiomental_sulcus", "skin_texture", "horizontal_creases", "jawline_transition"]
    },
    "cheek_right": {
        "attributes": ["malar_eminence", "skin_turgor", "vascular_pattern", "hairline_transition", "submalar_hollow"]
    },
    "cheek_left": {
        "attributes": ["malar_eminence", "skin_turgor", "vascular_pattern", "hairline_transition", "submalar_hollow"]
    }
}


PROMPT_TEMPLATES = {
    "mouth_outline": "Describe the lip contour including color, shape symmetry, vermilion border definition and lip seal competence.",
    "upper_mouth": "Describe the upper lip features including philtrum shape, cupid's bow definition, mucosal color and vertical proportion.",
    "lower_mouth": "Describe the lower lip characteristics including labiomental angle, vermilion thickness, horizontal contour and skin texture.",
    "mandibular_muscle": "Describe the mandibular region including muscle definition, jawline contour and skin tension patterns.",
    "eye_outline": "Describe the eye shape including palpebral fissure dimensions, canthal angles and eyelid margin characteristics.",
    "iris_outline_left": "Describe the left iris characteristics including color, pupil size and limbal ring definition.",
    "outer_eyelid_left": "Describe the left eyelid features including skin texture, crease depth and lash orientation.",
    "iris_outline_right": "Describe the right iris characteristics including color, pupil size and limbal ring definition.",
    "outer_eyelid_right": "Describe the right eyelid features including skin texture, crease depth and lash orientation.",
    "nose_outline": "Describe the nasal shape including dorsal contour, tip definition and alar positioning.",
    "nasal_bone": "Describe the nasal bone structure including bridge straightness and osteocartilaginous junction.",
    "nose_tip": "Describe the nasal tip features including lobule shape, rotation angle and skin thickness.",
    "left_ala_nasi": "Describe the left nasal ala including shape, nostril show and dynamic mobility.",
    "right_ala_nasi": "Describe the right nasal ala including shape, nostril show and dynamic mobility.",
    "forehead_right": "Describe the right forehead characteristics including skin texture and bony prominence.",
    "forehead_left": "Describe the left forehead characteristics including skin texture and bony prominence.",
    "forehead_center_outline": "Describe the central forehead features including glabellar contour and midline shape.",
    "eye_side_right": "Describe the right periocular area including lateral canthus position and temporal contour.",
    "eye_side_left": "Describe the left periocular area including lateral canthus position and temporal contour.",
    "perioral_right_up": "Describe the right perioral region including nasolabial fold depth and cheek contour.",
    "perioral_left_up": "Describe the left perioral region including nasolabial fold depth and cheek contour.",
    "chin": "Describe the chin morphology including protrusion degree and labiomental fold definition.",
    "cheek_right": "Describe the right cheek features including malar prominence and skin turgor.",
    "cheek_left": "Describe the left cheek features including malar prominence and skin turgor.",
    "default": "Describe the visible features of this facial region including color, shape and texture characteristics."
}

import torch
from huggingface_hub import login
try:
    login(token='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')
except Exception as e:
    logging.warning(f"Hugging Face login fail: {str(e)}")

class BLIPClinicalAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if the local model file exists
        if os.path.exists(Config.MODEL_LOCAL_PATH):
            logging.info(f"Load the model from local: {Config.MODEL_LOCAL_PATH}")
            self.processor = Blip2Processor.from_pretrained(
                Config.MODEL_LOCAL_PATH,
                local_files_only=True,
                torch_dtype=torch.float16
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                Config.MODEL_LOCAL_PATH,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            logging.warning("The local model does not exist and will be downloaded from the network")
            self.processor = Blip2Processor.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.generation_config = {
            "max_new_tokens": 400,
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": self.processor.tokenizer.eos_token_id
        }

    def analyze_image(self, image_path, prompt):
        try:
            raw_image = Image.open(image_path).convert("RGB")
            file_stem = Path(image_path).stem
            part_name = '_'.join(file_stem.split('_')[1:])
            if part_name not in PROMPT_TEMPLATES:
                import logging
                logging.error(f"Invalid part name {part_name} in file: {file_stem}")
                return ""
            # Adjust prompt word format
            # Structure for BLIP2: Questions: Answers:
            structured_prompt = (
            f"Question: Analyze the {part_name.replace('_', ' ')} for clinical features. "
            f"Focus on: {PROMPT_TEMPLATES[part_name]}. "
            "Provide detailed observations. Answer:"
        )

            inputs = self.processor(
            raw_image,
            text=structured_prompt,
            return_tensors="pt"
        ).to(self.device, torch.float16)

            # Add attention_mask processing
            inputs["attention_mask"] = inputs["input_ids"].ne(self.processor.tokenizer.pad_token_id)

            outputs = self.model.generate(
            **inputs,
            **self.generation_config
        )

            # Post-processing generates results
            full_output = self.processor.decode(outputs[0], skip_special_tokens=True)
            # Extract the valid answer part
            return full_output.split("Answer:")[-1].strip()
        except KeyError as ke:
            logging.error(f"Prompt template missing for {part_name}")
            return ""
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}", exc_info=True)
            return ""

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = BLIPClinicalAnalyzer()
        return cls._instance

def process_image(args):
    """修复后的处理函数"""
    image_path, part_name = args
    try:
        image_path, part_dir = args
        part_name = Path(part_dir).name
        
        # 双重验证文件名格式
        if not re.match(r"^\d{4}_" + re.escape(part_name) + r"$", Path(image_path).stem):
            logging.error(f"文件名格式不匹配: {image_path}")
            return None

        # Generate basic prompt words
        base_prompt = PROMPT_TEMPLATES[part_name]
        # Build structured prompts
        structured_prompt = (
            f"Please describe the {part_name.replace('_', ' ')} in this image. "
            f"Focus on: {base_prompt} "
            "Provide a concise description including color, shape and texture features."
        )

        analyzer = BLIPClinicalAnalyzer.get_instance()
        description = analyzer.analyze_image(image_path, structured_prompt)

        # Filter duplicate content
        if description.startswith(base_prompt):
            description = description[len(base_prompt):].strip()

        return {
            "image_id": Path(image_path).stem,
            "part": part_name,
            "raw_description": description,
            **extract_clinical_features(description, part_name)
        }
    except Exception as e:
        logging.error(f"Fail: {image_path} - {str(e)}")
        return None

def extract_clinical_features(text, part_name):
    features = {
        "measurements": [],
        "colors": [],
        "symmetry_issues": [],
        "texture_terms": [],
        "attributes": []  # 改为新的属性字段
    }

    try:
        # 获取当前部位的配置（不再使用default）
        config = MEDICAL_FEATURES.get(part_name, {})
        
        # 测量值提取保持不变
        measurements = re.findall(r"(\d+\.?\d*)\s*(mm|cm|degrees|ml|kg|%)", text, re.IGNORECASE)
        features["measurements"] = [f"{num} {unit}" for num, unit in measurements]

        # 颜色特征
        color_terms = ["red", "pale", "cyanotic", "erythematous", "pigmented",
                      "jaundiced", "vitiligo", "hypopigmented", "hyperpigmented"]
        features["colors"] = [term for term in color_terms if term in text.lower()]

        # 对称性分析
        symmetry_terms = ["asymmetric", "uneven", "disproportionate", "misaligned",
                         "lopsided", "unbalanced", "lateral deviation"]
        features["symmetry_issues"] = [term for term in symmetry_terms if term in text.lower()]

        # 纹理特征
        texture_terms = ["smooth", "wrinkled", "thickened", "scarred", "atrophic",
                        "pebbled", "lichenified", "excoriated", "sclerotic"]
        features["texture_terms"] = [term for term in texture_terms if term in text.lower()]

        # 属性匹配（使用新的attributes配置）
        attributes = config.get("attributes", [])
        features["attributes"] = [
            attr for attr in attributes 
            if re.search(r"\b" + re.escape(attr) + r"\b", text.lower())
        ]

    except Exception as e:
        logging.error(f"Feature extraction error: {str(e)}", exc_info=True)
    
    return features

def analyze_part(part_dir):
    part_name = Path(part_dir).name
    image_exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    image_paths = []

    for ext in image_exts:
        image_paths.extend(glob(os.path.join(part_dir, ext)))

    if not image_paths:
        logging.warning(f" {part_name} has no valid images")
        return

    logging.info(f"Start analyzing the part with {part_name} ({len(image_paths)} images)")
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_image, (path, part_name)): path
                  for path in image_paths}

        for future in tqdm(as_completed(futures), total=len(futures),
                         desc=f"Dealing {part_name}"):
            result = future.result()
            if result:
                results.append(result)

    if not results:
        logging.warning(f" {part_name} has no valid results")
        return

    # Generate detailed report
    report = {
    "metadata": {
        "part_name": part_name,
        "image_count": len(results),
        "total_measurements": sum(len(r["measurements"]) for r in results),
        # 修改临床特征统计为属性特征
        "attributes": Counter([attr for r in results for attr in r["attributes"]]) 
    },
    "data": results
}

    # Save structured report
    save_analysis_report(report, part_name)
    logging.info(f"complete analysis for {part_name}")

def save_analysis_report(report, part_name):
    output_dir = Path(r"C:\Users\liangming.jiang21\FYP\analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Creating an Excel Writer
    output_path = output_dir / f"{part_name}_analysis.xlsx"

    with pd.ExcelWriter(output_path) as writer:
        # Original description worksheet
        pd.DataFrame([{"Image ID": r["image_id"], "Description": r["raw_description"]}
                     for r in report["data"]]).to_excel(
            writer, sheet_name="原始描述", index=False
        )

       

    logging.info(f"report saved to {output_path}")

def main():
    required_parts = {
        "cheek_left", "cheek_right", "chin", "eye_outline",
        "eye_side_left", "eye_side_right", "forehead_center_outline",
        "forehead_left", "forehead_right", "iris_outline_left",
        "iris_outline_right", "left_ala_nasi", "lower_mouth",
        "mandibular_muscle", "mouth_outline", "nasal_bone",
        "nose_outline", "nose_tip", "outer_eyelid_left",
        "outer_eyelid_right", "perioral_left_up", "perioral_right_up",
        "right_ala_nasi", "upper_mouth"
    }

    missing_medical = required_parts - set(MEDICAL_FEATURES.keys())
    missing_prompts = required_parts - set(PROMPT_TEMPLATES.keys())
    
    if missing_medical:
        logging.error(f"Missing MEDICAL_FEATURES for: {', '.join(missing_medical)}")
    if missing_prompts:
        logging.error(f"Missing PROMPT_TEMPLATES for: {', '.join(missing_prompts)}")
    if missing_medical or missing_prompts:
        return
    input_base = Path(r"C:\Users\liangming.jiang21\FYP\outputs-non-autism")
    part_dirs = [d for d in input_base.iterdir() if d.is_dir()]

    if not part_dirs:
        logging.error("No valid part directory found")
        return

    # Initializing a model instance
    analyzer = BLIPClinicalAnalyzer(Config.MODEL_NAME)
    BLIPClinicalAnalyzer._instance = analyzer

    # Use thread pool to handle all parts
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(analyze_part, str(part_dir)): part_dir
                  for part_dir in part_dirs}

        for future in tqdm(as_completed(futures), total=len(futures),
                         desc="Overall progress"):
            future.result()

    logging.info("All analysis tasks completed")

if __name__ == "__main__":
    # Windows Multi-Process Protection
    if os.name == "nt":
        import threading
        torch.multiprocessing.set_start_method('spawn', force=True)

    main()
