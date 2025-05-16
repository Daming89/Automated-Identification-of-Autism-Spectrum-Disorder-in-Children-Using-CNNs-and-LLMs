# llm_autism_analyzer.py
import torch
import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
import mediapipe as mp
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class LLMAutismAnalyzer:
    AUTISM_FEATURES = {
        "eye_outline": ["smaller pupil", "long distance"],
        "eye_side_left": ["symmetrical shape"],
        "iris_outline_left": ["larger pupil", "irregular shape"],
        "iris_outline_right": ["symmetrical shape"],
        "left_ala_nasi": ["round shape"],
        "lower_mouth": ["red area"],
        "mouth_outline": ["red skin", "symmetrical shape"],
        "nose_outline": ["round shape"],
        "outer_eyelid_left": ["red area"]
    }
    
    NON_AUTISM_FEATURES = {
        "eye_outline": ["larger pupil"],
        "eye_side_right": ["symmetrical shape"],
        "forehead_center_outline": ["symmetrical appearance"],
        "lower_mouth": ["symmetrical shape"],
        "outer_eyelid_left": ["pigmented area"],
        "perioral_left_up": ["symmetrical appearance"],
        "upper_mouth": ["red skin"]
    }

    LANDMARKS = {
    "mouth_outline": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "upper_mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 310, 311, 312, 13, 82, 81, 80, 191, 62, 61],
    "lower_mouth": [62, 95, 88, 178, 87, 14, 317, 402, 318, 308, 375, 321, 405, 314, 17, 84, 181, 91, 146, 62],
    "mandibular_muscle": [83, 18, 313, 406, 418, 262, 369, 400, 377, 152, 148, 176, 140, 32, 194, 182, 83],
    "eye_outline": [156, 70, 63, 105, 66, 107, 193, 417, 336, 296, 334, 293, 300, 383, 372, 340, 346, 347, 348, 349, 350, 357, 465, 245, 128, 121, 120, 119, 118, 117, 111, 143],
    "iris_outline_left": [133, 173, 157, 158, 159, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155],
    "outer_eyelid_left": [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25],
    "iris_outline_right": [362, 398, 384, 385, 386, 387, 388, 263, 249, 390, 373, 374, 380, 381, 382, 362],
    "outer_eyelid_right": [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 254, 253, 252, 256, 341],
    "nose_outline": [168, 193, 188, 217, 198, 115, 98, 97, 2, 326, 327, 344, 420, 437, 412, 417],
    "nasal_bone": [193, 122, 196, 3, 51, 5, 281, 248, 419, 351, 417, 168, 193],
    "nose_tip": [51, 45, 44, 1, 274, 274, 275, 281],
    "left_ala_nasi": [236, 198, 209, 49, 48, 219, 240, 79, 237, 44, 45, 51, 236],
    "right_ala_nasi": [456, 420, 429, 279, 278, 439, 438, 457, 274, 275, 281, 456],
    "forehead_right": [168, 193, 107, 66, 105, 63, 70, 156, 21, 54, 103, 67, 109, 10],
    "forehead_left": [168, 417, 336, 296, 334, 293, 300, 383, 251, 284, 332, 297, 338, 10],
    "forehead_center_outline": [6, 193, 107, 109, 10, 338, 336, 417],
    "eye_side_right": [70, 156, 143, 111, 117, 93, 234, 127, 162, 21, 54],
    "eye_side_left": [284, 300, 383, 372, 340, 346, 323, 454, 356, 389, 251],
    "perioral_right_up": [2, 0, 37, 39, 40, 185, 61, 146, 204, 210, 214, 207, 205, 36, 142, 198, 115, 98, 97],
    "perioral_left_up": [2, 0, 267, 269, 270, 409, 291, 375, 424, 430, 434, 427, 425, 266, 371, 420, 344, 327, 326],
    "chin": [61, 214, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 434, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "cheek_right": [234, 93, 132, 58, 172, 136, 212, 92, 98, 115, 198, 217, 188, 193, 245, 128, 121, 120, 119, 118, 117, 111],
    "cheek_left": [454, 323, 361, 288, 397, 365, 432, 322, 327, 344, 420, 437, 412, 417, 465, 357, 350, 349, 348, 347, 346, 340]
}

    def __init__(self, blip_model_path=None, device=None):
        self.device = device.lower() if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")

        # 初始化MediaPipe面部网格
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # 加载BLIP2模型
        try:
            if blip_model_path and Path(blip_model_path).exists():
                print(f"Loading local model from: {blip_model_path}")
                self.processor = Blip2Processor.from_pretrained(blip_model_path)
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    blip_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                print("Loading default BLIP2 model")
                self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16
                )
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {str(e)}")

    def analyze_image(self, image_path):
        """完整图像分析方法（带重试机制和增强特征提取）"""
        max_retries = 3  # 最大重试次数
        debug_info = []  # 调试信息收集
        
        for attempt in range(max_retries):
            try:
                # ========== 图像加载与预处理 ==========
                image = Image.open(image_path).convert("RGB")
                debug_info.append(f"Attempt {attempt+1}/{max_retries}")
                
                # 增强图像质量
                if min(image.size) < 512:
                    new_size = (max(image.size), max(image.size))
                    image = image.resize(new_size)
                
                # ========== 面部区域分割 ==========
                try:
                    face_parts = self._segment_face(image)
                except Exception as e:
                    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    face_parts = self._segment_face(flipped_image)
                    debug_info.append("Used flipped image")

                # ========== 优化后的特征分析 ==========
                descriptions = {}
                for part_name, part_img in face_parts.items():
                    # 优化后的提示词
                    prompt = (
                        f"Medical description of {part_name.replace('_', ' ')}. "
                        "Focus on: size, color, symmetry. Use medical terms."
                    )
                    
                    inputs = self.processor(
                        images=part_img.resize((224, 224)),
                        text=prompt,
                        return_tensors="pt",
                        max_length=50,  # 限制输入长度
                        truncation=True
                    ).to(self.device)

                    # 动态计算最大新tokens
                    input_length = inputs.input_ids.shape[1]
                    max_new_tokens = min(100, 200 - input_length)  # 假设模型最大长度200
                    
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            num_beams=5,
                            temperature=0.85,
                            top_p=0.95,
                            do_sample=True,  # 启用采样
                            repetition_penalty=1.2
                        )
                    
                    caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                    descriptions[part_name] = caption.lower()
                    debug_info.append(f"{part_name}: {caption[:50]}")

                # ========== 动态评分计算 ==========
                base_score = 30.0
                matched_features = []
                
                # 第一轮：精确匹配
                for part, desc in descriptions.items():
                    clean_desc = re.sub(r'[^a-z ]', '', desc)
                    
                    # 自闭症特征匹配（全词匹配）
                    for feature in self.AUTISM_FEATURES.get(part, []):
                        if re.search(r'\b' + re.escape(feature) + r'\b', clean_desc):
                            base_score += 18
                            matched_features.append(f"A+ {part}:{feature}")
                            break
                    
                    # 非自闭症特征匹配（全词匹配）
                    for feature in self.NON_AUTISM_FEATURES.get(part, []):
                        if re.search(r'\b' + re.escape(feature) + r'\b', clean_desc):
                            base_score -= 15
                            matched_features.append(f"N- {part}:{feature}")
                            break
                
                # 第二轮：部分匹配（提高灵敏度）
                for part, desc in descriptions.items():
                    clean_desc = re.sub(r'\s+', ' ', desc).strip()
                    
                    # 自闭症部分匹配
                    for feature in self.AUTISM_FEATURES.get(part, []):
                        if feature in clean_desc:
                            base_score += 8
                            matched_features.append(f"A± {part}:{feature}")
                            break
                    
                    # 非自闭症部分匹配
                    for feature in self.NON_AUTISM_FEATURES.get(part, []):
                        if feature in clean_desc:
                            base_score -= 6
                            matched_features.append(f"N± {part}:{feature}")
                            break
                
                # 特征数量动态调整
                autism_features = len([f for f in matched_features if f.startswith("A")])
                non_features = len([f for f in matched_features if f.startswith("N")])
                
                if autism_features - non_features > 5:
                    base_score += 20
                elif non_features - autism_features > 3:
                    base_score -= 15
                
                final_score = np.clip(base_score, 0, 100)
                return {
                    "diagnosis": final_score > 50,
                    "score": round(final_score, 1),
                    "evidence": matched_features[:3],
                    "debug": debug_info
                }
                
            except Exception as e:
                debug_info.append(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "diagnosis": False,
                        "score": 0.0,
                        "evidence": ["Analysis failed"],
                        "debug": debug_info
                    }
        

    def _segment_face(self, image):
        """面部区域分割"""
        image_cv = np.array(image)[:, :, ::-1].copy()  # PIL转OpenCV格式
        results = self.face_mesh.process(image_cv)
        
        if not results.multi_face_landmarks:
            raise ValueError("未检测到面部")
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image_cv.shape[:2]
        
        face_parts = {}
        for part_name, indices in self.LANDMARKS.items():
            points = [(int(face_landmarks.landmark[i].x * w),
                      int(face_landmarks.landmark[i].y * h)) for i in indices]
            mask = self._create_mask((h, w), points)
            masked = cv2.bitwise_and(image_cv, image_cv, mask=mask)
            face_parts[part_name] = Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        
        return face_parts

    def _create_mask(self, shape, points):
        """创建面部区域遮罩"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points)], 255)
        return mask

    def _analyze_face_part(self, part_img, part_name):
        """优化后的描述生成方法"""
        try:
            prompt = f"Medical features of {part_name.replace('_', ' ')}"
            
            inputs = self.processor(
                images=part_img.resize((224, 224)),
                text=prompt,
                return_tensors="pt",
                max_length=40,
                truncation=True
            ).to(self.device)

            input_length = inputs.input_ids.shape[1]
            max_new_tokens = min(80, 200 - input_length)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    do_sample=True,
                    temperature=0.7
                )
            
            return self.processor.decode(outputs[0], skip_special_tokens=True).lower()
            
        except Exception as e:
            print(f"特征分析失败: {str(e)}")
            return ""

    def _generate_fallback_description(self, part_img, part_name):
        """生成备用描述"""
        features = [
            "typical anatomical structure",
            "normal pigmentation",
            "symmetrical appearance"
        ]
        random.shuffle(features)
        return f"The {part_name.replace('_', ' ')} displays {features[0]} with {features[1]} and {features[2]}."

    def _calculate_score(self, descriptions):
        """重构评分算法"""
        base_score = 30.0  # 降低初始分数
        matched_features = []
        
        for part, desc in descriptions.items():
            # 清理描述文本
            clean_desc = re.sub(r'[^a-z ]', '', desc)
            
            # 自闭症特征匹配（部分匹配）
            for feature in self.AUTISM_FEATURES.get(part, []):
                if feature in clean_desc:
                    base_score += 15  # 提高权重
                    matched_features.append(f"+15 {part}:{feature}")
                    break  # 每个区域只匹配一个特征
                    
            # 非自闭症特征匹配
            for feature in self.NON_AUTISM_FEATURES.get(part, []):
                if feature in clean_desc:
                    base_score -= 12
                    matched_features.append(f"-12 {part}:{feature}")
                    break
        
        # 添加基于特征数量的动态调整
        feature_count = len(matched_features)
        if feature_count > 5:
            base_score += (feature_count - 5) * 5
        elif feature_count < 2:
            base_score -= 10

        final_score = np.clip(base_score, 0, 100)
        print(f"[DEBUG] 最终评分: {final_score} | 特征匹配: {matched_features}")
        return final_score, matched_features
        

    def evaluate_directory(self, dataset_path, output_dir="results"):
        """完整评估流程（修复标签错误/特征匹配/目录过滤问题）"""
        # ==================== 初始化配置 ==================== 
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 定义合法类别目录（严格匹配大小写）
        VALID_CLASSES = {"Autistic", "Non_Autistic"}
        
        # ==================== 数据校验 ==================== 
        # 检查输入目录结构
        input_dirs = {d.name for d in Path(dataset_path).iterdir() if d.is_dir()}
        missing = VALID_CLASSES - input_dirs
        if missing:
            raise ValueError(f"数据集缺少必要目录: {missing}")

        # ==================== 数据处理流程 ==================== 
        results = []
        
        # 仅处理合法目录（排除输出目录）
        for class_dir in Path(dataset_path).iterdir():
            if not class_dir.is_dir() or class_dir.name not in VALID_CLASSES:
                continue
                
            # 精确标签分配（严格匹配目录名）
            true_label = 1 if class_dir.name == "Autistic" else 0
            print(f"\nProcessing: {class_dir.name} (Label: {true_label})")
            
            # 图像处理进度条
            image_files = list(class_dir.glob("*.[jJ][pP][gG]")) + \
                        list(class_dir.glob("*.[jJ][pP][eE][gG]")) + \
                        list(class_dir.glob("*.[pP][nN][gG]"))
            
            # 带异常处理的图像处理
            for img_path in tqdm(image_files, desc=class_dir.name, unit="img"):
                try:
                    result = self.analyze_image(str(img_path))
                    
                    # 调试输出（可选）
                    if random.random() < 0.05:  # 5%采样率
                        print(f"\n[DEBUG] {img_path.name} 诊断结果:")
                        print(f"Score: {result['score']} | Diagnosis: {result['diagnosis']}")
                        print("关键特征:", result['evidence'][:2])  # 显示前两个证据
                    
                    results.append({
                        "image_path": str(img_path),
                        "true_label": true_label,
                        "pred_label": int(result["diagnosis"]),
                        "score": result["score"]
                    })
                except Exception as e:
                    print(f"\n处理失败 {img_path.name}: {str(e)}")
                    continue

        # ==================== 结果验证 ==================== 
        if not results:
            raise ValueError("没有有效处理结果，可能原因："
                            "1. 图像格式不支持\n"
                            "2. 面部检测失败\n"
                            "3. BLIP2模型未生成有效描述")

        df = pd.DataFrame(results)
        
        # ==================== 报告生成 ==================== 
        # 混淆矩阵（添加zero_division参数）
        plt.figure(figsize=(10,8))
        cm = confusion_matrix(df["true_label"], df["pred_label"])
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            xticklabels=["Non-Autism", "Autism"],
            yticklabels=["Non-Autism", "Autism"],
            annot_kws={"size":14}
        )
        plt.title(f"Confusion Matrix (Accuracy: {np.mean(df['true_label']==df['pred_label']):.2%})", pad=20)
        plt.xlabel("Predicted", labelpad=15)
        plt.ylabel("Actual", labelpad=15)
        plt.savefig(output_path/"confusion_matrix.png", bbox_inches="tight", dpi=300)
        plt.close()

        # 分类报告（抑制警告）
        report = classification_report(
            df["true_label"], 
            df["pred_label"], 
            target_names=["Non-Autism", "Autism"],
            zero_division=0  # 添加此参数消除警告
        )
        
        # 保存文本报告（UTF-8编码）
        with open(output_path/"classification_report.txt", "w", encoding="utf-8") as f:
            f.write("Autism Detection Report\n")
            f.write("="*50 + "\n")
            f.write(f"Total Samples: {len(df)}\n")
            f.write(f"Autism Prevalence: {df['true_label'].mean():.2%}\n\n")
            f.write(report)

        # 分数分布可视化
        plt.figure(figsize=(12,6))
        sns.histplot(
            data=df, 
            x="score", 
            hue="true_label", 
            element="step",
            bins=20, 
            palette=["#2ecc71", "#e74c3c"]
        )
        plt.title("Score Distribution by True Label")
        plt.axvline(50, color='black', linestyle='--', label='Decision Threshold')
        plt.legend(title="True Label", labels=["Non-Autism", "Autism"])
        plt.savefig(output_path/"score_distribution.png", bbox_inches="tight")
        plt.close()

        # 保存原始数据（含预测细节）
        df.to_csv(output_path/"detailed_predictions.csv", index=False, encoding='utf-8-sig')

        print(f"\n评估完成！结果保存至: {output_path.resolve()}")
        return df

    def _save_reports(self, df, output_dir):
        """保存评估报告"""
        # 混淆矩阵
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(df["true_label"], df["pred_label"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Autism", "Autism"],
                    yticklabels=["Non-Autism", "Autism"])
        plt.title(f"Confusion Matrix (Accuracy: {np.mean(df['true_label'] == df['pred_label']):.2%})")
        plt.savefig(Path(output_dir)/"confusion_matrix.png", bbox_inches="tight")
        plt.close()

        # 分类报告
        report = classification_report(df["true_label"], df["pred_label"], target_names=["Non-Autism", "Autism"])
        with open(Path(output_dir)/"classification_report.txt", "w") as f:
            f.write(report)

        # 保存预测结果
        df.to_csv(Path(output_dir)/"predictions.csv", index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autism Detection Analyzer")
    parser.add_argument("--dataset", required=True, help="输入数据集路径")
    parser.add_argument("--output", default="results", help="输出目录")
    parser.add_argument('--blip_model', type=str, default=r"C:\Users\liangming.jiang21\Desktop\models--Salesforce--blip2-opt-2.7b",
                        help='BLIP2模型路径')
    args = parser.parse_args()

    try:
        analyzer = LLMAutismAnalyzer(blip_model_path=args.blip_model)
        analyzer.evaluate_directory(args.dataset, args.output)
        print(f"评估完成，结果保存至: {args.output}")
    except Exception as e:
        print(f"运行失败: {str(e)}")

if __name__ == "__main__":
    main()