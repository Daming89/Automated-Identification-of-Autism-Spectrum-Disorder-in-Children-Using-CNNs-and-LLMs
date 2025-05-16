import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
from pathlib import Path
import logging

# 定义路径
AUTISM_PATH = r"C:\Users\liangming.jiang21\FYP\analysis_results\Autism_1"
NON_AUTISM_PATH = r"C:\Users\liangming.jiang21\FYP\analysis_results\Non_autism_1"
AUTISM_IMAGES = r"C:\Users\liangming.jiang21\FYP\outputs-autism"
NON_AUTISM_IMAGES = r"C:\Users\liangming.jiang21\FYP\outputs-non-autism"
# Autistic traits map
AUTISM_FEATURES = {
    "eye outline": ["smaller pupil"],
    "eye side left": ["symmetrical shape"],
    "iris outline left": ["larger pupil", "irregular shape", "normal pupil"],
    "iris outline right": ["symmetrical shape", "larger pupil"],
    "left Ala Nasi": ["round shape"],
    "Lower Mouth": ["red area"],
    "mouth outline": ["red skin", "symmetrical shape", "red area"],
    "nose outline": ["round shape"],
    "outer eyelid left": ["red area"]
}

# Non-autistic traits map
NON_AUTISM_FEATURES = {
    "eye outline": ["larger pupil"],
    "eye side right": ["symmetrical shape"],
    "forehead center outline": ["symmetrical appearance"],
    "Lower Mouth": ["symmetrical shape"],
    "outer eyelid left": ["pigmented area", "symmetrical shape"],
    "perioral left up": ["symmetrical appearance"],
    "upper mouth": ["red skin"]
}


class FacialPartDataset(Dataset):

    def __init__(self, autism_path, non_autism_path, autism_img_path, non_autism_img_path, processor):
        self.processor = processor
        self.samples = []
        
        # 初始化日志
        self._init_logging()
        
        # 加载两类数据
        self._load_dataset_group(
            data_path=autism_path,
            img_base=autism_img_path,
            label=1,
            dataset_type="Autism"
        )
        self._load_dataset_group(
            data_path=non_autism_path,
            img_base=non_autism_img_path,
            label=0,
            dataset_type="Non-autism"
        )
        
        self._final_report()

    def _init_logging(self):
        """初始化调试日志"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("\n" + "="*60)
        self.logger.info("Initializing FacialPartDataset")

    def _load_dataset_group(self, data_path, img_base, label, dataset_type):
        """加载单个数据集组（Autism/Non-autism）"""
        self.logger.info(f"\n{'='*30} Loading {dataset_type} Data {'='*30}")
        
        excel_files = list(Path(data_path).glob("*_analysis.xlsx"))
        self.logger.info(f"Found {len(excel_files)} analysis files")
        
        for excel_file in excel_files:
            self._process_excel_file(excel_file, img_base, label, dataset_type)

    def _process_excel_file(self, excel_file, img_base, label, dataset_type):
        """处理单个Excel分析文件"""
        try:
            part_name = excel_file.stem.replace("_analysis", "")
            self.logger.info(f"\nProcessing part: {part_name} ({dataset_type})")
            
            # 读取并验证Excel
            df = self._read_and_validate_excel(excel_file)
            if df is None:
                return
                
            # 构建图像目录
            img_dir = Path(img_base) / part_name
            if not self._validate_image_dir(img_dir):
                return
                
            # 处理数据行
            valid_count = 0
            for idx, row in df.iterrows():
                if self._process_data_row(row, img_dir, part_name, label):
                    valid_count += 1
                    
            self.logger.info(f"Valid samples: {valid_count}/{len(df)}")

        except Exception as e:
            self.logger.error(f"Error processing {excel_file.name}: {str(e)}", exc_info=True)

    def _read_and_validate_excel(self, excel_file):
        """读取并验证Excel文件"""
        try:
            df = pd.read_excel(excel_file, engine='openpyxl')
            required_cols = {'Image ID', 'Description'}
            
            if not required_cols.issubset(df.columns):
                self.logger.error(f"Missing required columns in {excel_file.name}")
                self.logger.error(f"Required: {required_cols} | Actual: {df.columns.tolist()}")
                return None
                
            self.logger.info(f"Successfully read Excel: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read {excel_file.name}: {str(e)}")
            return None

    def _validate_image_dir(self, img_dir):
        """验证图像目录"""
        if not img_dir.exists():
            self.logger.error(f"Image directory missing: {img_dir}")
            return False
            
        self.logger.info(f"Image directory: {img_dir}")
        return True

    def _process_data_row(self, row, img_dir, part_name, label):
        """处理单行数据"""
        try:
            # 解析复合ID（数字_部位名称）
            raw_id = str(row['Image ID']).strip()
            img_id, is_valid = self._parse_image_id(raw_id, part_name)
            if not is_valid:
                return False
                
            # 构建图像路径
            img_path = self._find_image_file(img_dir, img_id, part_name)
            if not img_path:
                return False
                
            # 验证图像完整性
            if not self._validate_image(img_path):
                return False
                
            # 添加样本
            self.samples.append((
                str(img_path),
                f"{part_name}: {row['Description']}",
                label
            ))
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing row: {str(e)}")
            return False

    def _parse_image_id(self, raw_id, expected_part):
        """解析复合ID格式（数字_部位名称）"""
        try:
            # 分割ID和实际部位
            parts = raw_id.split('_')
            if len(parts) < 2:
                raise ValueError("Invalid ID format")
                
            numeric_part = parts[0]
            actual_part = '_'.join(parts[1:])
            
            # 验证部位一致性
            if actual_part != expected_part:
                self.logger.warning(f"Part mismatch in ID: {raw_id} | Expected: {expected_part}")
                return None, False
                
            # 转换数字部分
            img_id = f"{int(numeric_part):04d}"
            return img_id, True
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Invalid Image ID: {raw_id} ({str(e)})")
            return None, False

    def _find_image_file(self, img_dir, img_id, part_name):
        """查找图像文件（支持多种扩展名）"""
        base_name = f"{img_id}_{part_name}"
        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        for ext in extensions:
            img_path = img_dir / f"{base_name}{ext}"
            if img_path.exists():
                self.logger.debug(f"Found image: {img_path.name}")
                return img_path
                
        self.logger.warning(f"Image not found: {base_name}.*")
        return None

    def _validate_image(self, img_path):
        """验证图像完整性"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except Exception as e:
            self.logger.error(f"Invalid image: {img_path.name} - {str(e)}")
            return False

    def _final_report(self):
        """生成最终加载报告"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"Total samples loaded: {len(self.samples)}")
        
        if len(self.samples) == 0:
            self.logger.error("CRITICAL ERROR: No samples loaded!")
            self._print_troubleshooting_guide()

    def _print_troubleshooting_guide(self):
        """显示故障排除指南"""
        guide = """
        TROUBLESHOOTING GUIDE:
        1. 文件结构验证
           - Excel文件名必须为 [部位]_analysis.xlsx（如 cheek_left_analysis.xlsx）
           - 图像目录必须与Excel中的部位名称一致
           - 图像文件名格式：4位数字_部位名称.扩展名（如 0123_cheek_left.png）

        2. 数据验证
           - Excel必须包含 Image ID 和 Description 列
           - Image ID 格式：数字_部位名称（如 0123_cheek_left）
           - 图像文件必须实际存在且可读取

        3. 常见错误处理
           - 出现Part mismatch警告：检查Excel文件名与Image ID中的部位名称是否一致
           - 图像未找到：检查文件名格式和扩展名
           - 无效的Image ID：确保ID格式为数字开头（如 0123_cheek_left）
        """
        self.logger.info(guide)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(
                images=image,
                text=text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            # 添加空标签序列
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs
        except Exception as e:
            logging.error(f"Error loading {img_path}: {str(e)}")
            return None


# 评分系统
def calculate_autism_score(part_descriptions):
    """
    根据面部部位描述计算自闭症可能性评分
    :param part_descriptions: 字典，键为部位名称，值为描述
    :return: 评分和依据
    """
    score = 50.0  # 初始分数
    evidence = []
    
    # 遍历每个部位及其描述
    for part, description in part_descriptions.items():
        description = description.lower()
        
        # 检查自闭症特征
        if part in AUTISM_FEATURES:
            for feature in AUTISM_FEATURES[part]:
                if feature.lower() in description:
                    score += 10.0
                    evidence.append(f"+10%: {part} 存在自闭症特征 '{feature}'")
        
        # 检查非自闭症特征
        if part in NON_AUTISM_FEATURES:
            for feature in NON_AUTISM_FEATURES[part]:
                if feature.lower() in description:
                    score -= 5.0
                    evidence.append(f"-5%: {part} 存在非自闭症特征 '{feature}'")
    
    # 限制分数在0-100之间
    score = max(0, min(100, score))
    
    return score, evidence

# 定义在全局作用域
def collate_fn(batch):
    return {
        "pixel_values": torch.cat([x["pixel_values"] for x in batch]),
        "input_ids": torch.cat([x["input_ids"] for x in batch]),
        "attention_mask": torch.cat([x["attention_mask"] for x in batch]),
        "labels": torch.cat([x["labels"] for x in batch])
    }

def fine_tune_model(model, processor, train_dataset, val_dataset=None, epochs=3, lr=3e-5, 
                    batch_size=8, accumulation_steps=4, output_dir=None):
    best_val_loss = float('inf')
    patience = 2
    no_improve = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=0,           
        pin_memory=True,
        persistent_workers=False
    )
 
    # 验证集DataLoader（如果使用）
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False
        )
 
    # 优化器配置
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), 
                     eps=1e-6, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader)*epochs,
        eta_min=1e-6
    )
 
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    best_val_loss = float('inf')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(progress_bar):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            
            loss = outputs.loss / accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
 
            scheduler.step(loss)
 
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
 
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
 
        # 验证阶段
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                    labels = batch["labels"].to(device)
                    outputs = model(**inputs, labels=labels)
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
 
            # 早停判断
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                # 保存模型
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
 
        # 每个epoch结束都保存最终模型
        final_model_path = output_dir / f"epoch_{epoch+1}"
        model.save_pretrained(final_model_path)
        processor.save_pretrained(final_model_path)
        print(f"Saved epoch model to {final_model_path}")
 
    # 训练结束保存最终模型
    final_save_path = output_dir / "final_model"
    model.save_pretrained(final_save_path)
    processor.save_pretrained(final_save_path)
    return model

# 使用微调后的模型生成描述并评分
def analyze_image(model, processor, image_path):
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 处理图像
    inputs = processor(images=image, return_tensors="pt",padding=True,
        truncation=True,
        max_length=128,
        return_tensors_type="pt")
    
    inputs["pixel_values"] = inputs.pixel_values.to(torch.float32).sub_(127.5).div_(127.5)
    
    # 将输入移到适当的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    
    # 生成描述
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    
    # 解码生成的文本
    generated_description = processor.decode(outputs[0], skip_special_tokens=True)
    

    part_descriptions = parse_generated_description(generated_description)
    
    # 计算评分
    score, evidence = calculate_autism_score(part_descriptions)
    
    return {
        "description": generated_description,
        "part_descriptions": part_descriptions,
        "score": score,
        "evidence": evidence
    }

# 解析生成的描述为各部位描述
def parse_generated_description(description):

    parts = {}
    
    # 分割不同部位的描述
    for part_desc in description.split(","):
        part_desc = part_desc.strip()
        if ":" in part_desc:
            part, desc = part_desc.split(":", 1)
            parts[part.strip()] = desc.strip()
    
    return parts

def verify_model_files(model_path):
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            missing_files.append(f)
    
    if missing_files:
        print(f"缺失关键文件: {missing_files}")
        return False
    return True
# 主函数
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def main():
    MODEL_LOCAL_PATH = r"C:\Users\liangming.jiang21\Desktop\models--Salesforce--blip2-opt-2.7b"
    MODEL_SAVE_PATH = r"C:\Users\liangming.jiang21\FYP\New Model"
 
    try:
        processor = Blip2Processor.from_pretrained(MODEL_LOCAL_PATH)
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_LOCAL_PATH,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
 
    # 创建数据集
    dataset = FacialPartDataset(
        AUTISM_PATH, 
        NON_AUTISM_PATH, 
        AUTISM_IMAGES, 
        NON_AUTISM_IMAGES,
        processor
    )
 
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
 
    # 验证模型文件完整性
    if not verify_model_files(MODEL_LOCAL_PATH):
        print("检测到模型文件缺失，请检查预训练模型目录")
        return
 
    # 执行微调
    fine_tuned_model = fine_tune_model(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5, 
        lr=2e-5,
        batch_size=8,
        output_dir=MODEL_SAVE_PATH  # 显式传递保存路径
    )
 
    # 加载最佳模型
    best_model_path = Path(MODEL_SAVE_PATH) / "best_model"
    if verify_model_files(best_model_path):
        best_model = Blip2ForConditionalGeneration.from_pretrained(
            best_model_path,
            torch_dtype=torch.float16
        )
    else:
        print("未找到有效最佳模型，使用最终模型")
        best_model = fine_tuned_model
 
    # 示例分析
    test_image_path = r"C:\Users\liangming.jiang21\FYP\consolidated(2940)\Autistic\0385.jpg"
    result = analyze_image(best_model, processor, test_image_path)
 
    print("\n==== 分析结果 ====")
    print(f"描述: {result['description']}")
    print(f"自闭症可能性评分: {result['score']:.1f}%")
    print("评分依据:")
    for evidence in result['evidence']:
        print(f"  - {evidence}")

if __name__ == "__main__":
    main()