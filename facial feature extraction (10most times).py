import pandas as pd
import numpy as np
from collections import Counter
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import data
from tqdm import tqdm

# 设置NLTK数据路径
nltk.data.path.append(r"C:\Users\liangming.jiang21\nltk_data")

# 自动下载缺失资源
try:
    data.find('corpora/omw-1.4')
    data.find('corpora/wordnet')
    data.find('corpora/stopwords')
except LookupError:
    nltk.download('omw-1.4', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

# 初始化NLP工具
stop_words = set(stopwords.words('english') + ['would', 'could', 'might', 'may', 'patient', 'side', 'face'])
lemmatizer = WordNetLemmatizer()

# 医学形容词和特征词库
MEDICAL_ADJECTIVES = {
    'size': ['large', 'small', 'normal', 'abnormal', 'enlarged', 'reduced'],
    'shape': ['round', 'oval', 'irregular', 'asymmetric', 'symmetrical'],
    'color': ['pale', 'red', 'pigmented', 'cyanotic', 'jaundiced'],
    'texture': ['smooth', 'rough', 'wrinkled', 'thickened', 'thin']
}

def extract_medical_features(text):
    """提取医学特征形容词"""
    features = []
    
    # 提取尺寸描述
    size_pattern = r'(larger|smaller|normal|abnormal|enlarged|reduced)\s*(pupil|iris|eyelid)'
    features.extend(re.findall(size_pattern, text))
    
    # 提取形状描述
    shape_pattern = r'(round|oval|irregular|asymmetric|symmetrical)\s*(shape|contour|appearance)'
    features.extend(re.findall(shape_pattern, text))
    
    # 提取颜色描述
    color_pattern = r'(pale|red|pigmented|cyanotic|jaundiced)\s*(skin|area|region)'
    features.extend(re.findall(color_pattern, text))
    
    # 提取比较描述
    comparative_pattern = r'(more|less)\s*(prominent|visible|defined)'
    features.extend(re.findall(comparative_pattern, text))
    
    return [' '.join(f).strip() for f in features if f]

def clean_text(text):
    """增强型医学文本清洗"""
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # 基础清洗
        text = re.sub(r'[^a-zA-Z\s-]', '', text.lower().strip())
        text = re.sub(r'\s+', ' ', text)
        
        # 词形还原
        processed_words = []
        for word in text.split():
            if len(word) < 4:  # 过滤短词
                continue
                
            try:
                base_word = lemmatizer.lemmatize(word)
            except:
                base_word = word
            
            processed_words.append(base_word)
        
        # 过滤停用词
        filtered = [
            word for word in processed_words
            if word not in stop_words 
            and len(word) > 3
        ]
        
        return ' '.join(filtered) if filtered else ""
    except Exception as e:
        print(f"清洗错误: {str(e)}")
        return ""

def analyze_descriptions(file_path):
    """分析医学特征描述"""
    try:
        df = pd.read_excel(file_path).fillna({'Description': ''})
        df['Description'] = df['Description'].astype(str)
        df['cleaned'] = df['Description'].apply(clean_text)
        
        # 提取医学特征
        medical_features = []
        for desc in df['cleaned']:
            medical_features.extend(extract_medical_features(desc))
        
        # 统计特征频率
        feature_counts = Counter(medical_features)
        
        # 过滤低频特征(出现<5次)
        filtered_features = {k:v for k,v in feature_counts.items() if v >= 5}
        
        # 按频率排序
        sorted_features = sorted(filtered_features.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'medical_features': sorted_features[:20],  # 返回前20个最显著特征
            'total_descriptions': len(df),
            'valid_descriptions': sum(df['cleaned'].str.len() > 0)
        }
    except Exception as e:
        print(f"分析失败: {os.path.basename(file_path)} - {str(e)}")
        return {'medical_features': [], 'total_descriptions': 0, 'valid_descriptions': 0}

def generate_report(input_dir, output_file):
    """生成医学特征报告"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) 
            if f.endswith('.xlsx') and '_analysis' in f]
    
    report_data = []
    for file in tqdm(files, desc="处理进度"):
        try:
            part_name = file.split('_analysis')[0].replace('_', ' ').title()
            file_path = os.path.join(input_dir, file)
            result = analyze_descriptions(file_path)
            
            # 格式化特征描述
            features_formatted = [
                f"{feat[0]} ({feat[1]})" 
                for feat in result['medical_features']
            ]
            
            report_data.append({
                'Facial Region': part_name,
                'Key Medical Features': '\n'.join(features_formatted[:15]),  # 显示前15个特征
                'Total Images': result['total_descriptions'],
                'Valid Descriptions': result['valid_descriptions']
            })
        except Exception as e:
            print(f"文件 {file} 处理失败: {str(e)}")
    
    # 生成报告
    summary_df = pd.DataFrame(report_data)
    
    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, index=False)
            
            # 设置列宽
            worksheet = writer.sheets['Sheet1']
            worksheet.set_column('A:A', 25)
            worksheet.set_column('B:B', 60)
            worksheet.set_column('C:D', 15)
            
        print(f"医学特征报告已生成: {output_file}")
    except Exception as e:
        print(f"保存失败: {str(e)}")

if __name__ == "__main__":
    input_dir = r"C:\Users\liangming.jiang21\FYP\analysis_results_6.7b\Autism"
    output_file = r"C:\Users\liangming.jiang21\FYP\analysis_results\medical_features_summary_Autism.xlsx"
    
    generate_report(input_dir, output_file)