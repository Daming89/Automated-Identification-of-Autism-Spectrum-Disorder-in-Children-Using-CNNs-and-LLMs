# Automated Identification of Autism Spectrum Disorder in Children Using CNNs and LLMs

## Project Overview

This project aims to develop an automated system for identifying Autism Spectrum Disorder (ASD) in children through facial feature analysis using a hybrid approach combining Convolutional Neural Networks (CNNs) and Large Language Models (LLMs). The system analyzes facial characteristics that are clinically associated with ASD and provides a diagnostic assessment.

## Key Features

- **Multi-Modal Analysis**: Combines computer vision (VGG19 CNN) with natural language processing (BLIP2 LLM) for comprehensive facial feature assessment
- **Detailed Facial Region Analysis**: Examines 24 distinct facial regions with specialized medical prompts
- **Clinical Feature Mapping**: Incorporates known ASD-associated facial characteristics from medical literature
- **Explainable AI**: Provides evidence-based scoring with clinical feature explanations
- **Hybrid Scoring System**: Weighted combination of CNN and LLM predictions

## Code Structure

The repository contains 9 main Python scripts that implement different components of the system:

1. **Medical Feature Extraction**: Extracts and analyzes medical features from facial descriptions
2. **BLIP Clinical Analyzer**: Uses BLIP2 model to generate clinical descriptions of facial regions
3. **Hybrid Autism Detector**: Combines VGG19 and BLIP2 models for hybrid ASD detection
4. **Model Evaluation**: Comprehensive evaluation framework for the hybrid model
5. **VGG19 Evaluation**: Standalone evaluation of the CNN component
6. **LLM Autism Analyzer**: Standalone LLM-based analysis of facial features
7. **Confusion Matrix Visualization**: Visualization tools for model performance
8. **Model Training**: Scripts for training the VGG19 model with data augmentation
9. **Fine-Tuning**: Scripts for fine-tuning the BLIP2 model on ASD facial features

## Technical Components

### CNN Component (VGG19)
- Custom-trained VGG19 model for initial ASD probability assessment
- Data augmentation pipeline for robust training
- Transfer learning from ImageNet weights

### LLM Component (BLIP2)
- Fine-tuned BLIP2 model for generating clinical descriptions of facial features
- Specialized medical prompt templates for each facial region
- Feature-based scoring system aligned with clinical ASD markers

### Hybrid System
- Weighted combination of CNN and LLM scores
- Dynamic scoring adjustment based on feature matches
- Comprehensive evidence reporting

## Running
 ```bash
from hybrid_autism_detector import HybridAutismDetector

detector = HybridAutismDetector(
    vgg_model_path="path/to/vgg_model.h5",
    blip_model_path="path/to/blip2_model"
)

result = detector.analyze_image("path/to/image.jpg")
print(f"ASD Probability: {result['combined_score']:.1f}%")
print("Evidence:", result['evidence'])
```
## Dataset
The system was trained and evaluated on a dataset of facial images from:
1. **Autistic**: 2938
2. **Non-Autistic**: 2938
Images were preprocessed and segmented into 24 facial regions for detailed analysis.
