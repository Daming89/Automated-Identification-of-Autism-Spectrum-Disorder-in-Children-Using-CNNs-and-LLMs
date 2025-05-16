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

1. **facial feature extraction(10 most times)**: Extracts and analyzes medical features from facial descriptions
2. **Facial part Description**: Uses BLIP2 model to generate clinical descriptions of facial regions
3. **Hybrid_Autism_Detector**: Combines VGG19 and BLIP2 models for hybrid ASD detection
4. **Hybrid_mode_test**: Comprehensive evaluation framework for the hybrid model
5. **only cnn**: Standalone evaluation of the CNN component
6. **ony llm**: Standalone LLM-based analysis of facial features
7. **training CNNs**: Scripts for training the CNN model with data augmentation
8. **training llms**: Scripts for fine-tuning the BLIP2 model on ASD facial features

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

##  Performances
1. **VGG19** : ![CNN19_result](images/CNN19_result.png)
2. **BLIP2**: ![BLIP2_result](images/BLIP2_result.png)
3. **Hybrid**: ![Hybrid_result](images/Hybrid_result.png)

## Results
1. This study explored the feasibility of a hybrid AI-based system for supporting autism screening by integrating Convolutional Neural Networks (CNNs) and Large Language Models (LLMs). The proposed architecture aimed to combine visual feature extraction with semantic interpretation to better emulate elements of clinical assessment. While the CNN component demonstrated a reasonable capacity for distinguishing between autistic and non-autistic facial images, the integration with LLMs, designed to provide descriptive reasoning, did not enhance performance and, in fact, reduced overall classification accuracy.
2. These findings emphasize the limitations of using static facial imagery and language models to approximate the behavioral observations that clinicians rely on in autism diagnosis. They also highlight the gap between current AI capabilities and the complexity of clinical reasoning, which typically depends on dynamic, context-rich behavioral data. Nonetheless, the research provides important contributions by identifying methodological barriers, clarifying the scope of AI in developmental disorder screening, and outlining considerations for future system design.
3. Despite the modest performance of the integrated model, this work has value as an exploratory step toward multimodal AI frameworks in healthcare. It underscores the importance of aligning input data types with the diagnostic processes they intend to support and offers a foundation for future studies seeking to improve the role of AI in early autism identification. 

