#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Autism Detector Model Evaluation Script
This script evaluates the HybridAutismDetector model and generates comprehensive classification reports.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)

# Import your HybridAutismDetector class
from hybrid_autism_detector import HybridAutismDetector


def get_image_paths(dir_path):
    """扫描指定目录（非递归）获取唯一图片路径"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    seen = set()  # 用于去重
    image_paths = []
    
    # 扫描所有可能的扩展名（不区分大小写）
    for ext in valid_extensions:
        for p in dir_path.glob('*' + ext):
            resolved_p = p.resolve()  # 解析符号链接并标准化路径
            if resolved_p not in seen:
                seen.add(resolved_p)
                image_paths.append(p)
        # 添加大写扩展名的扫描（实际在 Windows 中可能不需要）
        for p in dir_path.glob('*' + ext.upper()):
            resolved_p = p.resolve()
            if resolved_p not in seen:
                seen.add(resolved_p)
                image_paths.append(p)
    
    return [str(p) for p in image_paths if p.is_file()]


def load_labels_from_csv(csv_path):
    """Load image labels from a CSV file"""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['image', 'label']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain 'image' and 'label' columns")
            return None
            
        # Convert to dictionary {image_name: label}
        labels = {row['image']: int(row['label']) for _, row in df.iterrows()}
        return labels
    except Exception as e:
        print(f"Error loading labels from CSV: {str(e)}")
        return None


def infer_labels_from_directories(test_dir):
    """从目录结构推断标签（增强版）"""
    labels = {}
    test_dir = Path(test_dir)
    
    # 自动检测Autistic/Non_Autistic子目录
    autism_dir = None
    non_autism_dir = None
    
    for subdir in test_dir.iterdir():
        if subdir.is_dir():
            if "autism" in subdir.name.lower() or "autistic" in subdir.name.lower():
                autism_dir = subdir
            elif "non" in subdir.name.lower() or "not" in subdir.name.lower():
                non_autism_dir = subdir
    
    if not autism_dir or not non_autism_dir:
        print("Error: Could not find both 'Autistic' and 'Non_Autistic' directories")
        return None
    
    # 收集自闭症图片
    for img_path in get_image_paths(autism_dir):
        labels[Path(img_path).name] = 1
    
    # 收集非自闭症图片
    for img_path in get_image_paths(non_autism_dir):
        labels[Path(img_path).name] = 1  # 保持键存在，后续覆盖值
    
    # 覆盖非自闭症标签
    for img_path in get_image_paths(non_autism_dir):
        labels[Path(img_path).name] = 0
    
    return labels


def evaluate_model(detector, test_data, ground_truth=None, weights=(0.9, 0.1), output_dir=None):
    """
    Evaluate the autism detector model and generate classification reports
    
    Args:
        detector: HybridAutismDetector instance
        test_data: Path to test directory or list of image paths
        ground_truth: Dictionary of {image_name: label}
        weights: Tuple of (VGG_weight, BLIP_weight)
        output_dir: Directory to save outputs (default: current directory)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Setup output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = '.'
        
    # Get test images
    if isinstance(test_data, str):
        print(f"Loading images from directory: {test_data}")
        image_paths = get_image_paths(test_data)
    else:
        # Assume test_data is already a list of paths
        image_paths = test_data
    
    print(f"Found {len(image_paths)} images for evaluation")
    
    # Determine ground truth
    if ground_truth is None:
        print("No ground truth provided, attempting to infer from directory structure...")
        ground_truth = infer_labels_from_directories(Path(test_data).parent if isinstance(test_data, str) else ".")
        if not ground_truth:
            print("Error: Failed to infer ground truth labels")
            return None
    
    print(f"Using {len(ground_truth)} ground truth labels")
    
    # Run predictions
    y_true = []
    y_pred = []
    probas = []
    results = []
    failed_images = []
    
    print("Starting model evaluation...")
    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            img_name = Path(img_path).name
            if img_name not in ground_truth:
                continue
                
            # Get prediction
            result = detector.analyze_image(
                img_path, 
                weights=weights, 
                verbose=False, 
                save_report=False
            )
            results.append(result)
            
            # Store result
            y_true.append(ground_truth[img_name])
            y_pred.append(1 if result["diagnosis"] else 0)
            probas.append(result["combined_score"] / 100.0)  # Convert to 0-1 range
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            failed_images.append(img_path)
    
    # Calculate metrics
    if len(y_true) == 0:
        print("No valid predictions were made, cannot evaluate model")
        return None
    
    print(f"Successfully processed {len(y_true)} images ({len(failed_images)} failed)")
    
    # Calculate basic metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=["Non-Autism", "Autism"], output_dict=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Non-Autism", "Autism"],
                yticklabels=["Non-Autism", "Autism"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.4f}, F1: {f1:.4f}')
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # Collect metrics
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "failed_images": [str(img) for img in failed_images],
        "total_images": len(image_paths),
        "processed_images": len(y_true),
        "autism_count": int(sum(y_true)),
        "predicted_autism": int(sum(y_pred)),
        "timestamp": timestamp
    }
    
    # Calculate ROC and AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, probas)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        # Save to file
        roc_path = os.path.join(output_dir, f"roc_curve_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.close()
        
        metrics["roc_auc"] = float(roc_auc)
    except Exception as e:
        print(f"Could not generate ROC curve: {str(e)}")
    
    # Generate detailed results table
    try:
        results_df = pd.DataFrame({
            'image': [Path(r['image_path']).name for r in results],
            'true_label': y_true,
            'predicted_label': y_pred,
            'probability': probas,
            'vgg_score': [r['vgg_score'] for r in results],
            'blip_score': [r['blip_score'] for r in results],
            'combined_score': [r['combined_score'] for r in results],
            'correct': [t == p for t, p in zip(y_true, y_pred)]
        })
        
        # Save detailed results
        results_path = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        metrics["detailed_results_file"] = results_path
    except Exception as e:
        print(f"Could not generate detailed results table: {str(e)}")
    
    # Print comprehensive report
    print("\n" + "="*80)
    print("MODEL EVALUATION REPORT")
    print("="*80)
    print(f"Total images: {len(image_paths)}")
    print(f"Successfully processed: {len(y_true)} ({len(y_true)/len(image_paths)*100:.1f}%)")
    print(f"Failed images: {len(failed_images)}")
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"  True Non-Autism, Predicted Non-Autism: {cm[0,0]}")
    print(f"  True Non-Autism, Predicted Autism:     {cm[0,1]}")
    print(f"  True Autism,     Predicted Non-Autism: {cm[1,0]}")
    print(f"  True Autism,     Predicted Autism:     {cm[1,1]}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Autism", "Autism"]))
    
    print(f"\nSaved confusion matrix to: {cm_path}")
    if "roc_auc" in metrics:
        print(f"Saved ROC curve to: {roc_path}")
    
    # Save the complete evaluation report to file
    report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Full report saved to: {report_file}")
    if "detailed_results_file" in metrics:
        print(f"Detailed results saved to: {metrics['detailed_results_file']}")
    
    print("="*80)
    
    return metrics


def create_subgroup_analysis(metrics, output_dir='.'):
    """Generate subgroup analysis based on evaluation results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if "detailed_results_file" not in metrics:
        print("No detailed results available for subgroup analysis")
        return
    
    try:
        # Load detailed results
        results_df = pd.read_csv(metrics["detailed_results_file"])
        
        # Performance by confidence level
        confidence_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        results_df['confidence_bin'] = pd.cut(
            results_df['probability'], 
            bins=confidence_bins, 
            labels=[f"{int(confidence_bins[i]*100)}-{int(confidence_bins[i+1]*100)}%" 
                    for i in range(len(confidence_bins)-1)]
        )
        
        # Accuracy by confidence bin
        accuracy_by_conf = results_df.groupby('confidence_bin')['correct'].mean()
        count_by_conf = results_df.groupby('confidence_bin').size()
        
        # Plot accuracy by confidence
        plt.figure(figsize=(10, 6))
        ax = accuracy_by_conf.plot(kind='bar', color='skyblue')
        plt.title('Accuracy by Confidence Level')
        plt.xlabel('Confidence Range')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        
        # Add count labels
        for i, (acc, count) in enumerate(zip(accuracy_by_conf, count_by_conf)):
            plt.text(i, acc + 0.05, f"n={count}", ha='center')
        
        # Add accuracy labels
        for i, acc in enumerate(accuracy_by_conf):
            plt.text(i, acc/2, f"{acc:.2f}", ha='center', color='white', fontweight='bold')
        
        # Save the figure
        conf_path = os.path.join(output_dir, f"accuracy_by_confidence_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(conf_path, dpi=300)
        plt.close()
        
        print(f"Subgroup analysis saved to: {conf_path}")
        
        # Save subgroup statistics
        subgroup_stats = {
            "accuracy_by_confidence": {str(idx): float(val) for idx, val in accuracy_by_conf.items()},
            "count_by_confidence": {str(idx): int(val) for idx, val in count_by_conf.items()}
        }
        
        subgroup_file = os.path.join(output_dir, f"subgroup_analysis_{timestamp}.json")
        with open(subgroup_file, 'w') as f:
            json.dump(subgroup_stats, f, indent=2)
            
        print(f"Subgroup statistics saved to: {subgroup_file}")
    
    except Exception as e:
        print(f"Error in subgroup analysis: {str(e)}")


def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Autism Detector')
    
    # Required arguments
    parser.add_argument('--vgg_model', type=str, required=True,
                        help='Path to VGG19 model file')
    
    # Data sources (at least one required)
    data_group = parser.add_argument_group('Data Sources (at least one required)')
    data_group.add_argument('--test_dir', type=str, default=None,
                        help='Directory containing test images')
    data_group.add_argument('--autism_dir', type=str, default=None,
                        help='Directory containing autism images')
    data_group.add_argument('--non_autism_dir', type=str, default=None,
                        help='Directory containing non-autism images')
    
    # Optional arguments
    parser.add_argument('--blip_model', type=str, default=r"C:\Users\liangming.jiang21\Desktop\models--Salesforce--blip2-opt-2.7b",
                        help='Path to BLIP2 model directory (optional)')
    parser.add_argument('--labels_file', type=str, default=None,
                        help='CSV file with image labels (image,label columns)')
    parser.add_argument('--output_dir', type=str, default=r"C:\Users\liangming.jiang21\FYP\dataset01\test1\results_hybrid",
                        help='Directory to save evaluation results')
    parser.add_argument('--vgg_weight', type=float, default=0.9,
                        help='Weight for VGG19 model (0-1)')
    parser.add_argument('--blip_weight', type=float, default=0.1,
                        help='Weight for BLIP2 model (0-1)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.test_dir and not (args.autism_dir and args.non_autism_dir):
        parser.error("Either --test_dir or both --autism_dir and --non_autism_dir must be provided")
    
    # Check if model exists
    if not os.path.exists(args.vgg_model):
        print(f"Error: VGG19 model not found at {args.vgg_model}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    print("Initializing Hybrid Autism Detector...")
    detector = HybridAutismDetector(
        vgg_model_path=args.vgg_model,
        blip_model_path=args.blip_model,
        device=args.device
    )
    
    # Set weights
    weights = (args.vgg_weight, args.blip_weight)
    print(f"Using weights: VGG={args.vgg_weight}, BLIP={args.blip_weight}")
    
    # Prepare test data and ground truth
    ground_truth = None
    test_data = None
    
    if args.labels_file:
        print(f"Loading labels from {args.labels_file}")
        ground_truth = load_labels_from_csv(args.labels_file)
    
    if args.test_dir:
        print(f"Using test directory: {args.test_dir}")
        test_data = args.test_dir
    
        # 新增：验证测试目录结构
        autism_dir = Path(args.test_dir) / "Autistic"
        non_autism_dir = Path(args.test_dir) / "Non_Autistic"
    
        if not autism_dir.exists() or not non_autism_dir.exists():
            print(f"Error: Test directory must contain 'Autistic' and 'Non_Autistic' subdirectories")
            return 1
    else:
        # Using autism and non-autism directories
        autism_images = get_image_paths(args.autism_dir)
        non_autism_images = get_image_paths(args.non_autism_dir)
        test_data = autism_images + non_autism_images
        
        # Create ground truth if not loaded from file
        if not ground_truth:
            ground_truth = {}
            for img in autism_images:
                ground_truth[Path(img).name] = 1
            for img in non_autism_images:
                ground_truth[Path(img).name] = 0
            
        print(f"Test data: {len(autism_images)} autism images, {len(non_autism_images)} non-autism images")
    
    # Run evaluation
    metrics = evaluate_model(
        detector, 
        test_data, 
        ground_truth, 
        weights=weights, 
        output_dir=args.output_dir
    )
    
    if metrics:
        # Generate subgroup analysis
        create_subgroup_analysis(metrics, args.output_dir)
        return 0
    else:
        print("Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
