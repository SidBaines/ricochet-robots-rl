#!/usr/bin/env python3
"""
Script to generate ROC curves for different types of binary classifiers.
Shows curves for: random classifier, medium-performance classifier, and good classifier.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def generate_roc_curve_data(y_true, y_scores):
    """
    Generate ROC curve data (FPR, TPR) from true labels and prediction scores.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_scores: Array of prediction scores (higher = more likely to be positive class)
    
    Returns:
        fpr: False Positive Rate values
        tpr: True Positive Rate values
        thresholds: Threshold values used
    """
    # Sort by scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Get unique thresholds
    thresholds = np.unique(y_scores_sorted)
    thresholds = np.append(thresholds, np.inf)  # Add infinity for threshold > max score
    
    fpr = []
    tpr = []
    
    for threshold in thresholds:
        # Predictions: 1 if score >= threshold, 0 otherwise
        y_pred = (y_scores_sorted >= threshold).astype(int)
        
        # Calculate confusion matrix elements
        tp = np.sum((y_pred == 1) & (y_true_sorted == 1))
        fp = np.sum((y_pred == 1) & (y_true_sorted == 0))
        fn = np.sum((y_pred == 0) & (y_true_sorted == 1))
        tn = np.sum((y_pred == 0) & (y_true_sorted == 0))
        
        # Calculate TPR and FPR
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    return np.array(fpr), np.array(tpr), thresholds

def generate_synthetic_data(n_samples=1000, positive_ratio=0.3):
    """
    Generate synthetic binary classification data.
    
    Args:
        n_samples: Number of samples to generate
        positive_ratio: Ratio of positive samples
    
    Returns:
        y_true: True binary labels
    """
    n_positive = int(n_samples * positive_ratio)
    n_negative = n_samples - n_positive
    
    y_true = np.concatenate([
        np.ones(n_positive),
        np.zeros(n_negative)
    ])
    
    # Shuffle the labels
    np.random.shuffle(y_true)
    
    return y_true

def generate_random_classifier_scores(y_true):
    """Generate random prediction scores (random classifier)."""
    return np.random.random(len(y_true))

def generate_medium_classifier_scores(y_true, noise_level=0.3):
    """
    Generate prediction scores for a medium-performance classifier.
    Adds some signal but with significant noise.
    """
    # Start with true labels as base signal
    scores = y_true.astype(float)
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(y_true))
    scores += noise
    
    # Add some additional random variation
    scores += np.random.normal(0, 0.1, len(y_true))
    
    # Ensure scores are in reasonable range
    scores = np.clip(scores, 0, 1)
    
    return scores

def generate_good_classifier_scores(y_true, noise_level=0.1):
    """
    Generate prediction scores for a good classifier.
    Strong signal with minimal noise.
    """
    # Start with true labels as base signal
    scores = y_true.astype(float)
    
    # Add small amount of noise
    noise = np.random.normal(0, noise_level, len(y_true))
    scores += noise
    
    # Add some additional variation but keep it small
    scores += np.random.normal(0, 0.05, len(y_true))
    
    # Ensure scores are in reasonable range
    scores = np.clip(scores, 0, 1)
    
    return scores

def calculate_auc(fpr, tpr):
    """Calculate Area Under the Curve (AUC) for ROC curve."""
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr_sorted, fpr_sorted)
    return auc

def main():
    """Generate and plot ROC curves for different classifiers."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic binary classification data...")
    y_true = generate_synthetic_data(n_samples=10000, positive_ratio=0.3)
    
    # Generate scores for different classifiers
    print("Generating classifier scores...")
    random_scores = generate_random_classifier_scores(y_true)
    medium_scores = generate_medium_classifier_scores(y_true, noise_level=0.7)
    good_scores = generate_good_classifier_scores(y_true, noise_level=0.3)
    
    # Calculate ROC curves
    print("Calculating ROC curves...")
    fpr_random, tpr_random, _ = generate_roc_curve_data(y_true, random_scores)
    fpr_medium, tpr_medium, _ = generate_roc_curve_data(y_true, medium_scores)
    fpr_good, tpr_good, _ = generate_roc_curve_data(y_true, good_scores)
    
    # Calculate AUC scores
    auc_random = calculate_auc(fpr_random, tpr_random)
    auc_medium = calculate_auc(fpr_medium, tpr_medium)
    auc_good = calculate_auc(fpr_good, tpr_good)
    
    # Create the plot
    print("Creating ROC curves plot...")
    plt.figure(figsize=(6, 8/10*6))
    
    # Plot ROC curves
    plt.plot(fpr_random, tpr_random, 'r-', linewidth=2, 
             label=f"Actual Random Classifier    (AUC = {auc_random:.3f})")
    plt.plot(fpr_medium, tpr_medium, 'b-', linewidth=2, 
             label=f'Medium Classifier               (AUC = {auc_medium:.3f})')
    plt.plot(fpr_good, tpr_good, 'g-', linewidth=2, 
             label=f'Good Classifier                   (AUC = {auc_good:.3f})')
    
    # Plot diagonal line (random classifier reference)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label="'Perfect' Random Classifier (AUC = 0.500)")
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (background efficiency)', fontsize=12)
    plt.ylabel('True Positive Rate (signal efficiency)', fontsize=12)
    plt.title('Example ROC Curves for Binary Classifiers', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11, prop={'size': 8})
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add text box with information
    # info_text = f'Dataset: {len(y_true)} samples\nPositive class ratio: {np.mean(y_true):.1%}'
    # plt.text(0.6, 0.2, info_text, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    #          fontsize=10, verticalalignment='top')
    
    # Tight layout and save
    plt.tight_layout()
    
    # Save the plot
    output_path = '/Users/sidbaines/Documents/Code/20250911_RLNew/roc_curves_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves plot saved to: {output_path}")
    
    # Also save as PDF for better quality
    output_path_pdf = '/Users/sidbaines/Documents/Code/20250911_RLNew/roc_curves_plot.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"ROC curves plot also saved as PDF: {output_path_pdf}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("ROC CURVE SUMMARY")
    print("="*50)
    print(f"Random Classifier AUC:  {auc_random:.3f}")
    print(f"Medium Classifier AUC:  {auc_medium:.3f}")
    print(f"Good Classifier AUC:    {auc_good:.3f}")
    print(f"Dataset size:           {len(y_true)} samples")
    print(f"Positive class ratio:   {np.mean(y_true):.1%}")

if __name__ == "__main__":
    main()
