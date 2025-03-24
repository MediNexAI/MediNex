"""
Model Evaluation Script

This module provides tools for evaluating trained MediNex AI models.
"""

import os
import argparse
import json
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import seaborn as sns
import pandas as pd

# Import MediNex modules
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.medical_imaging.lung_ct_model import LungCTModel
from models.medical_imaging.chest_xray_model import ChestXrayModel
from training.data_loader import DatasetFactory, MedicalDatasetLoader
from config import OUTPUT_DIR


def evaluate_model(
    model_path: str,
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    save_results: bool = True,
    visualize: bool = True
) -> Dict[str, float]:
    """
    Evaluate a trained model on a test dataset.
    
    Args:
        model_path: Path to the saved model
        data_dir: Directory containing dataset
        batch_size: Batch size for evaluation
        save_results: Whether to save evaluation results to a file
        visualize: Whether to visualize evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n=== Evaluating Model: {model_path} ===")
    
    # Load model metadata to determine model type
    metadata_path = f"{model_path}_metadata.json"
    if not os.path.exists(metadata_path):
        raise ValueError(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Determine model type and load model
    model_name = metadata.get("name", "").lower()
    
    if "lungct" in model_name.replace(" ", ""):
        # Load Lung CT model
        model = LungCTModel.load(model_path)
        model_type = "lung_ct"
    elif "chestxray" in model_name.replace(" ", ""):
        # Load Chest X-ray model
        model = ChestXrayModel.load(model_path)
        model_type = "chest_xray"
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    print(f"Loaded {model_type} model: {model.metadata.name} v{model.metadata.version}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    if model_type == "lung_ct":
        _, _, test_ds = DatasetFactory.create_lung_ct_dataset(
            data_dir=data_dir,
            target_size=model.config["input_shape"][:2],
            batch_size=batch_size,
            augment=False
        )
        class_names = ["Normal", "Nodule"]
    else:  # chest_xray
        _, _, test_ds = DatasetFactory.create_chest_xray_dataset(
            data_dir=data_dir,
            target_size=model.config["input_shape"][:2],
            batch_size=batch_size,
            augment=False
        )
        class_names = [
            "Normal", "Pneumonia", "Tuberculosis", "COVID-19", "Lung Opacity", 
            "Pleural Effusion", "Cardiomegaly", "Nodule", "Mass", "Hernia", 
            "Infiltration", "Fibrosis", "Atelectasis", "Consolidation"
        ]
    
    # Get predictions and true labels
    print("\nGenerating predictions...")
    y_pred_raw = model.predict(test_ds)
    
    # Get true labels
    all_images = []
    all_labels = []
    for images, labels in test_ds:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    
    y_true = np.concatenate(all_labels, axis=0)
    test_images = np.concatenate(all_images, axis=0)
    
    # Process predictions based on the problem type
    num_classes = model.config["num_classes"]
    if num_classes == 2:
        # Binary classification
        y_pred_prob = y_pred_raw.ravel()
        y_pred = (y_pred_prob >= 0.5).astype(int)
    else:
        # Multi-class classification
        y_pred_prob = y_pred_raw
        y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_prob, num_classes)
    
    # Print metrics
    print("\nEvaluation metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"- {metric_name}: {metric_value:.4f}")
        else:
            print(f"- {metric_name}: {metric_value}")
    
    # Save results if requested
    if save_results:
        results_dir = os.path.join(OUTPUT_DIR, "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(results_dir, f"{model.metadata.name}_{timestamp}.json")
        
        # Create serializable metrics
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            elif isinstance(v, np.float32) or isinstance(v, np.float64):
                serializable_metrics[k] = float(v)
            else:
                serializable_metrics[k] = v
        
        # Add model metadata
        result_data = {
            "model_name": model.metadata.name,
            "model_version": model.metadata.version,
            "model_type": model_type,
            "metrics": serializable_metrics,
            "timestamp": timestamp
        }
        
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\nEvaluation results saved to: {result_path}")
    
    # Visualize if requested
    if visualize:
        # Visualize confusion matrix
        visualize_confusion_matrix(y_true, y_pred, class_names)
        
        # For binary classification, visualize ROC curve
        if num_classes == 2:
            visualize_roc_curve(y_true, y_pred_prob)
        
        # Visualize sample predictions
        indices = np.random.choice(len(y_true), min(10, len(y_true)), replace=False)
        sample_images = test_images[indices]
        sample_labels = y_true[indices]
        
        # Use model's visualization method if available
        if hasattr(model, 'visualize_predictions'):
            model.visualize_predictions(
                sample_images, 
                sample_labels,
                class_names=class_names,
                show_heatmap=hasattr(model, 'gradcam_heatmap')
            )
        else:
            # Generic visualization
            visualize_predictions(
                sample_images, 
                sample_labels, 
                y_pred[indices],
                class_names
            )
    
    return metrics


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_pred_prob: np.ndarray, 
    num_classes: int
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_prob: Prediction probabilities
        num_classes: Number of classes
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    if num_classes == 2:
        # Binary classification
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["f1_score"] = f1_score(y_true, y_pred)
        
        # AUC and ROC
        try:
            metrics["auc"] = roc_auc_score(y_true, y_pred_prob)
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
        except Exception as e:
            print(f"Error calculating ROC/AUC: {e}")
            metrics["auc"] = np.nan
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["confusion_matrix"] = [[tn, fp], [fn, tp]]
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
    else:
        # Multi-class classification
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted')
        metrics["recall"] = recall_score(y_true, y_pred, average='weighted')
        metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics["class_report"] = class_report
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        
        # Multi-class AUC (one-vs-rest)
        try:
            metrics["auc"] = roc_auc_score(
                tf.keras.utils.to_categorical(y_true),
                y_pred_prob,
                average='weighted',
                multi_class='ovr'
            )
        except Exception as e:
            print(f"Error calculating multi-class AUC: {e}")
            metrics["auc"] = np.nan
    
    return metrics


def visualize_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str]
) -> None:
    """
    Visualize confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def visualize_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray) -> None:
    """
    Visualize ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_prob: Prediction probabilities
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_predictions(
    images: np.ndarray, 
    true_labels: np.ndarray, 
    pred_labels: np.ndarray,
    class_names: List[str]
) -> None:
    """
    Visualize model predictions on sample images.
    
    Args:
        images: Input images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
    """
    n_images = len(images)
    
    # Determine grid layout
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(3*cols, 3*rows))
    
    for i in range(n_images):
        plt.subplot(rows, cols, i + 1)
        
        # Display the image
        if images[i].shape[-1] == 1:
            plt.imshow(images[i].squeeze(), cmap='gray')
        else:
            plt.imshow(images[i])
        
        # Get class names
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        
        # Set title with true and predicted labels
        title = f"True: {true_class}\nPred: {pred_class}"
        color = "green" if pred_labels[i] == true_labels[i] else "red"
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MediNex AI models")
    
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the saved model (without _metadata.json suffix)")
    parser.add_argument("--data_dir", type=str, help="Directory containing dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--no_save", action="store_true", help="Don't save evaluation results")
    parser.add_argument("--no_visualize", action="store_true", help="Don't visualize evaluation results")
    
    return parser.parse_args()


def main():
    """Main entry point for evaluation script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled on {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"Error configuring GPUs: {e}")
    
    # Evaluate model
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        save_results=not args.no_save,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main() 