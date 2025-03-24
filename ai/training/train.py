"""
Model Training Script

This module provides functionality for training MediNex AI models.
"""

import os
import argparse
import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt

# Import MediNex modules
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.medical_imaging.lung_ct_model import LungCTModel
from models.medical_imaging.chest_xray_model import ChestXrayModel
from training.data_loader import DatasetFactory
from config import OUTPUT_DIR, TRAINING_CONFIG


def train_model(
    model_type: str,
    data_dir: str,
    model_name: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_pretrained: bool = True,
    use_mixed_precision: bool = True,
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a medical imaging model.
    
    Args:
        model_type: Type of model to train ('lung_ct' or 'chest_xray')
        data_dir: Directory containing the dataset
        model_name: Custom name for the model (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        use_pretrained: Whether to use pretrained weights
        use_mixed_precision: Whether to use mixed precision training
        output_dir: Directory to save model and training outputs
        **kwargs: Additional model-specific parameters
        
    Returns:
        Dictionary containing training history and model metadata
    """
    print(f"\n=== Training {model_type.upper()} Model ===")
    
    # Set up output directory
    output_dir = output_dir or os.path.join(OUTPUT_DIR, "trained_models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure mixed precision if requested and available
    if use_mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision training enabled")
        except Exception as e:
            print(f"Failed to enable mixed precision: {e}")
    
    # Set up training parameters
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Generate model name if not provided
    if not model_name:
        model_name = f"{model_type.replace('_', '-')}-{timestamp}"
        
    # Configure training parameters
    training_params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_pretrained": use_pretrained,
    }
    
    # Update with any additional parameters
    training_params.update(kwargs)
    
    # Set up data and model based on model type
    if model_type == 'lung_ct':
        # Set up lung CT datasets
        train_ds, val_ds, test_ds = DatasetFactory.create_lung_ct_dataset(
            data_dir=data_dir,
            batch_size=batch_size,
            augment=True,
            slice_mode=kwargs.get('slice_mode', 'middle')
        )
        
        # Get model input shape from dataset
        for images, _ in train_ds.take(1):
            # Get shape from the first image in batch
            input_shape = images[0].shape
            break
        
        # Create lung CT model
        model = LungCTModel(
            name=model_name,
            input_shape=input_shape,
            num_classes=2,  # Binary classification: normal vs. nodule
            learning_rate=learning_rate,
            use_pretrained=use_pretrained,
            backbone=kwargs.get('backbone', 'resnet50'),
            dropout_rate=kwargs.get('dropout_rate', 0.5)
        )
        
    elif model_type == 'chest_xray':
        # Set up chest X-ray datasets
        train_ds, val_ds, test_ds = DatasetFactory.create_chest_xray_dataset(
            data_dir=data_dir,
            batch_size=batch_size,
            augment=True
        )
        
        # Get model input shape from dataset
        for images, _ in train_ds.take(1):
            # Get shape from the first image in batch
            input_shape = images[0].shape
            break
        
        # Create chest X-ray model
        model = ChestXrayModel(
            name=model_name,
            input_shape=input_shape,
            num_classes=14,  # Multiple classes of findings
            learning_rate=learning_rate,
            use_pretrained=use_pretrained,
            backbone=kwargs.get('backbone', 'efficientnetb3'),
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            use_attention=kwargs.get('use_attention', True)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Build the model
    model.build()
    
    # Compute class weights if needed
    if kwargs.get('use_class_weights', True):
        print("Computing class weights to handle imbalanced data...")
        class_weights = DatasetFactory.get_class_weights(data_dir, model_type)
        print(f"Class weights: {class_weights}")
    else:
        class_weights = None
    
    # Set up callbacks
    callbacks = create_training_callbacks(
        model_name=model_name,
        output_dir=output_dir,
        patience=kwargs.get('patience', 10),
        min_delta=kwargs.get('min_delta', 0.001),
        monitor=kwargs.get('monitor', 'val_loss')
    )
    
    # Train the model
    print("\nStarting model training...")
    history = model.train(
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weights=class_weights
    )
    
    # Save the trained model
    model_path = os.path.join(output_dir, model_name)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate the model on test set
    print("\nEvaluating model on test dataset...")
    evaluation = model.evaluate(test_ds)
    print(f"Test evaluation: {evaluation}")
    
    # Plot training history
    plot_training_history(history, model_name, output_dir)
    
    # Prepare results
    results = {
        "model_name": model_name,
        "model_type": model_type,
        "training_params": training_params,
        "history": history,
        "evaluation": evaluation,
        "model_path": model_path
    }
    
    # Save results to JSON
    results_path = os.path.join(output_dir, f"{model_name}_training_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = json.loads(
            json.dumps(results, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
        )
        json.dump(serializable_results, f, indent=2)
    
    print(f"Training results saved to {results_path}")
    
    return results


def create_training_callbacks(
    model_name: str,
    output_dir: str,
    patience: int = 10,
    min_delta: float = 0.001,
    monitor: str = 'val_loss'
) -> List[tf.keras.callbacks.Callback]:
    """
    Create callbacks for model training.
    
    Args:
        model_name: Name of the model
        output_dir: Directory to save outputs
        patience: Number of epochs with no improvement after which training will be stopped
        min_delta: Minimum change in monitored value to qualify as improvement
        monitor: Metric to monitor for early stopping and checkpoints
        
    Returns:
        List of Keras callbacks
    """
    # Create directories for checkpoints and logs
    checkpoint_dir = os.path.join(output_dir, 'checkpoints', model_name)
    log_dir = os.path.join(output_dir, 'logs', model_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = []
    
    # ModelCheckpoint - Save the best model
    checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5')
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=False,
        monitor=monitor,
        mode='auto',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # EarlyStopping - Stop training when a monitored metric has stopped improving
    early_stopping = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # ReduceLROnPlateau - Reduce learning rate when a monitored metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-6,
        verbose=1,
        mode='auto'
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard - Visualize training progress
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    return callbacks


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    output_dir: str
) -> None:
    """
    Plot and save training history.
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='Training Accuracy')
    elif 'acc' in history:
        plt.plot(history['acc'], label='Training Accuracy')
        
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    elif 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy')
        
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, f"{model_name}_training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {plot_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MediNex AI models")
    
    parser.add_argument("--model_type", type=str, required=True, 
                      choices=['lung_ct', 'chest_xray'],
                      help="Type of model to train")
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing the dataset")
    parser.add_argument("--model_name", type=str,
                      help="Custom name for the model")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Initial learning rate")
    parser.add_argument("--no_pretrained", action="store_true",
                      help="Don't use pretrained weights")
    parser.add_argument("--no_mixed_precision", action="store_true",
                      help="Don't use mixed precision training")
    parser.add_argument("--output_dir", type=str,
                      help="Directory to save model and training outputs")
    parser.add_argument("--backbone", type=str,
                      help="Backbone architecture for the model")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                      help="Dropout rate for regularization")
    parser.add_argument("--slice_mode", type=str, default="middle", 
                      choices=["middle", "all", "sample"],
                      help="How to handle 3D volumes (for lung_ct only)")
    parser.add_argument("--no_class_weights", action="store_true",
                      help="Don't use class weights for imbalanced data")
    parser.add_argument("--patience", type=int, default=10,
                      help="Patience for early stopping")
    parser.add_argument("--no_attention", action="store_true",
                      help="Don't use attention mechanisms (for chest_xray only)")
    
    return parser.parse_args()


def main():
    """Main entry point for training script."""
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
    
    # Set up additional parameters from arguments
    kwargs = {
        "backbone": args.backbone,
        "dropout_rate": args.dropout_rate,
        "use_class_weights": not args.no_class_weights,
        "patience": args.patience
    }
    
    # Add model-specific parameters
    if args.model_type == 'lung_ct':
        kwargs["slice_mode"] = args.slice_mode
    elif args.model_type == 'chest_xray':
        kwargs["use_attention"] = not args.no_attention
    
    # Train the model
    train_model(
        model_type=args.model_type,
        data_dir=args.data_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_pretrained=not args.no_pretrained,
        use_mixed_precision=not args.no_mixed_precision,
        output_dir=args.output_dir,
        **kwargs
    )


if __name__ == "__main__":
    main() 