"""
Lung CT Analysis Model

This module implements a convolutional neural network for 
detecting lung nodules in CT scans.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization,
    Flatten, Dense, Activation, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Import base model class
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.model_utils.base_model import TensorFlowBaseModel
from config import LOGS_DIR


class LungCTModel(TensorFlowBaseModel):
    """
    Lung CT scan analysis model for detecting lung nodules.
    
    This model analyzes CT scan slices to detect potential lung nodules,
    which may indicate lung cancer or other respiratory conditions.
    """
    
    def __init__(self, model_id: str = "lung_ct", **kwargs):
        """
        Initialize the lung CT model.
        
        Args:
            model_id: Model identifier (default: "lung_ct")
            **kwargs: Additional arguments to override configuration
        """
        super().__init__(model_id, **kwargs)
        
        # Custom configurations specific to lung CT analysis
        self.backbone = kwargs.get("backbone", "resnet50")
        self.use_pretrained = kwargs.get("use_pretrained", True)
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.fc_layers = kwargs.get("fc_layers", [256, 128])
        self.l2_regularization = kwargs.get("l2_regularization", 0.001)
    
    def build(self) -> None:
        """
        Build the model architecture based on the configuration.
        """
        # Input layer matching the specified input shape
        input_shape = self.config["input_shape"]
        inputs = Input(shape=input_shape)
        
        # Create backbone based on configuration
        if self.backbone == "resnet50":
            # ResNet50 backbone
            if input_shape[2] == 1:
                # If grayscale, replicate to 3 channels
                x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
                backbone = ResNet50(
                    include_top=False,
                    weights='imagenet' if self.use_pretrained else None,
                    input_shape=(input_shape[0], input_shape[1], 3),
                    pooling='avg'
                )
                features = backbone(x)
            else:
                backbone = ResNet50(
                    include_top=False,
                    weights='imagenet' if self.use_pretrained else None,
                    input_shape=input_shape,
                    pooling='avg'
                )
                features = backbone(inputs)
                
        elif self.backbone == "densenet121":
            # DenseNet121 backbone
            if input_shape[2] == 1:
                # If grayscale, replicate to 3 channels
                x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
                backbone = DenseNet121(
                    include_top=False,
                    weights='imagenet' if self.use_pretrained else None,
                    input_shape=(input_shape[0], input_shape[1], 3),
                    pooling='avg'
                )
                features = backbone(x)
            else:
                backbone = DenseNet121(
                    include_top=False,
                    weights='imagenet' if self.use_pretrained else None,
                    input_shape=input_shape,
                    pooling='avg'
                )
                features = backbone(inputs)
                
        else:
            # Custom CNN backbone without pretrained weights
            x = inputs
            
            # Multiple convolutional blocks with increasing filters
            filter_sizes = [32, 64, 128, 256]
            for filters in filter_sizes:
                x = Conv2D(filters, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(filters, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(0.2)(x)
            
            # Global pooling to reduce spatial dimensions
            features = GlobalAveragePooling2D()(x)
        
        # Fully connected layers
        x = features
        for units in self.fc_layers:
            x = Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization)
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        num_classes = self.config["num_classes"]
        if num_classes == 2:
            # Binary classification (nodule or not)
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            # Multi-class classification
            outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
    
    def compile(self) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        """
        # Get configuration values
        learning_rate = self.config.get("learning_rate", 0.0001)
        
        # Set up optimizer
        optimizer = Adam(learning_rate=learning_rate)
        
        # Set up loss function
        if self.config["num_classes"] == 2:
            loss = 'binary_crossentropy'
            metrics = [
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = [
                'accuracy',
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def train(self, 
              train_data: tf.data.Dataset, 
              validation_data: Optional[tf.data.Dataset] = None, 
              **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given dataset.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history and results
        """
        # Ensure model is built and compiled
        if self.model is None:
            self.build()
            self.compile()
        
        # Get training parameters
        epochs = kwargs.get("epochs", self.config.get("epochs", 100))
        batch_size = kwargs.get("batch_size", self.config.get("batch_size", 16))
        patience = kwargs.get("patience", self.config.get("patience", 15))
        monitor_metric = kwargs.get("monitor_metric", "val_loss")
        
        # Create log directory for TensorBoard
        current_time = int(time.time())
        log_dir = os.path.join(LOGS_DIR, f"{self.metadata.name}_{current_time}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor=monitor_metric, 
                patience=patience, 
                verbose=1, 
                restore_best_weights=True
            ),
            
            # Save best model during training
            ModelCheckpoint(
                os.path.join(log_dir, 'best_model.h5'),
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reduce learning rate when plateaus
            ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.5,
                patience=patience // 3,
                verbose=1,
                min_lr=1e-6
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        # Additional callbacks from kwargs
        if "callbacks" in kwargs:
            callbacks.extend(kwargs["callbacks"])
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Update metadata with the latest accuracy
        if history.history.get('val_accuracy'):
            self.metadata.accuracy = float(max(history.history['val_accuracy']))
        elif history.history.get('accuracy'):
            self.metadata.accuracy = float(max(history.history['accuracy']))
        
        # Update other performance metrics in metadata
        metrics = {}
        for metric in ['val_precision', 'val_recall', 'val_auc']:
            if metric in history.history:
                metrics[metric.replace('val_', '')] = float(max(history.history[metric]))
        
        if metrics:
            self.metadata.performance_metrics.update(metrics)
        
        # Return training history
        return {
            "history": history.history,
            "epochs_completed": len(history.history['loss']),
            "final_metrics": {k: history.history[k][-1] for k in history.history.keys()}
        }
    
    def predict(self, 
                data: Union[np.ndarray, tf.data.Dataset], 
                **kwargs) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
        """
        # Ensure model is built
        if self.model is None:
            raise ValueError("Model must be built before making predictions")
        
        # Handle different input types
        if isinstance(data, np.ndarray):
            # Single image or batch of images as numpy array
            if len(data.shape) == 3:  # Single image without batch dimension
                # Add batch dimension
                data = np.expand_dims(data, axis=0)
            
            # Make prediction
            predictions = self.model.predict(data)
            
        elif isinstance(data, tf.data.Dataset):
            # TensorFlow dataset
            predictions = self.model.predict(data)
            
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return predictions
    
    def evaluate(self, 
                 test_data: tf.data.Dataset, 
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure model is built
        if self.model is None:
            raise ValueError("Model must be built before evaluation")
        
        # Get predictions
        y_pred_raw = self.model.predict(test_data)
        
        # Get true labels
        y_true = np.concatenate([y for _, y in test_data], axis=0)
        
        # Process predictions based on the problem type
        if self.config["num_classes"] == 2:
            # Binary classification
            y_pred_prob = y_pred_raw.ravel()
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "auc": roc_auc_score(y_true, y_pred_prob)
            }
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
        else:
            # Multi-class classification
            y_pred_prob = y_pred_raw
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "f1_score": f1_score(y_true, y_pred, average='weighted')
            }
            
            # Multi-class AUC (one-vs-rest)
            try:
                metrics["auc"] = roc_auc_score(
                    tf.keras.utils.to_categorical(y_true),
                    y_pred_prob,
                    average='weighted',
                    multi_class='ovr'
                )
            except ValueError:
                # Handle case where only one class is present
                metrics["auc"] = 0.0
        
        # Store metrics in metadata
        self.metadata.accuracy = float(metrics["accuracy"])
        self.metadata.performance_metrics.update({
            k: float(v) for k, v in metrics.items() if k != "accuracy"
        })
        
        # Return metrics
        return metrics
    
    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
        """
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axs[0].plot(history.get('accuracy', []), label='Training Accuracy')
        axs[0].plot(history.get('val_accuracy', []), label='Validation Accuracy')
        axs[0].set_title('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend()
        
        # Plot loss
        axs[1].plot(history.get('loss', []), label='Training Loss')
        axs[1].plot(history.get('val_loss', []), label='Validation Loss')
        axs[1].set_title('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, 
                              images: np.ndarray, 
                              true_labels: np.ndarray, 
                              class_names: Optional[List[str]] = None) -> None:
        """
        Visualize model predictions on sample images.
        
        Args:
            images: Input images
            true_labels: True labels
            class_names: List of class names
        """
        # Make predictions
        predictions = self.predict(images)
        
        # Process predictions based on the problem type
        if self.config["num_classes"] == 2:
            # Binary classification
            pred_labels = (predictions.ravel() >= 0.5).astype(int)
            pred_scores = predictions.ravel()
        else:
            # Multi-class classification
            pred_labels = np.argmax(predictions, axis=1)
            pred_scores = np.max(predictions, axis=1)
        
        # Default class names if not provided
        if class_names is None:
            if self.config["num_classes"] == 2:
                class_names = ["Negative", "Positive"]
            else:
                class_names = [f"Class {i}" for i in range(self.config["num_classes"])]
        
        # Visualize predictions
        n_images = min(len(images), 5)  # Display up to 5 images
        plt.figure(figsize=(15, 3 * n_images))
        
        for i in range(n_images):
            plt.subplot(n_images, 1, i + 1)
            
            # Display the image
            if images[i].shape[-1] == 1:
                plt.imshow(images[i].squeeze(), cmap='gray')
            else:
                plt.imshow(images[i])
            
            # Set title with true and predicted labels
            title = f"True: {class_names[true_labels[i]]} | Pred: {class_names[pred_labels[i]]} ({pred_scores[i]:.2f})"
            color = "green" if pred_labels[i] == true_labels[i] else "red"
            plt.title(title, color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# Usage example (to be removed in production)
if __name__ == "__main__":
    # Create model instance
    model = LungCTModel()
    
    # Build and compile the model
    model.build()
    model.compile()
    
    # Print model summary
    print(model.summary())
    
    # Example of training with dummy data
    print("Note: This is a dummy example. Replace with actual data in production.")
    input_shape = model.config["input_shape"]
    
    # Create dummy data
    dummy_x = np.random.random((100, *input_shape))
    dummy_y = np.random.randint(0, 2, (100, 1))
    
    # Convert to TensorFlow dataset
    dummy_ds = tf.data.Dataset.from_tensor_slices((dummy_x, dummy_y))
    dummy_ds = dummy_ds.batch(16)
    
    # Train the model
    train_results = model.train(
        dummy_ds,
        validation_data=dummy_ds,
        epochs=2  # Small number for demonstration
    )
    
    # Show training history
    model.plot_training_history(train_results["history"])
    
    # Save the model
    model_path = model.save()
    print(f"Model saved to: {model_path}")
    
    # Load the model back
    loaded_model = LungCTModel.load(model_path)
    print("Model loaded successfully") 