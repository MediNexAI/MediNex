"""
Chest X-ray Analysis Model

This module implements a neural network for analyzing 
chest X-ray images to detect common conditions like pneumonia, 
tuberculosis, and other abnormalities.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization,
    Flatten, Dense, Activation, GlobalAveragePooling2D, 
    SeparableConv2D, Add, Concatenate
)
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
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


class ChestXrayModel(TensorFlowBaseModel):
    """
    Chest X-ray analysis model for detecting common respiratory conditions.
    
    This model is designed to analyze chest X-ray images and detect conditions
    such as pneumonia, tuberculosis, COVID-19, and other abnormalities.
    """
    
    def __init__(self, model_id: str = "chest_xray", **kwargs):
        """
        Initialize the chest X-ray model.
        
        Args:
            model_id: Model identifier (default: "chest_xray")
            **kwargs: Additional arguments to override configuration
        """
        super().__init__(model_id, **kwargs)
        
        # Custom configurations specific to chest X-ray analysis
        self.backbone = kwargs.get("backbone", "efficientnet")
        self.use_pretrained = kwargs.get("use_pretrained", True)
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.attention_module = kwargs.get("attention_module", True)
        self.fc_layers = kwargs.get("fc_layers", [256, 128])
        self.l2_regularization = kwargs.get("l2_regularization", 0.001)
        
    def _attention_block(self, inputs, filters):
        """
        Create an attention block to focus on relevant features.
        
        Args:
            inputs: Input tensor
            filters: Number of filters in the convolutional layers
            
        Returns:
            Tensor with attention applied
        """
        # Channel attention
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        avg_pool = tf.keras.layers.Reshape((1, 1, filters))(avg_pool)
        
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, filters))(max_pool)
        
        # Shared MLP for both pooled features
        shared_mlp_1 = tf.keras.layers.Conv2D(filters // 8, kernel_size=1, activation='relu')
        shared_mlp_2 = tf.keras.layers.Conv2D(filters, kernel_size=1)
        
        avg_pool = shared_mlp_1(avg_pool)
        avg_pool = shared_mlp_2(avg_pool)
        
        max_pool = shared_mlp_1(max_pool)
        max_pool = shared_mlp_2(max_pool)
        
        channel_attention = tf.keras.layers.Add()([avg_pool, max_pool])
        channel_attention = tf.keras.layers.Activation('sigmoid')(channel_attention)
        
        # Apply channel attention
        channel_refined = tf.keras.layers.Multiply()([inputs, channel_attention])
        
        # Spatial attention
        avg_spatial = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True)
        )(channel_refined)
        
        max_spatial = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.max(x, axis=3, keepdims=True)
        )(channel_refined)
        
        spatial_concat = tf.keras.layers.Concatenate()([avg_spatial, max_spatial])
        spatial_attention = tf.keras.layers.Conv2D(
            1, kernel_size=7, padding='same', activation='sigmoid'
        )(spatial_concat)
        
        # Apply spatial attention
        refined_features = tf.keras.layers.Multiply()([channel_refined, spatial_attention])
        
        return refined_features
    
    def build(self) -> None:
        """
        Build the model architecture based on the configuration.
        """
        # Input layer matching the specified input shape
        input_shape = self.config["input_shape"]
        inputs = Input(shape=input_shape)
        
        # Create backbone based on configuration
        if self.backbone == "efficientnet":
            # EfficientNet backbone
            if input_shape[2] == 1:
                # If grayscale, replicate to 3 channels
                x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
                backbone = EfficientNetB0(
                    include_top=False,
                    weights='imagenet' if self.use_pretrained else None,
                    input_shape=(input_shape[0], input_shape[1], 3),
                    pooling='avg'
                )
                features = backbone(x)
            else:
                backbone = EfficientNetB0(
                    include_top=False,
                    weights='imagenet' if self.use_pretrained else None,
                    input_shape=input_shape,
                    pooling='avg'
                )
                features = backbone(inputs)
                
        elif self.backbone == "mobilenetv2":
            # MobileNetV2 backbone
            if input_shape[2] == 1:
                # If grayscale, replicate to 3 channels
                x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
                backbone = MobileNetV2(
                    include_top=False,
                    weights='imagenet' if self.use_pretrained else None,
                    input_shape=(input_shape[0], input_shape[1], 3),
                    pooling='avg'
                )
                features = backbone(x)
            else:
                backbone = MobileNetV2(
                    include_top=False,
                    weights='imagenet' if self.use_pretrained else None,
                    input_shape=input_shape,
                    pooling='avg'
                )
                features = backbone(inputs)
                
        else:
            # Custom CNN backbone without pretrained weights
            x = inputs
            
            # First convolutional block
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            
            # Apply attention if enabled
            if self.attention_module:
                x = self._attention_block(x, 32)
            
            # Depthwise separable convolution blocks for efficiency
            for filters in [64, 128, 256]:
                x = SeparableConv2D(filters, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = SeparableConv2D(filters, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(0.2)(x)
                
                # Apply attention if enabled
                if self.attention_module:
                    x = self._attention_block(x, filters)
            
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
            # Binary classification (normal vs. abnormal)
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            # Multi-class classification (specific conditions)
            outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
    
    def compile(self) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        """
        # Get configuration values
        learning_rate = self.config.get("learning_rate", 0.0001)
        
        # Set up optimizer with learning rate
        optimizer = Adam(learning_rate=learning_rate)
        
        # Set up loss function based on the problem type
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
        patience = kwargs.get("patience", self.config.get("patience", 15))
        monitor_metric = kwargs.get("monitor_metric", "val_loss")
        class_weights = kwargs.get("class_weights", None)
        
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
            class_weight=class_weights,  # Handle class imbalance if provided
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
            predictions = self.model.predict(data, **kwargs)
            
        elif isinstance(data, tf.data.Dataset):
            # TensorFlow dataset
            predictions = self.model.predict(data, **kwargs)
            
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
    
    def gradcam_heatmap(self, 
                         image: np.ndarray, 
                         layer_name: Optional[str] = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap to visualize model attention.
        
        Args:
            image: Input image (single image, not batch)
            layer_name: Name of the layer to use for Grad-CAM
            
        Returns:
            Heatmap as a numpy array
        """
        # Ensure model is built
        if self.model is None:
            raise ValueError("Model must be built before generating heatmap")
        
        # If layer name not provided, use the last convolutional layer
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
                    layer_name = layer.name
                    break
        
        # Create a model that maps the input to the output of the last conv layer
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Add batch dimension if not present
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Record gradients with GradientTape
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image)
            if self.config["num_classes"] == 2:
                # Binary classification
                loss = predictions[:, 0]
            else:
                # Multi-class - use the predicted class
                pred_index = tf.argmax(predictions[0])
                loss = predictions[:, pred_index]
                
        # Get gradients
        grads = tape.gradient(loss, conv_output)
        
        # Average gradients spatially
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps with gradients
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap
    
    def overlay_heatmap(self, 
                         image: np.ndarray, 
                         heatmap: np.ndarray, 
                         alpha: float = 0.4) -> np.ndarray:
        """
        Overlay the heatmap on the original image.
        
        Args:
            image: Original image
            heatmap: Heatmap generated by gradcam_heatmap
            alpha: Transparency factor for overlay
            
        Returns:
            Image with heatmap overlay
        """
        # Resize heatmap to match image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.image.resize(
            heatmap[tf.newaxis, ..., tf.newaxis],
            [image.shape[0], image.shape[1]]
        ).numpy().squeeze()
        
        # Apply colormap
        heatmap = plt.cm.jet(heatmap)[:, :, :3]
        heatmap = np.uint8(255 * heatmap)
        
        # Convert original image to RGB if grayscale
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
        
        # Ensure image is in 0-255 range
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        else:
            image = np.uint8(image)
        
        # Create overlay
        overlaid_image = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid_image
    
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
                              class_names: Optional[List[str]] = None, 
                              show_heatmap: bool = False) -> None:
        """
        Visualize model predictions on sample images.
        
        Args:
            images: Input images
            true_labels: True labels
            class_names: List of class names
            show_heatmap: Whether to show Grad-CAM heatmap
        """
        # Import cv2 here to avoid dependency if not needed
        if show_heatmap:
            import cv2
        
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
                class_names = ["Normal", "Abnormal"]
            else:
                class_names = [f"Class {i}" for i in range(self.config["num_classes"])]
        
        # Visualize predictions
        n_images = min(len(images), 5)  # Display up to 5 images
        n_cols = 2 if show_heatmap else 1
        plt.figure(figsize=(15, 3 * n_images))
        
        for i in range(n_images):
            # Display original image
            plt.subplot(n_images, n_cols, i * n_cols + 1)
            
            # Handle grayscale vs. color images
            if images[i].shape[-1] == 1:
                plt.imshow(images[i].squeeze(), cmap='gray')
            else:
                plt.imshow(images[i])
            
            # Set title with true and predicted labels
            title = f"True: {class_names[true_labels[i]]} | Pred: {class_names[pred_labels[i]]} ({pred_scores[i]:.2f})"
            color = "green" if pred_labels[i] == true_labels[i] else "red"
            plt.title(title, color=color)
            plt.axis('off')
            
            # Display heatmap if requested
            if show_heatmap:
                plt.subplot(n_images, n_cols, i * n_cols + 2)
                
                # Generate and overlay heatmap
                heatmap = self.gradcam_heatmap(images[i])
                if images[i].shape[-1] == 1:
                    # For grayscale images
                    img_for_overlay = np.tile(images[i].squeeze()[:, :, np.newaxis], (1, 1, 3))
                else:
                    img_for_overlay = images[i]
                    
                overlaid = self.overlay_heatmap(img_for_overlay, heatmap)
                plt.imshow(overlaid)
                plt.title("Grad-CAM Heatmap")
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# Usage example (to be removed in production)
if __name__ == "__main__":
    import cv2  # For heatmap visualization
    
    # Create model instance
    model = ChestXrayModel()
    
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
    loaded_model = ChestXrayModel.load(model_path)
    print("Model loaded successfully") 