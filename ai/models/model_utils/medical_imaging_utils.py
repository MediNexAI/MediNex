"""
Medical Imaging Utilities

This module provides utilities for loading, preprocessing, and augmenting 
medical imaging data.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pydicom
import SimpleITK as sitk
import nibabel as nib
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class ImageLoader:
    """Class for loading various medical image formats."""
    
    @staticmethod
    def load_dicom(filepath: str) -> np.ndarray:
        """
        Load a DICOM image.
        
        Args:
            filepath: Path to the DICOM file
            
        Returns:
            Image as numpy array
        """
        dicom_data = pydicom.dcmread(filepath)
        return dicom_data.pixel_array
    
    @staticmethod
    def load_dicom_series(directory: str) -> np.ndarray:
        """
        Load a series of DICOM images from a directory.
        
        Args:
            directory: Directory containing DICOM files
            
        Returns:
            Series of images as 3D numpy array
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Convert to numpy array
        array = sitk.GetArrayFromImage(image)
        return array
    
    @staticmethod
    def load_nifti(filepath: str) -> np.ndarray:
        """
        Load a NIfTI image.
        
        Args:
            filepath: Path to the NIfTI file
            
        Returns:
            Image as numpy array
        """
        nifti_img = nib.load(filepath)
        return nifti_img.get_fdata()
    
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """
        Load a standard image file (JPG, PNG, etc.).
        
        Args:
            filepath: Path to the image file
            
        Returns:
            Image as numpy array
        """
        return np.array(Image.open(filepath))
    
    @classmethod
    def load_any_image(cls, filepath: str) -> np.ndarray:
        """
        Load any supported image type based on extension.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            Image as numpy array
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.dcm':
            return cls.load_dicom(filepath)
        elif ext in ['.nii', '.nii.gz']:
            return cls.load_nifti(filepath)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            return cls.load_image(filepath)
        else:
            raise ValueError(f"Unsupported image format: {ext}")


class ImagePreprocessor:
    """Class for preprocessing medical images."""
    
    @staticmethod
    def normalize(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Normalize image to specified range.
        
        Args:
            image: Input image
            min_val: Minimum value after normalization
            max_val: Maximum value after normalization
            
        Returns:
            Normalized image
        """
        img_min = np.min(image)
        img_max = np.max(image)
        
        # Avoid division by zero
        if img_max == img_min:
            return np.zeros_like(image) + min_val
        
        normalized = min_val + (image - img_min) * (max_val - min_val) / (img_max - img_min)
        return normalized
    
    @staticmethod
    def resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (height, width)
            
        Returns:
            Resized image
        """
        # Handle different dimensionalities
        if len(image.shape) == 2:  # Grayscale
            return cv2.resize(image, (target_size[1], target_size[0]))
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale with channel
            return cv2.resize(image[:, :, 0], (target_size[1], target_size[0]))[:, :, np.newaxis]
        elif len(image.shape) == 3:  # RGB
            return cv2.resize(image, (target_size[1], target_size[0]))
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    
    @staticmethod
    def apply_window(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
        """
        Apply windowing to CT or other medical images.
        
        Args:
            image: Input image
            window_center: Window center (level)
            window_width: Window width
            
        Returns:
            Windowed image
        """
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        
        windowed = np.clip(image, img_min, img_max)
        windowed = (windowed - img_min) / (img_max - img_min)
        
        return windowed
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor
            
        Returns:
            Contrast-adjusted image
        """
        mean = np.mean(image)
        adjusted = mean + factor * (image - mean)
        return np.clip(adjusted, 0, 1)
    
    @staticmethod
    def preprocess_for_model(
        image: np.ndarray, 
        target_size: Tuple[int, int],
        channels: int = 1,
        normalize: bool = True,
        rescale: bool = False
    ) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image
            target_size: Target size (height, width)
            channels: Number of channels in output
            normalize: Whether to normalize the image
            rescale: Whether to rescale the image to [0, 1]
            
        Returns:
            Preprocessed image
        """
        # Resize
        if image.shape[:2] != target_size:
            image = ImagePreprocessor.resize(image, target_size)
        
        # Normalize
        if normalize:
            image = ImagePreprocessor.normalize(image)
        elif rescale and image.dtype != np.float32:
            # Simple rescaling for uint8 or uint16 images
            image = image.astype(np.float32) / np.max([np.max(image), 1])
        
        # Adjust channels
        if len(image.shape) == 2 and channels == 1:
            image = image[:, :, np.newaxis]
        elif len(image.shape) == 2 and channels == 3:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1 and channels == 3:
            image = np.repeat(image, 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 3 and channels == 1:
            # Convert RGB to grayscale
            image = np.mean(image, axis=2, keepdims=True)
        
        return image


class ImageAugmenter:
    """Class for augmenting medical images."""
    
    @staticmethod
    def random_rotate(image: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
        """
        Randomly rotate an image.
        
        Args:
            image: Input image
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Rotated image
        """
        angle = np.random.uniform(-max_angle, max_angle)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Handle different dimensionalities
        if len(image.shape) == 2:  # Grayscale
            return cv2.warpAffine(image, rotation_matrix, (width, height))
        else:  # RGB or with channel dimension
            return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    @staticmethod
    def random_shift(image: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
        """
        Randomly shift an image.
        
        Args:
            image: Input image
            max_shift: Maximum shift as a fraction of image dimension
            
        Returns:
            Shifted image
        """
        height, width = image.shape[:2]
        tx = np.random.uniform(-max_shift, max_shift) * width
        ty = np.random.uniform(-max_shift, max_shift) * height
        
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        
        if len(image.shape) == 2:  # Grayscale
            return cv2.warpAffine(image, translation_matrix, (width, height))
        else:  # RGB or with channel dimension
            return cv2.warpAffine(image, translation_matrix, (width, height))
    
    @staticmethod
    def random_zoom(image: np.ndarray, zoom_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Randomly zoom an image.
        
        Args:
            image: Input image
            zoom_range: Range of zoom factors
            
        Returns:
            Zoomed image
        """
        height, width = image.shape[:2]
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        
        zoom_matrix = np.array([
            [zx, 0, 0],
            [0, zy, 0]
        ], dtype=np.float32)
        
        result = cv2.warpAffine(
            image, 
            zoom_matrix, 
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return result
    
    @staticmethod
    def random_flip(image: np.ndarray) -> np.ndarray:
        """
        Randomly flip an image horizontally.
        
        Args:
            image: Input image
            
        Returns:
            Flipped image
        """
        if np.random.random() > 0.5:
            return np.fliplr(image)
        return image
    
    @staticmethod
    def random_brightness(image: np.ndarray, max_delta: float = 0.2) -> np.ndarray:
        """
        Randomly adjust brightness.
        
        Args:
            image: Input image
            max_delta: Maximum brightness adjustment
            
        Returns:
            Brightness-adjusted image
        """
        delta = np.random.uniform(-max_delta, max_delta)
        adjusted = image + delta
        return np.clip(adjusted, 0, 1)
    
    @staticmethod
    def random_contrast(image: np.ndarray, contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Randomly adjust contrast.
        
        Args:
            image: Input image
            contrast_range: Range of contrast factors
            
        Returns:
            Contrast-adjusted image
        """
        factor = np.random.uniform(contrast_range[0], contrast_range[1])
        mean = np.mean(image)
        adjusted = mean + factor * (image - mean)
        return np.clip(adjusted, 0, 1)
    
    @staticmethod
    def apply_random_augmentations(
        image: np.ndarray, 
        augmentation_list: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply a random selection of augmentations to an image.
        
        Args:
            image: Input image
            augmentation_list: List of augmentation types to apply
            
        Returns:
            Augmented image
        """
        if augmentation_list is None:
            augmentation_list = ['rotate', 'shift', 'zoom', 'flip', 'brightness', 'contrast']
        
        augmented = image.copy()
        
        # Apply each augmentation with 50% probability
        for aug_type in augmentation_list:
            if np.random.random() > 0.5:
                if aug_type == 'rotate':
                    augmented = ImageAugmenter.random_rotate(augmented)
                elif aug_type == 'shift':
                    augmented = ImageAugmenter.random_shift(augmented)
                elif aug_type == 'zoom':
                    augmented = ImageAugmenter.random_zoom(augmented)
                elif aug_type == 'flip':
                    augmented = ImageAugmenter.random_flip(augmented)
                elif aug_type == 'brightness':
                    augmented = ImageAugmenter.random_brightness(augmented)
                elif aug_type == 'contrast':
                    augmented = ImageAugmenter.random_contrast(augmented)
        
        return augmented


class DatasetPreparation:
    """Class for preparing datasets for model training."""
    
    @staticmethod
    def load_dataset_from_directory(
        directory: str,
        label_map: Dict[str, int],
        target_size: Tuple[int, int] = (256, 256),
        channels: int = 1,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        batch_size: int = 32,
        augment: bool = True,
        seed: int = 42
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load a dataset from a directory structure.
        
        Expects a directory structure with subdirectories for each class:
        directory/
          class1/
            image1.jpg
            image2.jpg
            ...
          class2/
            image1.jpg
            ...
          ...
        
        Args:
            directory: Root directory containing class subdirectories
            label_map: Dictionary mapping class directory names to numeric labels
            target_size: Target image size
            channels: Number of image channels
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            batch_size: Batch size for datasets
            augment: Whether to apply data augmentation
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        # Collect all image paths and labels
        image_paths = []
        labels = []
        
        for class_name, label in label_map.items():
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.dcm', '*.nii', '*.nii.gz']:
                pattern = os.path.join(class_dir, ext)
                image_paths.extend(glob.glob(pattern))
                labels.extend([label] * len(glob.glob(pattern)))
        
        # Split into train, validation, test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=test_split, stratify=labels, random_state=seed
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=validation_split / (1 - test_split),
            stratify=train_labels, random_state=seed
        )
        
        # Define preprocessing function
        def preprocess_image(path, label):
            # Read and preprocess image
            img = tf.py_function(
                lambda p: ImagePreprocessor.preprocess_for_model(
                    ImageLoader.load_any_image(p.numpy().decode()), 
                    target_size, 
                    channels
                ),
                [path],
                tf.float32
            )
            img.set_shape((target_size[0], target_size[1], channels))
            return img, label
        
        # Define augmentation function
        def augment_image(image, label):
            aug_img = tf.py_function(
                lambda img: ImageAugmenter.apply_random_augmentations(img.numpy()),
                [image],
                tf.float32
            )
            aug_img.set_shape(image.shape)
            return aug_img, label
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        train_ds = train_ds.map(preprocess_image)
        if augment:
            train_ds = train_ds.map(augment_image)
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_ds = val_ds.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
        test_ds = test_ds.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds
    
    @staticmethod
    def visualize_examples(dataset: tf.data.Dataset, class_names: List[str], n_examples: int = 5) -> None:
        """
        Visualize examples from a dataset.
        
        Args:
            dataset: TensorFlow dataset
            class_names: List of class names
            n_examples: Number of examples to visualize
        """
        # Get a batch of examples
        examples = next(iter(dataset))
        images, labels = examples
        
        # Convert to numpy for easier handling
        images = images.numpy()
        labels = labels.numpy()
        
        plt.figure(figsize=(12, 8))
        for i in range(min(n_examples, len(images))):
            plt.subplot(1, n_examples, i + 1)
            img = images[i]
            
            # Handle different channel configurations
            if img.shape[-1] == 1:
                img = img.squeeze()
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
                
            plt.title(f"Class: {class_names[labels[i]]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show() 