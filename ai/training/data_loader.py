"""
Data Loader Module for MediNex AI

This module provides utilities for loading and preprocessing medical imaging datasets.
"""

import os
import glob
from typing import Tuple, List, Dict, Any, Optional, Callable, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class MedicalDatasetLoader:
    """Base class for medical dataset loaders."""
    
    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
        augment: bool = True
    ):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory containing the dataset
            target_size: Target image size (height, width)
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            seed: Random seed for reproducibility
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.seed = seed
        self.augment = augment
        
        # Validate data directory
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
    
    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create train, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        raise NotImplementedError("Subclasses must implement create_datasets()")
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        raise NotImplementedError("Subclasses must implement get_class_weights()")
    
    def create_augmentation_pipeline(self) -> ImageDataGenerator:
        """
        Create data augmentation pipeline.
        
        Returns:
            ImageDataGenerator configured for medical imaging
        """
        if not self.augment:
            return ImageDataGenerator(rescale=1./255)
        
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,  # Usually medical images have a specific orientation
            fill_mode='nearest'
        )
    
    @staticmethod
    def preprocess_image(
        image: tf.Tensor, 
        target_size: Tuple[int, int],
        normalize: bool = True
    ) -> tf.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: Input image tensor
            target_size: Target size (height, width)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image
        image = tf.image.resize(image, target_size)
        
        # Ensure 3 channels (convert grayscale to RGB if needed)
        if image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        
        # Normalize pixel values to [0, 1]
        if normalize:
            image = tf.cast(image, tf.float32) / 255.0
        
        return image
    
    @staticmethod
    def load_and_preprocess_image(
        image_path: str,
        target_size: Tuple[int, int],
        label: int = None
    ) -> Tuple[tf.Tensor, Any]:
        """
        Load and preprocess an image from disk.
        
        Args:
            image_path: Path to image file
            target_size: Target size (height, width)
            label: Optional class label
            
        Returns:
            Tuple of (preprocessed_image, label)
        """
        # Read image file
        image = tf.io.read_file(image_path)
        
        # Decode image
        try:
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
        except tf.errors.InvalidArgumentError:
            # Fallback for corrupted images
            print(f"Warning: Could not decode image {image_path}. Using placeholder.")
            image = tf.zeros([*target_size, 3], dtype=tf.uint8)
        
        # Set the shape since decode_image does not set the shape
        image.set_shape([None, None, 3])
        
        # Preprocess the image
        image = MedicalDatasetLoader.preprocess_image(image, target_size)
        
        return (image, label) if label is not None else image


class ChestXrayDatasetLoader(MedicalDatasetLoader):
    """Loader for chest X-ray datasets."""
    
    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
        augment: bool = True
    ):
        """Initialize chest X-ray dataset loader."""
        super().__init__(
            data_dir, target_size, batch_size, 
            validation_split, test_split, seed, augment
        )
        
        # Get all class directories
        self.class_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, "*")) 
                                  if os.path.isdir(d)])
        
        # Map class names to indices
        self.class_names = [os.path.basename(d) for d in self.class_dirs]
        self.class_map = {name: i for i, name in enumerate(self.class_names)}
        
        print(f"Found {len(self.class_names)} classes: {', '.join(self.class_names)}")
        
        # Load all image paths and labels
        self.image_paths, self.labels = self._load_image_paths_and_labels()
        
        # Check if we have sufficient data
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
            
        print(f"Found {len(self.image_paths)} images across {len(self.class_names)} classes")
    
    def _load_image_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        """
        Load all image paths and corresponding labels.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        # Supported image extensions
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        
        for class_dir in self.class_dirs:
            class_name = os.path.basename(class_dir)
            class_index = self.class_map[class_name]
            
            # Find all image files in the class directory
            class_images = []
            for ext in extensions:
                class_images.extend(glob.glob(os.path.join(class_dir, ext)))
                class_images.extend(glob.glob(os.path.join(class_dir, "**", ext), recursive=True))
            
            # Add to master lists
            image_paths.extend(class_images)
            labels.extend([class_index] * len(class_images))
            
            print(f"  Class '{class_name}': {len(class_images)} images")
        
        return image_paths, labels
    
    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create train, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        # Split into train+val and test sets
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.image_paths, self.labels, 
            test_size=self.test_split, 
            stratify=self.labels,
            random_state=self.seed
        )
        
        # Split train+val into train and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=self.validation_split / (1 - self.test_split),
            stratify=train_val_labels,
            random_state=self.seed
        )
        
        # Create tf.data.Dataset objects
        train_ds = self._create_dataset(train_paths, train_labels, augment=self.augment)
        val_ds = self._create_dataset(val_paths, val_labels, augment=False)
        test_ds = self._create_dataset(test_paths, test_labels, augment=False)
        
        return train_ds, val_ds, test_ds
    
    def _create_dataset(
        self,
        image_paths: List[str],
        labels: List[int],
        augment: bool = False
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from image paths and labels.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding class labels
            augment: Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset object
        """
        # Create a dataset from the image paths and labels
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Map the filenames to the actual images and preprocess them
        ds = ds.map(
            lambda x, y: self._load_and_preprocess_with_augmentation(x, y, augment),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle, batch, and prefetch the dataset
        ds = ds.shuffle(buffer_size=len(image_paths), seed=self.seed)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def _load_and_preprocess_with_augmentation(
        self,
        image_path: tf.Tensor,
        label: tf.Tensor,
        augment: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess an image with optional augmentation.
        
        Args:
            image_path: Path to image file
            label: Class label
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (preprocessed_image, label)
        """
        # Load and preprocess the image
        image, label = MedicalDatasetLoader.load_and_preprocess_image(
            image_path, self.target_size, label
        )
        
        # Apply augmentation if requested
        if augment:
            # Simple augmentations using tf.image
            # Random horizontal flip
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
            
            # Random brightness/contrast adjustment
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            
            # Ensure pixel values are still in [0, 1]
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        # Count samples per class
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Calculate weights
        total_samples = len(self.labels)
        n_classes = len(self.class_names)
        
        class_weights = {}
        for class_idx in range(n_classes):
            count = class_counts.get(class_idx, 0)
            if count == 0:
                class_weights[class_idx] = 1.0
            else:
                # Inverse frequency weighting
                class_weights[class_idx] = total_samples / (n_classes * count)
        
        return class_weights


class LungCTDatasetLoader(MedicalDatasetLoader):
    """Loader for lung CT datasets, optimized for 3D volumes."""
    
    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 16,  # Smaller batch size for 3D data
        validation_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
        augment: bool = True,
        slice_mode: str = "middle"  # "middle", "all", or "sample"
    ):
        """
        Initialize lung CT dataset loader.
        
        Args:
            slice_mode: How to handle 3D volumes. Options:
                - "middle": Use only middle slice (faster)
                - "all": Use all slices (more data, but slower)
                - "sample": Sample random slices (balanced approach)
        """
        super().__init__(
            data_dir, target_size, batch_size, 
            validation_split, test_split, seed, augment
        )
        
        self.slice_mode = slice_mode
        
        # Get all class directories
        self.class_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, "*")) 
                                  if os.path.isdir(d)])
        
        # Map class names to indices
        self.class_names = [os.path.basename(d) for d in self.class_dirs]
        self.class_map = {name: i for i, name in enumerate(self.class_names)}
        
        print(f"Found {len(self.class_names)} classes: {', '.join(self.class_names)}")
        
        # Load all CT volume paths and labels
        self.volume_paths, self.labels = self._load_volume_paths_and_labels()
        
        # Check if we have sufficient data
        if len(self.volume_paths) == 0:
            raise ValueError(f"No CT volumes found in {data_dir}")
            
        print(f"Found {len(self.volume_paths)} CT volumes across {len(self.class_names)} classes")
    
    def _load_volume_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        """
        Load all CT volume paths and corresponding labels.
        
        Returns:
            Tuple of (volume_paths, labels)
        """
        volume_paths = []
        labels = []
        
        # CT scan directories may contain dicom files or already processed npy files
        extensions = ["*.dcm", "*.nii.gz", "*.nii", "*.npy"]
        
        for class_dir in self.class_dirs:
            class_name = os.path.basename(class_dir)
            class_index = self.class_map[class_name]
            
            # Find all patient directories
            patient_dirs = [d for d in glob.glob(os.path.join(class_dir, "*")) 
                            if os.path.isdir(d)]
            
            # If no patient dirs, assume class_dir contains volumes directly
            if not patient_dirs:
                patient_dirs = [class_dir]
            
            # Process each patient directory
            for patient_dir in patient_dirs:
                # Find all volume files for this patient
                volume_files = []
                for ext in extensions:
                    volume_files.extend(glob.glob(os.path.join(patient_dir, ext)))
                    volume_files.extend(glob.glob(os.path.join(patient_dir, "**", ext), recursive=True))
                
                # For CT scans, we typically store one volume per patient
                # If multiple files, we assume they're slices of the same volume
                if volume_files:
                    # Add the patient directory and label
                    volume_paths.append(patient_dir)
                    labels.append(class_index)
            
            print(f"  Class '{class_name}': {labels.count(class_index)} volumes")
        
        return volume_paths, labels
    
    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create train, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        # Split into train+val and test sets
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.volume_paths, self.labels, 
            test_size=self.test_split, 
            stratify=self.labels,
            random_state=self.seed
        )
        
        # Split train+val into train and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=self.validation_split / (1 - self.test_split),
            stratify=train_val_labels,
            random_state=self.seed
        )
        
        # Create tf.data.Dataset objects
        train_ds = self._create_dataset(train_paths, train_labels, augment=self.augment)
        val_ds = self._create_dataset(val_paths, val_labels, augment=False)
        test_ds = self._create_dataset(test_paths, test_labels, augment=False)
        
        return train_ds, val_ds, test_ds
    
    def _create_dataset(
        self,
        volume_paths: List[str],
        labels: List[int],
        augment: bool = False
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from volume paths and labels.
        
        Args:
            volume_paths: List of volume paths
            labels: List of corresponding class labels
            augment: Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset object
        """
        # Create a dataset from the volume paths and labels
        ds = tf.data.Dataset.from_tensor_slices((volume_paths, labels))
        
        # Map the paths to actual volume slices and preprocess them
        ds = ds.map(
            lambda x, y: self._load_and_preprocess_ct_volume(x, y, augment),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # For the 'all' mode, we need to handle multiple slices per volume
        if self.slice_mode == "all":
            # Unbatch the slices to get individual slices
            ds = ds.unbatch()
        
        # Shuffle, batch, and prefetch the dataset
        ds = ds.shuffle(buffer_size=1000, seed=self.seed)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def _load_and_preprocess_ct_volume(
        self,
        volume_path: tf.Tensor,
        label: tf.Tensor,
        augment: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess a CT volume (or slices).
        
        Args:
            volume_path: Path to CT volume
            label: Class label
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (preprocessed_volume_slices, label)
        """
        # Placeholder function since actual loading would depend on file format
        # In a real implementation, this would load DICOM/NIfTI files
        
        # For demonstration, let's assume we have a mock function to load volumes
        # In practice, you'd use libraries like pydicom, nibabel, etc.
        
        # Simulate loading a volume with a random number of slices (10-50)
        # In a real app, this would be replaced with actual volume loading code
        volume_path_str = volume_path.numpy().decode('utf-8')
        num_slices = np.random.randint(10, 50)
        
        # Create a mock volume of random slices
        volume = tf.random.normal([num_slices, *self.target_size, 1])
        
        # Apply slice selection based on slice_mode
        if self.slice_mode == "middle":
            # Select only the middle slice
            middle_index = num_slices // 2
            selected_slices = volume[middle_index:middle_index+1]
        elif self.slice_mode == "sample":
            # Sample a random subset of slices (e.g., 5 slices)
            num_samples = min(5, num_slices)
            indices = tf.random.shuffle(tf.range(num_slices))[:num_samples]
            selected_slices = tf.gather(volume, indices)
        else:  # "all"
            # Use all slices
            selected_slices = volume
        
        # Apply preprocessing to each slice
        processed_slices = tf.map_fn(
            lambda slice_img: self._preprocess_ct_slice(slice_img, augment),
            selected_slices,
            fn_output_signature=tf.float32
        )
        
        # Replicate the label for each slice (if we have multiple slices)
        labels = tf.repeat(label, tf.shape(processed_slices)[0])
        
        return processed_slices, labels
    
    def _preprocess_ct_slice(
        self,
        slice_img: tf.Tensor,
        augment: bool = False
    ) -> tf.Tensor:
        """
        Preprocess a single CT slice.
        
        Args:
            slice_img: CT slice image
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed CT slice
        """
        # Convert to 3 channels if needed
        if slice_img.shape[-1] == 1:
            slice_img = tf.image.grayscale_to_rgb(slice_img)
        
        # Apply augmentation if requested
        if augment:
            # Random horizontal flip
            if tf.random.uniform(()) > 0.5:
                slice_img = tf.image.flip_left_right(slice_img)
            
            # Random rotation (using tf.image won't give us rotation, but we could use other methods)
            # In practice, you'd implement more domain-specific augmentations
            
            # Random brightness/contrast
            slice_img = tf.image.random_brightness(slice_img, max_delta=0.1)
            slice_img = tf.image.random_contrast(slice_img, lower=0.9, upper=1.1)
            
            # Ensure values are in [0, 1]
            slice_img = tf.clip_by_value(slice_img, 0.0, 1.0)
        
        return slice_img
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        # Count samples per class
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Calculate weights
        total_samples = len(self.labels)
        n_classes = len(self.class_names)
        
        class_weights = {}
        for class_idx in range(n_classes):
            count = class_counts.get(class_idx, 0)
            if count == 0:
                class_weights[class_idx] = 1.0
            else:
                # Inverse frequency weighting
                class_weights[class_idx] = total_samples / (n_classes * count)
        
        return class_weights


class DatasetFactory:
    """Factory class for creating different types of medical datasets."""
    
    @staticmethod
    def create_chest_xray_dataset(
        data_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
        augment: bool = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create chest X-ray datasets for training, validation, and testing.
        
        Args:
            data_dir: Directory containing dataset
            target_size: Target image size (height, width)
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            seed: Random seed for reproducibility
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        loader = ChestXrayDatasetLoader(
            data_dir=data_dir,
            target_size=target_size,
            batch_size=batch_size,
            validation_split=validation_split,
            test_split=test_split,
            seed=seed,
            augment=augment
        )
        
        return loader.create_datasets()
    
    @staticmethod
    def create_lung_ct_dataset(
        data_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 16,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
        augment: bool = True,
        slice_mode: str = "middle"
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create lung CT datasets for training, validation, and testing.
        
        Args:
            data_dir: Directory containing dataset
            target_size: Target image size (height, width)
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            seed: Random seed for reproducibility
            augment: Whether to apply data augmentation
            slice_mode: How to handle 3D volumes (middle, all, sample)
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        loader = LungCTDatasetLoader(
            data_dir=data_dir,
            target_size=target_size,
            batch_size=batch_size,
            validation_split=validation_split,
            test_split=test_split,
            seed=seed,
            augment=augment,
            slice_mode=slice_mode
        )
        
        return loader.create_datasets()
    
    @staticmethod
    def get_class_weights(
        data_dir: str,
        dataset_type: str = "chest_xray"
    ) -> Dict[int, float]:
        """
        Get class weights for a specified dataset.
        
        Args:
            data_dir: Directory containing dataset
            dataset_type: Type of dataset (chest_xray or lung_ct)
            
        Returns:
            Dictionary mapping class indices to weights
        """
        if dataset_type == "chest_xray":
            loader = ChestXrayDatasetLoader(data_dir=data_dir)
        elif dataset_type == "lung_ct":
            loader = LungCTDatasetLoader(data_dir=data_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return loader.get_class_weights()


# Example usage
if __name__ == "__main__":
    print("Medical Imaging Dataset Loader Example")
    
    # Create lung CT dataset
    print("\nLoading Lung CT Dataset:")
    try:
        train_ds, val_ds, test_ds = DatasetFactory.create_lung_ct_dataset()
        print("Lung CT dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading Lung CT dataset: {e}")
    
    # Create chest X-ray dataset
    print("\nLoading Chest X-ray Dataset:")
    try:
        train_ds, val_ds, test_ds = DatasetFactory.create_chest_xray_dataset()
        print("Chest X-ray dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading Chest X-ray dataset: {e}") 