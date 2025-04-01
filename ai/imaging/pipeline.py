"""
MediNex AI Medical Imaging Pipeline

This module implements the medical imaging pipeline for processing and analyzing medical images.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from ..llm.model_connector import MedicalLLMConnector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImageMetadata:
    """Medical image metadata."""
    modality: str  # e.g., "X-ray", "MRI", "CT"
    body_part: str
    orientation: str
    acquisition_date: Optional[str] = None
    study_id: Optional[str] = None
    series_id: Optional[str] = None
    equipment_info: Optional[Dict[str, str]] = None

@dataclass
class ImageAnalysisResult:
    """Results from image analysis."""
    findings: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    annotations: Optional[Dict[str, Any]] = None
    llm_interpretation: Optional[str] = None

class MedicalImagePipeline:
    """
    Medical imaging pipeline for processing and analyzing medical images.
    """
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        model_path: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the medical imaging pipeline.
        
        Args:
            llm_config: Configuration for the LLM
            model_path: Path to the pre-trained model
            device: Device to run the model on ("cuda" or "cpu")
            cache_dir: Directory for caching
        """
        self.llm = MedicalLLMConnector(llm_config)
        self.cache_dir = cache_dir
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load models
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Create cache directory if needed
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def load_image(
        self,
        image_path: str,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Load and preprocess a medical image.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing
            
        Returns:
            Tuple of (numpy array, PIL Image)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array, image
    
    def extract_metadata(
        self,
        image_path: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> ImageMetadata:
        """
        Extract metadata from a medical image.
        
        Args:
            image_path: Path to the image file
            additional_info: Additional metadata
            
        Returns:
            ImageMetadata object
        """
        # TODO: Implement DICOM metadata extraction
        # For now, return basic metadata
        metadata = ImageMetadata(
            modality="Unknown",
            body_part="Unknown",
            orientation="Unknown"
        )
        
        if additional_info:
            # Update metadata with additional info
            for key, value in additional_info.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
        
        return metadata
    
    def preprocess_image(
        self,
        image: np.ndarray,
        normalize: bool = True,
        enhance_contrast: bool = True
    ) -> np.ndarray:
        """
        Preprocess a medical image.
        
        Args:
            image: Input image array
            normalize: Whether to normalize pixel values
            enhance_contrast: Whether to enhance contrast
            
        Returns:
            Preprocessed image array
        """
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if enhance_contrast:
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image.astype(np.uint8))
        
        if normalize:
            # Normalize to [0,1]
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def analyze_image(
        self,
        image_path: str,
        metadata: Optional[ImageMetadata] = None
    ) -> ImageAnalysisResult:
        """
        Analyze a medical image.
        
        Args:
            image_path: Path to the image file
            metadata: Optional image metadata
            
        Returns:
            ImageAnalysisResult object
        """
        # Load and preprocess image
        image_array, pil_image = self.load_image(image_path)
        processed_image = self.preprocess_image(image_array)
        
        # Prepare image for model
        inputs = self.image_processor(
            pil_image,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predictions and confidence scores
        predictions = []
        confidence_scores = {}
        
        for idx in torch.argsort(probs[0], descending=True)[:5]:
            label = self.model.config.id2label[idx.item()]
            score = probs[0][idx].item()
            
            predictions.append({
                "label": label,
                "confidence": score
            })
            confidence_scores[label] = score
        
        # Generate LLM interpretation
        context = (
            f"Analyzing a medical image with the following findings:\n"
            f"{', '.join(f'{p['label']} ({p['confidence']:.2%})' for p in predictions)}\n"
            f"\nMetadata:\n"
            f"Modality: {metadata.modality if metadata else 'Unknown'}\n"
            f"Body Part: {metadata.body_part if metadata else 'Unknown'}\n"
            f"Please provide a detailed medical interpretation of these findings."
        )
        
        llm_interpretation = self.llm.generate_text(context)
        
        return ImageAnalysisResult(
            findings=predictions,
            confidence_scores=confidence_scores,
            annotations=None,  # TODO: Implement region annotations
            llm_interpretation=llm_interpretation
        )
    
    def batch_analyze(
        self,
        image_paths: List[str],
        metadata_list: Optional[List[ImageMetadata]] = None
    ) -> List[ImageAnalysisResult]:
        """
        Analyze multiple medical images in batch.
        
        Args:
            image_paths: List of image file paths
            metadata_list: Optional list of image metadata
            
        Returns:
            List of ImageAnalysisResult objects
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            metadata = metadata_list[i] if metadata_list else None
            result = self.analyze_image(image_path, metadata)
            results.append(result)
        
        return results
    
    def save_results(
        self,
        results: ImageAnalysisResult,
        output_dir: str,
        image_path: str
    ) -> str:
        """
        Save analysis results.
        
        Args:
            results: Analysis results
            output_dir: Output directory
            image_path: Original image path
            
        Returns:
            Path to saved results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_analysis.json")
        
        # Prepare results for saving
        save_data = {
            "image_path": image_path,
            "findings": results.findings,
            "confidence_scores": results.confidence_scores,
            "llm_interpretation": results.llm_interpretation
        }
        
        if results.annotations:
            save_data["annotations"] = results.annotations
        
        # Save results
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        
        return output_path 