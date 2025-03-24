"""
Medical Imaging Analysis Pipeline

This module provides a pipeline for analyzing medical images using
computer vision techniques and enhancing the analysis with LLM-based
interpretations and insights.
"""

import os
import logging
import json
import time
import base64
from typing import Dict, List, Optional, Union, Any, Tuple
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ExifTags
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalImageProcessor:
    """
    Handles preprocessing of medical images for analysis.
    
    This class provides methods for loading, standardizing, and enhancing
    medical images from various modalities (X-ray, MRI, CT, etc.) to prepare
    them for analysis by computer vision models.
    """
    
    # Supported image types and their typical dimensions/characteristics
    MODALITY_SPECS = {
        "xray": {"target_size": (512, 512), "color_mode": "grayscale"},
        "mri": {"target_size": (256, 256), "color_mode": "grayscale"},
        "ct": {"target_size": (512, 512), "color_mode": "grayscale"},
        "ultrasound": {"target_size": (256, 256), "color_mode": "grayscale"},
        "pathology": {"target_size": (512, 512), "color_mode": "rgb"},
        "dermatology": {"target_size": (299, 299), "color_mode": "rgb"},
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the medical image processor.
        
        Args:
            cache_dir: Directory to cache processed images
        """
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Created image cache directory: {cache_dir}")
    
    def load_image(self, image_path: Union[str, Path, BytesIO]) -> np.ndarray:
        """
        Load an image from a file path or BytesIO object.
        
        Args:
            image_path: Path to the image file or BytesIO object
            
        Returns:
            Numpy array containing the image data
        """
        try:
            if isinstance(image_path, (str, Path)):
                # Load from file path
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Use OpenCV for better DICOM and medical format handling
                img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                
                # Convert from BGR to RGB if it's a color image
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                logger.info(f"Loaded image from {image_path}, shape: {img.shape}")
                return img
                
            elif isinstance(image_path, BytesIO):
                # Load from BytesIO
                image_path.seek(0)
                pil_img = Image.open(image_path)
                img = np.array(pil_img)
                logger.info(f"Loaded image from BytesIO, shape: {img.shape}")
                return img
                
            else:
                raise TypeError("image_path must be a string path or BytesIO object")
                
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    def preprocess_image(
        self,
        image: np.ndarray,
        modality: str = "xray",
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        enhance_contrast: bool = False
    ) -> np.ndarray:
        """
        Preprocess a medical image for analysis.
        
        Args:
            image: Numpy array containing the image data
            modality: Medical image modality (xray, mri, ct, etc.)
            target_size: Target dimensions (height, width)
            normalize: Whether to normalize pixel values
            enhance_contrast: Whether to apply contrast enhancement
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Get modality specifications or use defaults
            modality = modality.lower()
            modality_specs = self.MODALITY_SPECS.get(
                modality, 
                {"target_size": (299, 299), "color_mode": "rgb"}
            )
            
            # Use provided target size or default from modality specs
            if target_size is None:
                target_size = modality_specs["target_size"]
            
            # Convert to grayscale if needed
            if modality_specs["color_mode"] == "grayscale" and len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Resize the image
            image = cv2.resize(image, target_size)
            
            # Enhance contrast if requested
            if enhance_contrast:
                if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    image = clahe.apply(image.astype(np.uint8))
                else:  # Color image
                    # Convert to LAB color space
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    # Apply CLAHE to L channel
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    # Merge the channels
                    lab = cv2.merge((l, a, b))
                    # Convert back to RGB
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Normalize pixel values if requested
            if normalize:
                if image.dtype != np.float32:
                    image = image.astype(np.float32)
                
                # Normalize to [0, 1]
                if image.max() > 1.0:
                    image = image / 255.0
            
            logger.info(f"Preprocessed {modality} image to shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def extract_image_metadata(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from a medical image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing image metadata
        """
        metadata = {
            "filename": os.path.basename(str(image_path)),
            "file_size": os.path.getsize(image_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat(),
            "format": os.path.splitext(image_path)[1].lower()[1:],
        }
        
        try:
            # Try to extract EXIF data if available
            img = Image.open(image_path)
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = {
                    ExifTags.TAGS[k]: v
                    for k, v in img._getexif().items()
                    if k in ExifTags.TAGS
                }
                metadata["exif"] = exif
            
            # Extract image dimensions
            metadata["width"], metadata["height"] = img.size
            metadata["channels"] = len(img.getbands())
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract detailed metadata: {str(e)}")
            # Return basic metadata if detailed extraction fails
            return metadata
    
    def image_to_base64(self, image: np.ndarray, format: str = "png") -> str:
        """
        Convert an image array to base64 string.
        
        Args:
            image: Image as numpy array
            format: Output image format (jpg, png)
            
        Returns:
            Base64 encoded string of the image
        """
        # Convert float images to uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create PIL Image
        pil_img = Image.fromarray(image)
        
        # Save to BytesIO
        buffered = BytesIO()
        pil_img.save(buffered, format=format)
        
        # Encode to base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


class MedicalVisionModel:
    """
    Encapsulates computer vision models for medical image analysis.
    
    This class provides methods to load and run various medical imaging
    analysis models for tasks like organ segmentation, anomaly detection,
    and classification of medical conditions.
    """
    
    def __init__(
        self,
        model_type: str = "general",
        models_dir: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the medical vision model.
        
        Args:
            model_type: Type of model to load (general, segmentation, classification)
            models_dir: Directory containing model files
            use_gpu: Whether to use GPU acceleration if available
        """
        self.model_type = model_type
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
        self.use_gpu = use_gpu and self._is_gpu_available()
        self.models = {}
        
        # Make sure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"Initialized medical vision model, type: {model_type}, GPU: {self.use_gpu}")
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for model inference."""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific medical imaging model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Boolean indicating if model was successfully loaded
        """
        # This is a placeholder implementation
        # In a real application, this would load specific models based on name and type
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Example implementation for a segmentation model using OpenCV DNN
            if self.model_type == "segmentation":
                # Path to the model files
                model_path = os.path.join(self.models_dir, f"{model_name}.pb")
                config_path = os.path.join(self.models_dir, f"{model_name}.pbtxt")
                
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return False
                
                # Load the model
                net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                
                # Use GPU if available and requested
                if self.use_gpu:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                
                self.models[model_name] = net
                logger.info(f"Successfully loaded segmentation model: {model_name}")
                return True
                
            elif self.model_type == "classification":
                # Similar implementation for classification models
                pass
                
            else:
                logger.warning(f"Unsupported model type: {self.model_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def detect_abnormalities(
        self,
        image: np.ndarray,
        modality: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect abnormalities in a medical image.
        
        Args:
            image: Preprocessed image as numpy array
            modality: Medical image modality
            threshold: Detection confidence threshold
            
        Returns:
            Dictionary with detection results
        """
        # This is a placeholder implementation
        # In a real application, this would run the appropriate model for the modality
        
        try:
            logger.info(f"Running abnormality detection on {modality} image")
            
            # Simulate a detection process
            time.sleep(0.5)  # Simulating processing time
            
            # Return dummy results
            return {
                "detected": True,
                "abnormalities": [
                    {
                        "type": "nodule",
                        "confidence": 0.85,
                        "location": [120, 150, 30, 30],  # x, y, width, height
                        "description": "Potential nodule detected in upper right quadrant"
                    }
                ],
                "processing_time": 0.5
            }
            
        except Exception as e:
            logger.error(f"Error in abnormality detection: {str(e)}")
            return {"detected": False, "error": str(e)}
    
    def segment_organs(
        self,
        image: np.ndarray,
        modality: str,
        organs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Segment organs in a medical image.
        
        Args:
            image: Preprocessed image as numpy array
            modality: Medical image modality
            organs: List of organs to segment (None = all)
            
        Returns:
            Dictionary with segmentation masks and metadata
        """
        # This is a placeholder implementation
        try:
            logger.info(f"Running organ segmentation on {modality} image")
            
            # Create a blank mask of the same size as the input image
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Simulate segmentation process
            time.sleep(0.5)  # Simulating processing time
            
            # Create a dummy segmentation mask
            if modality == "xray" and (organs is None or "lungs" in organs):
                # Simulate lung segmentation for X-rays
                center_x, center_y = width // 2, height // 2
                # Left lung
                cv2.ellipse(mask, (center_x - width//6, center_y), 
                           (width//8, height//3), 0, 0, 360, 1, -1)
                # Right lung
                cv2.ellipse(mask, (center_x + width//6, center_y), 
                           (width//8, height//3), 0, 0, 360, 2, -1)
            
            # Return segmentation results
            return {
                "success": True,
                "mask": mask,
                "organs_found": ["lungs"] if modality == "xray" else [],
                "processing_time": 0.5
            }
            
        except Exception as e:
            logger.error(f"Error in organ segmentation: {str(e)}")
            return {"success": False, "error": str(e)}


class MedicalImagingLLMPipeline:
    """
    Pipeline combining medical image analysis with LLM-based interpretation.
    
    This class orchestrates the process of analyzing medical images using
    computer vision techniques and then enhancing and explaining the results
    using large language models for improved clinical relevance.
    """
    
    def __init__(self, llm_connector=None, cache_dir: Optional[str] = None):
        """
        Initialize the medical imaging LLM pipeline.
        
        Args:
            llm_connector: Instance of an LLM connector for text generation
            cache_dir: Directory to cache intermediate results
        """
        self.image_processor = MedicalImageProcessor(cache_dir=cache_dir)
        self.vision_model = MedicalVisionModel()
        self.llm_connector = llm_connector
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(os.path.join(cache_dir, "results"), exist_ok=True)
        
        logger.info("Initialized medical imaging LLM pipeline")
    
    def set_llm_connector(self, llm_connector) -> None:
        """
        Set the LLM connector for the pipeline.
        
        Args:
            llm_connector: Instance of an LLM connector for text generation
        """
        self.llm_connector = llm_connector
        logger.info("Set LLM connector for the pipeline")
    
    def analyze_image(
        self,
        image_path: Union[str, Path, BytesIO],
        modality: str,
        analysis_type: str = "general",
        clinical_context: Optional[str] = None,
        patient_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a medical image and generate an interpretation.
        
        Args:
            image_path: Path to the image file or BytesIO object
            modality: Medical image modality (xray, mri, ct, etc.)
            analysis_type: Type of analysis to perform (general, detailed, etc.)
            clinical_context: Additional clinical context for LLM
            patient_info: Patient information for context
            
        Returns:
            Dictionary with analysis results and interpretation
        """
        start_time = time.time()
        results = {
            "success": False,
            "modality": modality,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Load and preprocess the image
            logger.info(f"Starting analysis of {modality} image")
            image = self.image_processor.load_image(image_path)
            processed_image = self.image_processor.preprocess_image(
                image, modality=modality, enhance_contrast=True
            )
            
            # Step 2: Run appropriate computer vision analysis
            if analysis_type == "abnormality_detection":
                vision_results = self.vision_model.detect_abnormalities(processed_image, modality)
            elif analysis_type == "segmentation":
                vision_results = self.vision_model.segment_organs(processed_image, modality)
            else:
                # Run both analyses for general analysis
                abnormality_results = self.vision_model.detect_abnormalities(processed_image, modality)
                segmentation_results = self.vision_model.segment_organs(processed_image, modality)
                vision_results = {
                    "abnormality_detection": abnormality_results,
                    "segmentation": segmentation_results
                }
            
            results["vision_analysis"] = vision_results
            
            # Step 3: Prepare data for LLM interpretation
            if not self.llm_connector:
                logger.warning("No LLM connector provided, skipping LLM interpretation")
                results["llm_interpretation"] = None
            else:
                # Create a prompt for the LLM based on vision results
                prompt = self._create_llm_prompt(
                    vision_results, modality, analysis_type, clinical_context, patient_info
                )
                
                # Generate LLM interpretation
                llm_response = self.llm_connector.generate_response(
                    query=prompt,
                    system_prompt="You are a medical imaging AI assistant. Analyze the provided results from computer vision models and generate a clinical interpretation."
                )
                
                if llm_response.get("success", False):
                    results["llm_interpretation"] = {
                        "text": llm_response.get("text", ""),
                        "model": llm_response.get("model", "")
                    }
                else:
                    logger.error(f"LLM generation failed: {llm_response.get('error', 'Unknown error')}")
                    results["llm_interpretation"] = {"error": llm_response.get("error", "LLM generation failed")}
            
            # Calculate processing time
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            results["success"] = True
            
            # Cache results if cache_dir is set
            if self.cache_dir:
                self._cache_results(results, image_path)
            
            logger.info(f"Completed analysis in {processing_time:.2f} seconds")
            logger.debug(f"Analysis result: {json.dumps(results, indent=2)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in image analysis pipeline: {str(e)}")
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
            return results
    
    def _create_llm_prompt(
        self,
        vision_results: Dict[str, Any],
        modality: str,
        analysis_type: str,
        clinical_context: Optional[str] = None,
        patient_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a prompt for the LLM based on vision model results.
        
        Args:
            vision_results: Results from the vision model analysis
            modality: Medical image modality
            analysis_type: Type of analysis performed
            clinical_context: Additional clinical context
            patient_info: Patient information
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt_parts = []
        
        # Add general instruction
        prompt_parts.append(f"Please analyze the following {modality} image results:")
        
        # Add vision model results
        prompt_parts.append("Computer Vision Analysis Results:")
        prompt_parts.append(json.dumps(vision_results, indent=2))
        
        # Add clinical context if provided
        if clinical_context:
            prompt_parts.append("Clinical Context:")
            prompt_parts.append(clinical_context)
        
        # Add patient information if provided
        if patient_info:
            prompt_parts.append("Patient Information:")
            prompt_parts.append(json.dumps(patient_info, indent=2))
        
        # Add specific instructions based on analysis type
        if analysis_type == "abnormality_detection":
            prompt_parts.append("Please provide a detailed interpretation of the detected abnormalities, their clinical significance, and recommended next steps.")
        elif analysis_type == "segmentation":
            prompt_parts.append("Please analyze the organ segmentation results and provide insights on organ structure, possible abnormalities, and clinical relevance.")
        else:
            prompt_parts.append("Please provide a comprehensive interpretation of all findings, their clinical significance, and recommended next steps.")
        
        # Combine all parts into a single prompt
        return "\n\n".join(prompt_parts)
    
    def _cache_results(self, results: Dict[str, Any], image_path: Union[str, Path, BytesIO]) -> None:
        """
        Cache analysis results to disk.
        
        Args:
            results: Analysis results to cache
            image_path: Original image path or BytesIO object
        """
        try:
            if not self.cache_dir:
                return
                
            # Generate a filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if isinstance(image_path, (str, Path)):
                image_basename = os.path.basename(str(image_path))
            else:
                image_basename = f"image_{timestamp}"
            
            # Create cache filename
            cache_file = os.path.join(
                self.cache_dir, 
                "results", 
                f"{image_basename.split('.')[0]}_{timestamp}.json"
            )
            
            # Write results to cache file
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Cached analysis results to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error caching results: {str(e)}")


# Example usage
if __name__ == "__main__":
    # This section is for testing and demonstration purposes
    from ai.llm.model_connector import MedicalLLMConnector
    
    # Initialize the LLM connector (example)
    llm = MedicalLLMConnector()
    llm.connect("openai", "gpt-4", os.environ.get("OPENAI_API_KEY"))
    
    # Initialize the pipeline
    pipeline = MedicalImagingLLMPipeline(llm_connector=llm, cache_dir="./cache")
    
    # Example image analysis
    result = pipeline.analyze_image(
        image_path="./test_images/chest_xray.jpg",
        modality="xray",
        analysis_type="general",
        clinical_context="Patient presented with persistent cough for 3 weeks"
    )
    
    logger.debug(f"Analysis result: {json.dumps(result, indent=2)}") 