"""
MediNex AI Model Distribution

This module provides functionality for packaging, distributing, and monitoring
MediNex AI models to different deployment targets and users.
"""

import os
import json
import logging
import time
import uuid
import shutil
import zipfile
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelVersion:
    """
    Represents a specific version of a MediNex AI model.
    """
    
    def __init__(
        self,
        version_id: str,
        model_id: str,
        version_number: str,
        description: str,
        artifacts_path: str,
        config: Dict[str, Any],
        created_at: Optional[str] = None,
        contributors: Optional[List[str]] = None,
        status: str = "draft"
    ):
        """
        Initialize a model version.
        
        Args:
            version_id: Unique identifier for this version
            model_id: Parent model identifier
            version_number: Semantic version number (e.g., "1.0.0")
            description: Description of this version
            artifacts_path: Path to model artifacts
            config: Model configuration
            created_at: Creation timestamp (ISO format)
            contributors: List of contributor IDs
            status: Version status ("draft", "released", "deprecated")
        """
        self.version_id = version_id
        self.model_id = model_id
        self.version_number = version_number
        self.description = description
        self.artifacts_path = artifacts_path
        self.config = config
        self.created_at = created_at or datetime.now().isoformat()
        self.contributors = contributors or []
        self.status = status
        self.checksum = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this version
        """
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "version_number": self.version_number,
            "description": self.description,
            "artifacts_path": self.artifacts_path,
            "config": self.config,
            "created_at": self.created_at,
            "contributors": self.contributors,
            "status": self.status,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """
        Create a ModelVersion from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ModelVersion instance
        """
        version = cls(
            version_id=data["version_id"],
            model_id=data["model_id"],
            version_number=data["version_number"],
            description=data["description"],
            artifacts_path=data["artifacts_path"],
            config=data["config"],
            created_at=data["created_at"],
            contributors=data["contributors"],
            status=data["status"]
        )
        version.checksum = data.get("checksum")
        return version


class ModelDistributor:
    """
    Manages packaging, distribution, and monitoring of MediNex AI models.
    """
    
    def __init__(self, storage_path: str = "./data/models"):
        """
        Initialize the model distributor.
        
        Args:
            storage_path: Path to store model data
        """
        self.storage_path = storage_path
        
        # Create storage directories
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "versions"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "packages"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "deployments"), exist_ok=True)
        
        # Path to model and version files
        self.models_file = os.path.join(storage_path, "models.json")
        self.versions_dir = os.path.join(storage_path, "versions")
        self.packages_dir = os.path.join(storage_path, "packages")
        self.deployments_file = os.path.join(storage_path, "deployments.json")
        self.licenses_file = os.path.join(storage_path, "licenses.json")
        
        # Load existing data or initialize empty data structures
        self.models = self._load_models()
        self.deployments = self._load_deployments()
        self.licenses = self._load_licenses()
        
        logger.info("Initialized model distributor")
    
    def _load_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Load models from storage.
        
        Returns:
            Dictionary of models
        """
        if os.path.exists(self.models_file):
            try:
                with open(self.models_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                return {}
        else:
            return {}
    
    def _load_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific version from storage.
        
        Args:
            version_id: Version ID
            
        Returns:
            Version data or None if not found
        """
        version_file = os.path.join(self.versions_dir, f"{version_id}.json")
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading version {version_id}: {str(e)}")
                return None
        else:
            return None
    
    def _load_deployments(self) -> Dict[str, Dict[str, Any]]:
        """
        Load deployments from storage.
        
        Returns:
            Dictionary of deployments
        """
        if os.path.exists(self.deployments_file):
            try:
                with open(self.deployments_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading deployments: {str(e)}")
                return {}
        else:
            return {}
    
    def _load_licenses(self) -> Dict[str, Dict[str, Any]]:
        """
        Load licenses from storage.
        
        Returns:
            Dictionary of licenses
        """
        if os.path.exists(self.licenses_file):
            try:
                with open(self.licenses_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading licenses: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_models(self) -> bool:
        """
        Save models to storage.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            with open(self.models_file, 'w') as f:
                json.dump(self.models, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def _save_version(self, version: ModelVersion) -> bool:
        """
        Save a version to storage.
        
        Args:
            version: ModelVersion instance
            
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            version_file = os.path.join(self.versions_dir, f"{version.version_id}.json")
            with open(version_file, 'w') as f:
                json.dump(version.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving version {version.version_id}: {str(e)}")
            return False
    
    def _save_deployments(self) -> bool:
        """
        Save deployments to storage.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            with open(self.deployments_file, 'w') as f:
                json.dump(self.deployments, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving deployments: {str(e)}")
            return False
    
    def _save_licenses(self) -> bool:
        """
        Save licenses to storage.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            with open(self.licenses_file, 'w') as f:
                json.dump(self.licenses, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving licenses: {str(e)}")
            return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Checksum as a hexadecimal string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_model(
        self,
        name: str,
        description: str,
        model_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a new model.
        
        Args:
            name: Model name
            description: Model description
            model_type: Type of model (e.g., "rag", "llm", "imaging")
            metadata: Additional metadata
            
        Returns:
            Model record
        """
        # Generate a unique ID for the model
        model_id = str(uuid.uuid4())
        
        # Create model record
        model = {
            "id": model_id,
            "name": name,
            "description": description,
            "type": model_type,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "versions": []
        }
        
        # Add to models dictionary
        self.models[model_id] = model
        
        # Save changes
        self._save_models()
        
        logger.info(f"Registered new model: {name} (ID: {model_id})")
        
        return model
    
    def create_version(
        self,
        model_id: str,
        version_number: str,
        description: str,
        artifacts_path: str,
        config: Dict[str, Any],
        contributors: Optional[List[str]] = None
    ) -> Optional[ModelVersion]:
        """
        Create a new version of a model.
        
        Args:
            model_id: Model ID
            version_number: Semantic version number
            description: Version description
            artifacts_path: Path to model artifacts
            config: Model configuration
            contributors: List of contributor IDs
            
        Returns:
            ModelVersion instance or None if failed
        """
        if model_id not in self.models:
            logger.error(f"Model not found: {model_id}")
            return None
        
        if not os.path.exists(artifacts_path):
            logger.error(f"Artifacts path not found: {artifacts_path}")
            return None
        
        # Generate a unique ID for the version
        version_id = str(uuid.uuid4())
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            version_number=version_number,
            description=description,
            artifacts_path=artifacts_path,
            config=config,
            contributors=contributors or []
        )
        
        # Save version
        self._save_version(version)
        
        # Update model record
        self.models[model_id]["versions"].append(version_id)
        self.models[model_id]["updated_at"] = datetime.now().isoformat()
        self._save_models()
        
        logger.info(f"Created new version {version_number} for model {model_id} (Version ID: {version_id})")
        
        return version
    
    def release_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        Release a version, making it available for distribution.
        
        Args:
            version_id: Version ID
            
        Returns:
            Updated ModelVersion or None if failed
        """
        version_data = self._load_version(version_id)
        if not version_data:
            logger.error(f"Version not found: {version_id}")
            return None
        
        version = ModelVersion.from_dict(version_data)
        
        # Update status
        version.status = "released"
        
        # Save changes
        self._save_version(version)
        
        logger.info(f"Released version {version.version_number} (ID: {version_id})")
        
        return version
    
    def deprecate_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        Deprecate a version, marking it as no longer recommended.
        
        Args:
            version_id: Version ID
            
        Returns:
            Updated ModelVersion or None if failed
        """
        version_data = self._load_version(version_id)
        if not version_data:
            logger.error(f"Version not found: {version_id}")
            return None
        
        version = ModelVersion.from_dict(version_data)
        
        # Update status
        version.status = "deprecated"
        
        # Save changes
        self._save_version(version)
        
        logger.info(f"Deprecated version {version.version_number} (ID: {version_id})")
        
        return version
    
    def package_version(
        self,
        version_id: str,
        package_name: Optional[str] = None,
        include_config: bool = True,
        include_readme: bool = True
    ) -> Optional[str]:
        """
        Package a model version for distribution.
        
        Args:
            version_id: Version ID
            package_name: Custom package name (default: auto-generated)
            include_config: Whether to include configuration
            include_readme: Whether to include README
            
        Returns:
            Path to the package or None if failed
        """
        version_data = self._load_version(version_id)
        if not version_data:
            logger.error(f"Version not found: {version_id}")
            return None
        
        version = ModelVersion.from_dict(version_data)
        model = self.models.get(version.model_id)
        
        if not model:
            logger.error(f"Model not found: {version.model_id}")
            return None
        
        # Generate package name if not provided
        if not package_name:
            sanitized_name = model["name"].lower().replace(" ", "_")
            package_name = f"{sanitized_name}-{version.version_number}"
        
        # Create package directory
        package_dir = os.path.join(self.packages_dir, f"{package_name}_temp")
        os.makedirs(package_dir, exist_ok=True)
        
        try:
            # Copy artifacts
            artifacts_dir = os.path.join(package_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            if os.path.isdir(version.artifacts_path):
                # Copy directory contents
                for item in os.listdir(version.artifacts_path):
                    source = os.path.join(version.artifacts_path, item)
                    dest = os.path.join(artifacts_dir, item)
                    if os.path.isdir(source):
                        shutil.copytree(source, dest)
                    else:
                        shutil.copy2(source, dest)
            else:
                # Copy single file
                shutil.copy2(version.artifacts_path, artifacts_dir)
            
            # Include config if requested
            if include_config:
                with open(os.path.join(package_dir, "config.json"), "w") as f:
                    json.dump(version.config, f, indent=2)
            
            # Include metadata
            metadata = {
                "model_name": model["name"],
                "model_description": model["description"],
                "model_type": model["type"],
                "version": version.version_number,
                "version_description": version.description,
                "created_at": version.created_at,
                "contributors": version.contributors
            }
            
            with open(os.path.join(package_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Include README if requested
            if include_readme:
                readme_content = f"""# {model["name"]} - v{version.version_number}

{model["description"]}

## Version Information

{version.description}

## Model Type

{model["type"]}

## Usage

This model package is part of the MediNex AI system and should be used
according to the license terms.

## Contributors

This model was created with contributions from {len(version.contributors)} contributors.
"""
                with open(os.path.join(package_dir, "README.md"), "w") as f:
                    f.write(readme_content)
            
            # Create ZIP package
            zip_path = os.path.join(self.packages_dir, f"{package_name}.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, package_dir)
                        zipf.write(file_path, arcname)
            
            # Calculate checksum
            checksum = self._calculate_checksum(zip_path)
            
            # Update version with checksum
            version.checksum = checksum
            self._save_version(version)
            
            # Clean up temp directory
            shutil.rmtree(package_dir)
            
            logger.info(f"Packaged version {version.version_number} (ID: {version_id}) as {zip_path}")
            
            return zip_path
            
        except Exception as e:
            logger.error(f"Error packaging version {version_id}: {str(e)}")
            # Clean up on error
            if os.path.exists(package_dir):
                shutil.rmtree(package_dir)
            return None
    
    def create_license(
        self,
        version_id: str,
        user_id: str,
        license_type: str,
        expiration_date: Optional[str] = None,
        usage_limits: Optional[Dict[str, Any]] = None,
        custom_terms: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a license for a model version.
        
        Args:
            version_id: Version ID
            user_id: User or organization ID
            license_type: Type of license ("standard", "academic", "commercial", "evaluation")
            expiration_date: Expiration date (ISO format, None for perpetual)
            usage_limits: Usage limitations
            custom_terms: Custom license terms
            
        Returns:
            License record
        """
        version_data = self._load_version(version_id)
        if not version_data:
            logger.error(f"Version not found: {version_id}")
            return {}
        
        # Generate a unique ID for the license
        license_id = str(uuid.uuid4())
        
        # Generate license key
        license_key = hashlib.sha256(f"{license_id}:{user_id}:{version_id}:{time.time()}".encode()).hexdigest()
        
        # Create license record
        license_record = {
            "id": license_id,
            "license_key": license_key,
            "version_id": version_id,
            "user_id": user_id,
            "type": license_type,
            "expiration_date": expiration_date,
            "usage_limits": usage_limits or {},
            "custom_terms": custom_terms,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_verified": None,
            "usage_stats": {
                "installations": 0,
                "activations": 0,
                "queries": 0,
                "last_used": None
            }
        }
        
        # Add to licenses dictionary
        self.licenses[license_id] = license_record
        
        # Save changes
        self._save_licenses()
        
        logger.info(f"Created license for version {version_id} and user {user_id} (License ID: {license_id})")
        
        return license_record
    
    def verify_license(
        self,
        license_key: str,
        version_id: str
    ) -> Dict[str, Any]:
        """
        Verify a license for a model version.
        
        Args:
            license_key: License key
            version_id: Version ID
            
        Returns:
            Dictionary with verification results
        """
        # Find license by key
        license_record = None
        license_id = None
        
        for lid, lic in self.licenses.items():
            if lic["license_key"] == license_key:
                license_record = lic
                license_id = lid
                break
        
        if not license_record:
            logger.warning(f"License not found for key: {license_key}")
            return {
                "valid": False,
                "reason": "license_not_found",
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if license is for the requested version
        if license_record["version_id"] != version_id:
            logger.warning(f"License {license_id} is not valid for version {version_id}")
            return {
                "valid": False,
                "reason": "version_mismatch",
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if license is active
        if license_record["status"] != "active":
            logger.warning(f"License {license_id} is not active (status: {license_record['status']})")
            return {
                "valid": False,
                "reason": "license_inactive",
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if license has expired
        if license_record["expiration_date"]:
            expiration = datetime.fromisoformat(license_record["expiration_date"])
            if datetime.now() > expiration:
                logger.warning(f"License {license_id} has expired on {license_record['expiration_date']}")
                return {
                    "valid": False,
                    "reason": "license_expired",
                    "timestamp": datetime.now().isoformat()
                }
        
        # Check usage limits if applicable
        usage_limits = license_record["usage_limits"]
        usage_stats = license_record["usage_stats"]
        
        if usage_limits.get("max_queries") and usage_stats["queries"] >= usage_limits["max_queries"]:
            logger.warning(f"License {license_id} has reached query limit")
            return {
                "valid": False,
                "reason": "query_limit_reached",
                "timestamp": datetime.now().isoformat()
            }
        
        # Update usage statistics
        license_record["last_verified"] = datetime.now().isoformat()
        license_record["usage_stats"]["activations"] += 1
        
        # Save changes
        self._save_licenses()
        
        logger.info(f"License {license_id} verified successfully for version {version_id}")
        
        return {
            "valid": True,
            "license_id": license_id,
            "license_type": license_record["type"],
            "user_id": license_record["user_id"],
            "expiration_date": license_record["expiration_date"],
            "usage_limits": usage_limits,
            "timestamp": datetime.now().isoformat()
        }
    
    def record_usage(self, license_id: str, usage_type: str = "query") -> bool:
        """
        Record usage of a licensed model.
        
        Args:
            license_id: License ID
            usage_type: Type of usage (e.g., "query", "training")
            
        Returns:
            True if successful, False otherwise
        """
        if license_id not in self.licenses:
            logger.error(f"License not found: {license_id}")
            return False
        
        license_record = self.licenses[license_id]
        
        # Update usage statistics
        license_record["usage_stats"]["last_used"] = datetime.now().isoformat()
        
        if usage_type == "query":
            license_record["usage_stats"]["queries"] += 1
        elif usage_type == "installation":
            license_record["usage_stats"]["installations"] += 1
        
        # Save changes
        self._save_licenses()
        
        logger.debug(f"Recorded {usage_type} usage for license {license_id}")
        
        return True
    
    def register_deployment(
        self,
        version_id: str,
        license_id: str,
        deployment_name: str,
        environment: str,
        endpoint_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a deployment of a model version.
        
        Args:
            version_id: Version ID
            license_id: License ID
            deployment_name: Name of the deployment
            environment: Deployment environment
            endpoint_url: URL of the deployed endpoint
            metadata: Additional metadata
            
        Returns:
            Deployment record
        """
        version_data = self._load_version(version_id)
        if not version_data:
            logger.error(f"Version not found: {version_id}")
            return {}
        
        if license_id not in self.licenses:
            logger.error(f"License not found: {license_id}")
            return {}
        
        # Generate a unique ID for the deployment
        deployment_id = str(uuid.uuid4())
        
        # Create deployment record
        deployment = {
            "id": deployment_id,
            "version_id": version_id,
            "license_id": license_id,
            "name": deployment_name,
            "environment": environment,
            "endpoint_url": endpoint_url,
            "metadata": metadata or {},
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat(),
            "usage_stats": {
                "total_queries": 0,
                "avg_response_time_ms": 0,
                "uptime_percentage": 100
            }
        }
        
        # Add to deployments dictionary
        self.deployments[deployment_id] = deployment
        
        # Record installation usage for the license
        self.record_usage(license_id, "installation")
        
        # Save changes
        self._save_deployments()
        
        logger.info(f"Registered deployment {deployment_name} for version {version_id} (Deployment ID: {deployment_id})")
        
        return deployment
    
    def update_deployment_heartbeat(
        self,
        deployment_id: str,
        status: str = "active",
        stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update the heartbeat for a deployment.
        
        Args:
            deployment_id: Deployment ID
            status: Current deployment status
            stats: Updated usage statistics
            
        Returns:
            True if successful, False otherwise
        """
        if deployment_id not in self.deployments:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        deployment = self.deployments[deployment_id]
        
        # Update status and heartbeat
        deployment["status"] = status
        deployment["last_heartbeat"] = datetime.now().isoformat()
        
        # Update statistics if provided
        if stats:
            deployment["usage_stats"].update(stats)
        
        # Save changes
        self._save_deployments()
        
        logger.debug(f"Updated heartbeat for deployment {deployment_id}")
        
        return True
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get the status of a deployment.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment status record
        """
        if deployment_id not in self.deployments:
            logger.error(f"Deployment not found: {deployment_id}")
            return {}
        
        deployment = self.deployments[deployment_id]
        
        # Check if heartbeat is recent (within last 10 minutes)
        last_heartbeat = datetime.fromisoformat(deployment["last_heartbeat"])
        time_since_heartbeat = datetime.now() - last_heartbeat
        
        status = deployment["status"]
        if status == "active" and time_since_heartbeat > timedelta(minutes=10):
            status = "unavailable"
        
        return {
            "id": deployment_id,
            "name": deployment["name"],
            "status": status,
            "version_id": deployment["version_id"],
            "environment": deployment["environment"],
            "endpoint_url": deployment["endpoint_url"],
            "last_heartbeat": deployment["last_heartbeat"],
            "time_since_heartbeat": time_since_heartbeat.total_seconds(),
            "usage_stats": deployment["usage_stats"]
        }
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model record or empty dict if not found
        """
        return self.models.get(model_id, {})
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        Get a model version by ID.
        
        Args:
            version_id: Version ID
            
        Returns:
            ModelVersion or None if not found
        """
        version_data = self._load_version(version_id)
        if not version_data:
            return None
        return ModelVersion.from_dict(version_data)
    
    def get_license(self, license_id: str) -> Dict[str, Any]:
        """
        Get a license by ID.
        
        Args:
            license_id: License ID
            
        Returns:
            License record or empty dict if not found
        """
        return self.licenses.get(license_id, {})
    
    def get_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all models, optionally filtered by type.
        
        Args:
            model_type: Filter by model type
            
        Returns:
            List of model records
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m["type"] == model_type]
        
        return models
    
    def get_versions_for_model(self, model_id: str) -> List[ModelVersion]:
        """
        Get all versions for a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of ModelVersion instances
        """
        if model_id not in self.models:
            logger.error(f"Model not found: {model_id}")
            return []
        
        versions = []
        for version_id in self.models[model_id]["versions"]:
            version_data = self._load_version(version_id)
            if version_data:
                versions.append(ModelVersion.from_dict(version_data))
        
        return versions
    
    def get_deployments(
        self,
        version_id: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get deployments with optional filtering.
        
        Args:
            version_id: Filter by version ID
            environment: Filter by environment
            status: Filter by status
            
        Returns:
            List of deployment records
        """
        deployments = list(self.deployments.values())
        
        if version_id:
            deployments = [d for d in deployments if d["version_id"] == version_id]
        
        if environment:
            deployments = [d for d in deployments if d["environment"] == environment]
        
        if status:
            deployments = [d for d in deployments if d["status"] == status]
        
        return deployments


# Example usage
if __name__ == "__main__":
    # Initialize model distributor
    distributor = ModelDistributor()
    
    # Register a model
    model = distributor.register_model(
        name="MediNex RAG Model",
        description="Medical knowledge retrieval model",
        model_type="rag",
        metadata={"domain": "medical", "languages": ["en"]}
    )
    
    # Create a version
    version = distributor.create_version(
        model_id=model["id"],
        version_number="1.0.0",
        description="Initial release of the medical knowledge retrieval model",
        artifacts_path="./data/sample/rag_model",
        config={"chunk_size": 512, "embedding_model": "text-embedding-3-small"},
        contributors=["contributor-1", "contributor-2"]
    )
    
    # Release the version
    distributor.release_version(version.version_id)
    
    # Package the version
    package_path = distributor.package_version(version.version_id)
    
    # Create a license
    license_record = distributor.create_license(
        version_id=version.version_id,
        user_id="user-1",
        license_type="evaluation",
        expiration_date=(datetime.now() + timedelta(days=30)).isoformat(),
        usage_limits={"max_queries": 1000}
    )
    
    # Register a deployment
    deployment = distributor.register_deployment(
        version_id=version.version_id,
        license_id=license_record["id"],
        deployment_name="Hospital A Deployment",
        environment="production",
        endpoint_url="https://api.hospitala.example/medinex"
    ) 