"""
Integration tests for the Model Distribution module.
"""

import pytest
import os
import json
import tempfile
import shutil
import time
from pathlib import Path

from ai.distribution.model_distribution import (
    ModelVersion, ModelPackage, ModelRegistry, DeploymentManager
)

@pytest.fixture
def test_model_dir():
    """Create a temporary directory with test model files."""
    temp_dir = tempfile.TemporaryDirectory()
    model_dir = Path(temp_dir.name) / "test_model"
    model_dir.mkdir(exist_ok=True)
    
    # Create dummy model files
    main_model = model_dir / "model.bin"
    vocab_file = model_dir / "vocab.txt"
    config_file = model_dir / "config.json"
    
    with open(main_model, "w") as f:
        f.write("dummy model data" * 100)  # Make it a bit larger
    
    with open(vocab_file, "w") as f:
        f.write("word1\nword2\nword3\nmedical\nhealth\ndisease\n")
    
    with open(config_file, "w") as f:
        json.dump({
            "model_type": "medical_llm",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "vocab_size": 30000
        }, f)
    
    yield temp_dir
    
    # Cleanup
    temp_dir.cleanup()

@pytest.fixture
def registry_setup():
    """Set up a model registry for testing."""
    # Create temporary directories
    temp_dir = tempfile.TemporaryDirectory()
    registry_dir = Path(temp_dir.name)
    versions_dir = registry_dir / "versions"
    packages_dir = registry_dir / "packages"
    
    versions_dir.mkdir(exist_ok=True)
    packages_dir.mkdir(exist_ok=True)
    
    # Create registry file
    registry_file = registry_dir / "registry.json"
    with open(registry_file, "w") as f:
        json.dump({"versions": {}, "packages": {}}, f)
    
    # Create registry
    registry = ModelRegistry(
        registry_file=str(registry_file),
        versions_dir=str(versions_dir),
        packages_dir=str(packages_dir)
    )
    
    yield registry, temp_dir
    
    # Cleanup
    temp_dir.cleanup()

@pytest.fixture
def deployment_setup(registry_setup):
    """Set up a deployment manager for testing."""
    registry, temp_dir = registry_setup
    
    # Create deployments directory
    deployments_dir = Path(temp_dir.name) / "deployments"
    deployments_dir.mkdir(exist_ok=True)
    
    # Create deployments file
    deployments_file = Path(temp_dir.name) / "deployments.json"
    with open(deployments_file, "w") as f:
        json.dump({}, f)
    
    # Create deployment manager
    manager = DeploymentManager(
        deployments_file=str(deployments_file),
        deployments_dir=str(deployments_dir),
        model_registry=registry
    )
    
    yield manager, registry, temp_dir
    
    # Cleanup is handled by registry_setup

class TestModelRegistryIntegration:
    """Integration tests for ModelRegistry."""
    
    def test_register_and_retrieve_model_version(self, test_model_dir, registry_setup):
        """Test registering a model version and retrieving it."""
        registry, _ = registry_setup
        model_dir = Path(test_model_dir.name) / "test_model"
        
        # Register model version
        version_id = registry.register_model_version(
            model_type="medical_llm",
            model_files={
                "main": str(model_dir / "model.bin"),
                "vocab": str(model_dir / "vocab.txt"),
                "config": str(model_dir / "config.json")
            },
            metadata={
                "description": "Test medical LLM model",
                "parameters": {
                    "embedding_size": 768,
                    "layers": 12
                }
            }
        )
        
        # Verify version was registered
        assert version_id in registry.versions
        
        # Retrieve the version
        version = registry.get_model_version(version_id)
        
        # Verify version data
        assert version["version_id"] == version_id
        assert version["model_type"] == "medical_llm"
        assert "main" in version["model_files"]
        assert "vocab" in version["model_files"]
        assert "config" in version["model_files"]
        assert "description" in version["metadata"]
        assert "parameters" in version["metadata"]
        
        # Verify files were properly tracked
        assert os.path.exists(version["model_files"]["main"])
        assert os.path.exists(version["model_files"]["vocab"])
        assert os.path.exists(version["model_files"]["config"])
    
    def test_create_and_retrieve_model_package(self, test_model_dir, registry_setup):
        """Test creating a model package and retrieving it."""
        registry, temp_dir = registry_setup
        model_dir = Path(test_model_dir.name) / "test_model"
        
        # Register model version
        version_id = registry.register_model_version(
            model_type="medical_llm",
            model_files={
                "main": str(model_dir / "model.bin"),
                "vocab": str(model_dir / "vocab.txt"),
                "config": str(model_dir / "config.json")
            },
            metadata={
                "description": "Test medical LLM model",
                "parameters": {
                    "embedding_size": 768,
                    "layers": 12
                }
            }
        )
        
        # Create model package
        package_id = registry.create_model_package(
            version_id=version_id,
            package_format="zip",
            license_info={
                "type": "MIT",
                "text": "Permission is hereby granted..."
            }
        )
        
        # Verify package was created
        assert package_id in registry.packages
        
        # Retrieve the package
        package = registry.get_model_package(package_id)
        
        # Verify package data
        assert package["package_id"] == package_id
        assert package["version_id"] == version_id
        assert package["package_format"] == "zip"
        assert "license_info" in package
        assert "package_path" in package
        
        # Verify package file exists
        assert os.path.exists(package["package_path"])
        assert package["package_path"].endswith(".zip")
    
    def test_list_model_versions_by_type(self, test_model_dir, registry_setup):
        """Test listing model versions by type."""
        registry, _ = registry_setup
        model_dir = Path(test_model_dir.name) / "test_model"
        
        # Register LLM model version
        llm_version_id = registry.register_model_version(
            model_type="medical_llm",
            model_files={
                "main": str(model_dir / "model.bin")
            },
            metadata={"description": "LLM model"}
        )
        
        # Register vision model version
        vision_version_id = registry.register_model_version(
            model_type="medical_vision",
            model_files={
                "main": str(model_dir / "model.bin")
            },
            metadata={"description": "Vision model"}
        )
        
        # List all versions
        all_versions = registry.list_model_versions()
        assert len(all_versions) == 2
        assert llm_version_id in all_versions
        assert vision_version_id in all_versions
        
        # List LLM versions
        llm_versions = registry.list_model_versions(model_type="medical_llm")
        assert len(llm_versions) == 1
        assert llm_version_id in llm_versions
        
        # List vision versions
        vision_versions = registry.list_model_versions(model_type="medical_vision")
        assert len(vision_versions) == 1
        assert vision_version_id in vision_versions

class TestEndToEndModelDistribution:
    """End-to-end tests for model distribution workflow."""
    
    def test_register_package_deploy_workflow(self, test_model_dir, deployment_setup):
        """Test the full workflow from registration to deployment."""
        manager, registry, temp_dir = deployment_setup
        model_dir = Path(test_model_dir.name) / "test_model"
        
        # 1. Register model version
        version_id = registry.register_model_version(
            model_type="medical_llm",
            model_files={
                "main": str(model_dir / "model.bin"),
                "vocab": str(model_dir / "vocab.txt"),
                "config": str(model_dir / "config.json")
            },
            metadata={
                "description": "Test medical LLM model",
                "version": "1.0.0",
                "parameters": {
                    "embedding_size": 768,
                    "layers": 12
                }
            }
        )
        
        # 2. Create model package
        package_id = registry.create_model_package(
            version_id=version_id,
            package_format="zip",
            license_info={
                "type": "MIT",
                "text": "Permission is hereby granted..."
            }
        )
        
        # 3. Deploy model
        deployment_id = manager.deploy_model(
            package_id=package_id,
            environment="production",
            metadata={
                "description": "Production deployment of medical LLM",
                "api_endpoint": "https://api.example.com/medinex/llm"
            }
        )
        
        # 4. Verify deployment
        deployment = manager.get_deployment(deployment_id)
        assert deployment["deployment_id"] == deployment_id
        assert deployment["package_id"] == package_id
        assert deployment["environment"] == "production"
        
        # 5. Verify deployment files exist
        deployment_path = deployment["deployment_path"]
        assert os.path.exists(deployment_path)
        assert os.path.exists(os.path.join(deployment_path, "model_files", "main"))
        assert os.path.exists(os.path.join(deployment_path, "model_files", "vocab"))
        assert os.path.exists(os.path.join(deployment_path, "model_files", "config"))
        assert os.path.exists(os.path.join(deployment_path, "metadata.json"))
        
        # 6. Undeploy model
        manager.undeploy_model(deployment_id)
        
        # 7. Verify deployment was removed
        assert deployment_id not in manager.deployments
        assert not os.path.exists(deployment_path)
    
    def test_deploy_multiple_environments(self, test_model_dir, deployment_setup):
        """Test deploying the same model to multiple environments."""
        manager, registry, temp_dir = deployment_setup
        model_dir = Path(test_model_dir.name) / "test_model"
        
        # Register model and create package
        version_id = registry.register_model_version(
            model_type="medical_llm",
            model_files={"main": str(model_dir / "model.bin")},
            metadata={"description": "Test model"}
        )
        
        package_id = registry.create_model_package(
            version_id=version_id,
            package_format="zip",
            license_info={"type": "MIT"}
        )
        
        # Deploy to staging
        staging_id = manager.deploy_model(
            package_id=package_id,
            environment="staging",
            metadata={"description": "Staging deployment"}
        )
        
        # Deploy to production
        production_id = manager.deploy_model(
            package_id=package_id,
            environment="production",
            metadata={"description": "Production deployment"}
        )
        
        # Deploy to test
        test_id = manager.deploy_model(
            package_id=package_id,
            environment="test",
            metadata={"description": "Test deployment"}
        )
        
        # List deployments by environment
        staging_deployments = manager.list_deployments(environment="staging")
        production_deployments = manager.list_deployments(environment="production")
        test_deployments = manager.list_deployments(environment="test")
        
        # Verify correct deployments are listed
        assert len(staging_deployments) == 1
        assert staging_id in staging_deployments
        
        assert len(production_deployments) == 1
        assert production_id in production_deployments
        
        assert len(test_deployments) == 1
        assert test_id in test_deployments
        
        # Verify all deployments exist in the list of all deployments
        all_deployments = manager.list_deployments()
        assert len(all_deployments) == 3
        assert staging_id in all_deployments
        assert production_id in all_deployments
        assert test_id in all_deployments