"""
Unit tests for the Model Distribution module.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
import tempfile
import datetime
import shutil

from ai.distribution.model_distribution import (
    ModelVersion, ModelPackage, ModelRegistry, DeploymentManager
)


class TestModelVersion:
    """Test cases for the ModelVersion class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "test_model")
        os.makedirs(self.model_path, exist_ok=True)
        
        # Create dummy model file
        with open(os.path.join(self.model_path, "model.bin"), "w") as f:
            f.write("dummy model data")
        
        # Create model version
        self.model_version = ModelVersion(
            version_id="v1.0.0",
            model_type="medical_llm",
            model_files={"main": os.path.join(self.model_path, "model.bin")},
            metadata={
                "description": "Test medical LLM model",
                "parameters": {
                    "embedding_size": 768,
                    "layers": 12
                }
            }
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the model version initializes correctly."""
        assert self.model_version.version_id == "v1.0.0"
        assert self.model_version.model_type == "medical_llm"
        assert "main" in self.model_version.model_files
        assert "description" in self.model_version.metadata
        assert "parameters" in self.model_version.metadata
        assert "creation_date" in self.model_version.metadata
        assert isinstance(self.model_version.metadata["creation_date"], str)

    def test_validate_files(self):
        """Test validation of model files."""
        # Valid files should not raise an exception
        self.model_version.validate_files()
        
        # Test with non-existent file
        invalid_model = ModelVersion(
            version_id="v1.0.0",
            model_type="medical_llm",
            model_files={"main": "/nonexistent/path/model.bin"}
        )
        with pytest.raises(FileNotFoundError):
            invalid_model.validate_files()

    def test_get_file_sizes(self):
        """Test getting file sizes."""
        file_sizes = self.model_version.get_file_sizes()
        
        assert "main" in file_sizes
        assert file_sizes["main"] > 0  # Should be the size of "dummy model data"
        assert file_sizes["total"] == file_sizes["main"]

    def test_to_dict(self):
        """Test converting model version to a dictionary."""
        model_dict = self.model_version.to_dict()
        
        assert model_dict["version_id"] == "v1.0.0"
        assert model_dict["model_type"] == "medical_llm"
        assert "model_files" in model_dict
        assert "metadata" in model_dict
        assert "file_sizes" in model_dict


class TestModelPackage:
    """Test cases for the ModelPackage class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.package_dir = os.path.join(self.temp_dir.name, "package")
        os.makedirs(self.package_dir, exist_ok=True)
        
        # Create mock model version
        self.mock_model_version = MagicMock(spec=ModelVersion)
        self.mock_model_version.version_id = "v1.0.0"
        self.mock_model_version.model_type = "medical_llm"
        self.mock_model_version.model_files = {"main": "/path/to/model.bin"}
        self.mock_model_version.metadata = {
            "description": "Test medical LLM model",
            "creation_date": datetime.datetime.now().isoformat(),
            "parameters": {"embedding_size": 768}
        }
        self.mock_model_version.get_file_sizes.return_value = {"main": 1024, "total": 1024}
        self.mock_model_version.to_dict.return_value = {
            "version_id": "v1.0.0",
            "model_type": "medical_llm",
            "model_files": {"main": "/path/to/model.bin"},
            "metadata": self.mock_model_version.metadata,
            "file_sizes": {"main": 1024, "total": 1024}
        }
        
        # Create model package
        self.model_package = ModelPackage(
            package_id="pkg-1",
            model_version=self.mock_model_version,
            output_dir=self.package_dir,
            package_format="zip",
            license_info={
                "type": "MIT",
                "text": "Permission is hereby granted..."
            }
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the model package initializes correctly."""
        assert self.model_package.package_id == "pkg-1"
        assert self.model_package.model_version == self.mock_model_version
        assert self.model_package.output_dir == self.package_dir
        assert self.model_package.package_format == "zip"
        assert "type" in self.model_package.license_info
        assert "text" in self.model_package.license_info
        assert "creation_date" in self.model_package.metadata
        assert isinstance(self.model_package.metadata["creation_date"], str)

    @patch("shutil.copy2")
    def test_prepare_package_files(self, mock_copy):
        """Test preparing package files."""
        # Create temporary package directory
        package_path = self.model_package._prepare_package_files()
        
        # Verify package directory was created
        assert os.path.exists(package_path)
        assert os.path.isdir(package_path)
        
        # Verify files were copied
        mock_copy.assert_called_once_with(
            self.mock_model_version.model_files["main"],
            os.path.join(package_path, "model_files", "main")
        )
        
        # Verify metadata file was created
        assert os.path.exists(os.path.join(package_path, "metadata.json"))

    @patch("zipfile.ZipFile")
    @patch.object(ModelPackage, "_prepare_package_files")
    def test_create_package_zip(self, mock_prepare, mock_zipfile):
        """Test creating a ZIP package."""
        # Configure mock to return a temp directory
        temp_package_dir = os.path.join(self.temp_dir.name, "temp_package")
        os.makedirs(temp_package_dir, exist_ok=True)
        mock_prepare.return_value = temp_package_dir
        
        # Mock ZipFile context manager
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Call create_package
        package_path = self.model_package.create_package()
        
        # Verify zip file was created
        mock_zipfile.assert_called_once()
        assert package_path.endswith(".zip")

    @patch("tarfile.open")
    @patch.object(ModelPackage, "_prepare_package_files")
    def test_create_package_tar(self, mock_prepare, mock_tarfile):
        """Test creating a TAR package."""
        # Configure mock to return a temp directory
        temp_package_dir = os.path.join(self.temp_dir.name, "temp_package")
        os.makedirs(temp_package_dir, exist_ok=True)
        mock_prepare.return_value = temp_package_dir
        
        # Set package format to tar
        self.model_package.package_format = "tar"
        
        # Mock tarfile context manager
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar
        
        # Call create_package
        package_path = self.model_package.create_package()
        
        # Verify tar file was created
        mock_tarfile.assert_called_once()
        assert package_path.endswith(".tar.gz")

    def test_generate_manifest(self):
        """Test generating a package manifest."""
        manifest = self.model_package.generate_manifest()
        
        assert manifest["package_id"] == "pkg-1"
        assert manifest["model_version"] == "v1.0.0"
        assert manifest["model_type"] == "medical_llm"
        assert "license_info" in manifest
        assert "metadata" in manifest
        assert "creation_date" in manifest["metadata"]


class TestModelRegistry:
    """Test cases for the ModelRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temp file for registry data
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        
        # Create temp directories for model versions and packages
        self.temp_dir = tempfile.TemporaryDirectory()
        self.versions_dir = os.path.join(self.temp_dir.name, "versions")
        self.packages_dir = os.path.join(self.temp_dir.name, "packages")
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(self.packages_dir, exist_ok=True)
        
        # Create registry
        self.registry = ModelRegistry(
            registry_file=self.temp_file.name,
            versions_dir=self.versions_dir,
            packages_dir=self.packages_dir
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the registry initializes correctly."""
        assert self.registry.registry_file == self.temp_file.name
        assert self.registry.versions_dir == self.versions_dir
        assert self.registry.packages_dir == self.packages_dir
        assert isinstance(self.registry.versions, dict)
        assert isinstance(self.registry.packages, dict)

    @patch.object(ModelVersion, "validate_files")
    def test_register_model_version(self, mock_validate):
        """Test registering a model version."""
        # Create a model version
        version_id = self.registry.register_model_version(
            model_type="medical_llm",
            model_files={"main": "/path/to/model.bin"},
            metadata={"description": "Test model"}
        )
        
        # Verify version was registered
        assert version_id in self.registry.versions
        assert self.registry.versions[version_id]["model_type"] == "medical_llm"
        assert "model_files" in self.registry.versions[version_id]
        assert "metadata" in self.registry.versions[version_id]
        
        # Verify registry file was updated
        with open(self.temp_file.name, 'r') as f:
            registry_data = json.load(f)
            assert version_id in registry_data["versions"]

    @patch.object(ModelPackage, "create_package")
    def test_create_model_package(self, mock_create_package):
        """Test creating a model package."""
        # Set up mock
        mock_create_package.return_value = os.path.join(self.packages_dir, "pkg-1.zip")
        
        # Register a version first
        version_id = self.registry.register_model_version(
            model_type="medical_llm",
            model_files={"main": "/path/to/model.bin"},
            metadata={"description": "Test model"}
        )
        
        # Create a package
        package_id = self.registry.create_model_package(
            version_id=version_id,
            package_format="zip",
            license_info={"type": "MIT"}
        )
        
        # Verify package was created
        assert package_id in self.registry.packages
        assert self.registry.packages[package_id]["version_id"] == version_id
        assert self.registry.packages[package_id]["package_format"] == "zip"
        assert "license_info" in self.registry.packages[package_id]
        
        # Verify registry file was updated
        with open(self.temp_file.name, 'r') as f:
            registry_data = json.load(f)
            assert package_id in registry_data["packages"]

    def test_get_model_version(self):
        """Test getting a model version."""
        # Register a version first
        version_id = self.registry.register_model_version(
            model_type="medical_llm",
            model_files={"main": "/path/to/model.bin"},
            metadata={"description": "Test model"}
        )
        
        # Get the version
        version = self.registry.get_model_version(version_id)
        
        # Verify version data
        assert version["version_id"] == version_id
        assert version["model_type"] == "medical_llm"
        assert "model_files" in version
        assert "metadata" in version
        
        # Test getting non-existent version
        with pytest.raises(ValueError):
            self.registry.get_model_version("non_existent")

    def test_get_model_package(self):
        """Test getting a model package."""
        # Register a version and create a package first
        version_id = self.registry.register_model_version(
            model_type="medical_llm",
            model_files={"main": "/path/to/model.bin"},
            metadata={"description": "Test model"}
        )
        
        # Mock package creation
        with patch.object(ModelPackage, "create_package") as mock_create:
            mock_create.return_value = os.path.join(self.packages_dir, "pkg-1.zip")
            package_id = self.registry.create_model_package(
                version_id=version_id,
                package_format="zip",
                license_info={"type": "MIT"}
            )
        
        # Get the package
        package = self.registry.get_model_package(package_id)
        
        # Verify package data
        assert package["package_id"] == package_id
        assert package["version_id"] == version_id
        assert package["package_format"] == "zip"
        assert "license_info" in package
        
        # Test getting non-existent package
        with pytest.raises(ValueError):
            self.registry.get_model_package("non_existent")

    def test_list_model_versions(self):
        """Test listing model versions."""
        # Register a couple of versions
        version1 = self.registry.register_model_version(
            model_type="medical_llm",
            model_files={"main": "/path/to/model1.bin"},
            metadata={"description": "Test model 1"}
        )
        version2 = self.registry.register_model_version(
            model_type="medical_vision",
            model_files={"main": "/path/to/model2.bin"},
            metadata={"description": "Test model 2"}
        )
        
        # List all versions
        versions = self.registry.list_model_versions()
        assert len(versions) == 2
        assert version1 in versions
        assert version2 in versions
        
        # List versions by type
        llm_versions = self.registry.list_model_versions(model_type="medical_llm")
        assert len(llm_versions) == 1
        assert version1 in llm_versions
        
        vision_versions = self.registry.list_model_versions(model_type="medical_vision")
        assert len(vision_versions) == 1
        assert version2 in vision_versions


class TestDeploymentManager:
    """Test cases for the DeploymentManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temp file for deployment data
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        
        # Create temp directories for deployments
        self.temp_dir = tempfile.TemporaryDirectory()
        self.deployments_dir = os.path.join(self.temp_dir.name, "deployments")
        os.makedirs(self.deployments_dir, exist_ok=True)
        
        # Create mock registry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_registry.get_model_package.return_value = {
            "package_id": "pkg-1",
            "version_id": "v1.0.0",
            "package_format": "zip",
            "package_path": "/path/to/package.zip",
            "license_info": {"type": "MIT"}
        }
        
        # Create deployment manager
        self.manager = DeploymentManager(
            deployments_file=self.temp_file.name,
            deployments_dir=self.deployments_dir,
            model_registry=self.mock_registry
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the deployment manager initializes correctly."""
        assert self.manager.deployments_file == self.temp_file.name
        assert self.manager.deployments_dir == self.deployments_dir
        assert self.manager.model_registry == self.mock_registry
        assert isinstance(self.manager.deployments, dict)

    @patch("zipfile.ZipFile")
    @patch("os.makedirs")
    def test_deploy_model(self, mock_makedirs, mock_zipfile):
        """Test deploying a model."""
        # Mock extract
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Deploy a model
        deployment_id = self.manager.deploy_model(
            package_id="pkg-1",
            environment="production",
            metadata={
                "description": "Production deployment of medical LLM",
                "api_endpoint": "https://api.example.com/medinex/llm"
            }
        )
        
        # Verify deployment was created
        assert deployment_id in self.manager.deployments
        assert self.manager.deployments[deployment_id]["package_id"] == "pkg-1"
        assert self.manager.deployments[deployment_id]["environment"] == "production"
        assert "description" in self.manager.deployments[deployment_id]["metadata"]
        assert "api_endpoint" in self.manager.deployments[deployment_id]["metadata"]
        assert "deployment_path" in self.manager.deployments[deployment_id]
        assert "deployment_date" in self.manager.deployments[deployment_id]
        
        # Verify package was extracted
        mock_zipfile.assert_called_once()
        mock_zip.extractall.assert_called_once()
        
        # Verify deployments file was updated
        with open(self.temp_file.name, 'r') as f:
            deployment_data = json.load(f)
            assert deployment_id in deployment_data

    def test_get_deployment(self):
        """Test getting a deployment."""
        # Create a test deployment in the manager's data
        deployment_id = "dep-1"
        self.manager.deployments[deployment_id] = {
            "deployment_id": deployment_id,
            "package_id": "pkg-1",
            "environment": "production",
            "deployment_path": os.path.join(self.deployments_dir, deployment_id),
            "deployment_date": datetime.datetime.now().isoformat(),
            "metadata": {
                "description": "Test deployment",
                "api_endpoint": "https://api.example.com/test"
            }
        }
        self.manager._save_deployments()
        
        # Get the deployment
        deployment = self.manager.get_deployment(deployment_id)
        
        # Verify deployment data
        assert deployment["deployment_id"] == deployment_id
        assert deployment["package_id"] == "pkg-1"
        assert deployment["environment"] == "production"
        assert "deployment_path" in deployment
        assert "deployment_date" in deployment
        assert "metadata" in deployment
        
        # Test getting non-existent deployment
        with pytest.raises(ValueError):
            self.manager.get_deployment("non_existent")

    def test_list_deployments(self):
        """Test listing deployments."""
        # Create a couple of test deployments
        self.manager.deployments["dep-1"] = {
            "deployment_id": "dep-1",
            "package_id": "pkg-1",
            "environment": "production",
            "deployment_path": "/path/to/deployment1",
            "deployment_date": datetime.datetime.now().isoformat(),
            "metadata": {"description": "Production deployment"}
        }
        self.manager.deployments["dep-2"] = {
            "deployment_id": "dep-2",
            "package_id": "pkg-2",
            "environment": "staging",
            "deployment_path": "/path/to/deployment2",
            "deployment_date": datetime.datetime.now().isoformat(),
            "metadata": {"description": "Staging deployment"}
        }
        
        # List all deployments
        deployments = self.manager.list_deployments()
        assert len(deployments) == 2
        assert "dep-1" in deployments
        assert "dep-2" in deployments
        
        # List deployments by environment
        prod_deployments = self.manager.list_deployments(environment="production")
        assert len(prod_deployments) == 1
        assert "dep-1" in prod_deployments
        
        staging_deployments = self.manager.list_deployments(environment="staging")
        assert len(staging_deployments) == 1
        assert "dep-2" in staging_deployments

    @patch("shutil.rmtree")
    def test_undeploy_model(self, mock_rmtree):
        """Test undeploying a model."""
        # Create a test deployment
        deployment_id = "dep-1"
        deployment_path = os.path.join(self.deployments_dir, deployment_id)
        self.manager.deployments[deployment_id] = {
            "deployment_id": deployment_id,
            "package_id": "pkg-1",
            "environment": "production",
            "deployment_path": deployment_path,
            "deployment_date": datetime.datetime.now().isoformat(),
            "metadata": {"description": "Test deployment"}
        }
        self.manager._save_deployments()
        
        # Undeploy the model
        self.manager.undeploy_model(deployment_id)
        
        # Verify deployment was removed
        assert deployment_id not in self.manager.deployments
        
        # Verify deployment directory was removed
        mock_rmtree.assert_called_once_with(deployment_path)
        
        # Verify deployments file was updated
        with open(self.temp_file.name, 'r') as f:
            deployment_data = json.load(f)
            assert deployment_id not in deployment_data 