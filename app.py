#!/usr/bin/env python3
"""
MediNex AI - Medical Knowledge Assistant

This is the main entry point for the MediNex AI application, providing a command-line
interface to interact with the system's various capabilities, including the knowledge base,
LLM querying, medical image analysis, and API server management.
"""

import os
import json
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("medinex")

# Import MediNex AI components
try:
    from ai.knowledge.medical_rag import MedicalKnowledgeBase, MedicalRAG
    from ai.knowledge.data_importer import MedicalDataImporter
    from ai.llm.model_connector import MedicalLLMConnector
    from ai.integrations.imaging_llm_pipeline import MedicalImagingPipeline
    from ai.api.core import create_api
    from ai.contributors.contributor_manager import ContributorManager
    from ai.contributors.revenue_sharing import RevenueShareSystem
    from ai.distribution.model_distribution import ModelDistributor
except ImportError as e:
    logger.error(f"Failed to import MediNex AI components: {str(e)}")
    logger.error("Make sure you have installed all required dependencies and are running from the project root.")
    sys.exit(1)

# Default configuration file path
DEFAULT_CONFIG_PATH = "config.json"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load the application configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.info("Using default configuration")
        return {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7
            },
            "knowledge_base": {
                "storage_path": "./data/knowledge",
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }


def create_directories() -> None:
    """Create necessary directories for the application."""
    directories = [
        "data",
        "data/knowledge",
        "data/sample",
        "models",
        "models/imaging",
        "cache",
        "cache/imaging",
        "logs",
        "data/contributors",
        "data/revenue",
        "data/models",
        "data/models/versions",
        "data/models/packages",
        "data/models/deployments",
        "data/tmp/model_artifacts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    logger.info("Created necessary directories")


def initialize_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize the MediNex AI system components.
    
    Args:
        config: Application configuration
        
    Returns:
        Dictionary containing system components
    """
    # Initialize knowledge base
    kb = MedicalKnowledgeBase(
        storage_path=config["knowledge_base"]["storage_path"],
        chunk_size=config["knowledge_base"]["chunk_size"],
        chunk_overlap=config["knowledge_base"]["chunk_overlap"]
    )
    
    # Initialize RAG
    rag = MedicalRAG(knowledge_base=kb)
    
    # Initialize data importer
    importer = MedicalDataImporter(kb)
    
    # Initialize LLM connector
    llm = MedicalLLMConnector(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"]
    )
    
    # Initialize imaging pipeline
    imaging = MedicalImagingPipeline(
        llm_connector=llm,
        models_dir=config["imaging"]["models_dir"],
        cache_dir=config["imaging"]["cache_dir"]
    )
    
    # Initialize contributor management system
    contributor_manager = ContributorManager(
        storage_path=config.get("contributors", {}).get("storage_path", "./data/contributors")
    )
    
    # Initialize revenue sharing system
    revenue_system = RevenueShareSystem(
        storage_path=config.get("revenue", {}).get("storage_path", "./data/revenue"),
        contributor_manager=contributor_manager
    )
    
    # Initialize model distribution system
    model_distributor = ModelDistributor(
        storage_path=config.get("distribution", {}).get("storage_path", "./data/models")
    )
    
    logger.info("System components initialized")
    
    return {
        "knowledge_base": kb,
        "rag": rag,
        "importer": importer,
        "llm": llm,
        "imaging": imaging,
        "contributor_manager": contributor_manager,
        "revenue_system": revenue_system,
        "model_distributor": model_distributor
    }


def cmd_init(args, config: Dict[str, Any]) -> None:
    """Initialize the MediNex AI system."""
    logger.info("Initializing MediNex AI system...")
    
    # Create necessary directories
    create_directories()
    
    # Initialize components
    components = initialize_system(config)
    
    logger.info("MediNex AI system initialized successfully")
    
    # Display environment status
    display_env_status()


def cmd_serve(args, config: Dict[str, Any]) -> None:
    """Start the API server."""
    logger.info(f"Starting MediNex AI API server on {args.host}:{args.port}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Create FastAPI application
    api = create_api({
        "rag": components["rag"],
        "knowledge_base": components["knowledge_base"],
        "llm_connector": components["llm"],
        "imaging_pipeline": components["imaging"],
        "contributor_manager": components["contributor_manager"],
        "revenue_system": components["revenue_system"],
        "model_distributor": components["model_distributor"]
    })
    
    # Run the server
    uvicorn.run(
        api,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )


def cmd_import(args, config: Dict[str, Any]) -> None:
    """Import data into the knowledge base."""
    logger.info(f"Importing data from {args.directory}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Import data
    start_time = time.time()
    result = components["importer"].import_directory(
        directory_path=args.directory,
        recursive=not args.no_recursive
    )
    
    # Display results
    logger.info(f"Import completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Successfully imported {result['successful_imports']} documents")
    logger.info(f"Failed to import {result['failed_imports']} documents")
    
    # Display statistics by file type
    if "by_type" in result:
        logger.info("Import statistics by file type:")
        for file_type, count in result["by_type"].items():
            if count > 0:
                logger.info(f"  {file_type}: {count}")


def cmd_query(args, config: Dict[str, Any]) -> None:
    """Query the MediNex AI system."""
    logger.info("Processing query...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Process query
    query = args.query
    use_kb = not args.no_kb
    
    start_time = time.time()
    
    if use_kb:
        # Use RAG for enhanced response
        response = components["rag"].query(
            query=query,
            max_context_docs=args.max_docs
        )
    else:
        # Use LLM directly
        response = components["llm"].generate_medical_response(
            query=query,
            context=None
        )
    
    # Display results
    logger.info(f"Query processed in {time.time() - start_time:.2f} seconds")
    print("\nQuery: " + query)
    print("\nResponse:")
    print(response)


def cmd_analyze_image(args, config: Dict[str, Any]) -> None:
    """Analyze a medical image."""
    logger.info(f"Analyzing image {args.image_path}...")
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Image not found: {args.image_path}")
        return
    
    # Initialize components
    components = initialize_system(config)
    
    # Analyze image
    start_time = time.time()
    
    with open(args.image_path, "rb") as f:
        image_data = f.read()
    
    result = components["imaging"].analyze_medical_image(
        image_data=image_data,
        prompt=args.prompt,
        analysis_type=args.type
    )
    
    # Display results
    logger.info(f"Image analysis completed in {time.time() - start_time:.2f} seconds")
    print("\nImage Analysis Results:")
    print(f"Findings: {result['findings']}")
    print(f"\nInterpretation: {result['interpretation']}")
    print(f"\nImpression: {result['impression']}")


def cmd_list_documents(args, config: Dict[str, Any]) -> None:
    """List documents in the knowledge base."""
    logger.info("Listing knowledge base documents...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Get documents
    docs = components["knowledge_base"].list_documents(
        limit=args.limit,
        offset=args.offset
    )
    
    # Display results
    print(f"\nFound {len(docs)} documents:")
    for doc in docs:
        print(f"ID: {doc['id']}")
        print(f"Title: {doc.get('title', 'No title')}")
        print(f"Source: {doc.get('source', 'Unknown')}")
        print(f"Created: {doc.get('created_at', 'Unknown')}")
        print("-" * 50)


def cmd_delete_document(args, config: Dict[str, Any]) -> None:
    """Delete a document from the knowledge base."""
    logger.info(f"Deleting document {args.doc_id}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Delete document
    result = components["knowledge_base"].delete_document(args.doc_id)
    
    # Display results
    if result:
        logger.info(f"Document {args.doc_id} deleted successfully")
    else:
        logger.error(f"Failed to delete document {args.doc_id}")


def cmd_register_contributor(args, config: Dict[str, Any]) -> None:
    """Register a new contributor."""
    logger.info(f"Registering contributor {args.name}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Register contributor
    contributor = components["contributor_manager"].register_contributor(
        name=args.name,
        email=args.email,
        institution=args.institution,
        specialization=args.specialization,
        metadata=json.loads(args.metadata) if args.metadata else {}
    )
    
    # Display results
    print(f"\nContributor registered successfully:")
    print(f"ID: {contributor['contributor_id']}")
    print(f"Name: {contributor['name']}")
    print(f"Email: {contributor['email']}")
    if contributor.get('institution'):
        print(f"Institution: {contributor['institution']}")
    if contributor.get('specialization'):
        print(f"Specialization: {contributor['specialization']}")
    print(f"Join Date: {contributor['join_date']}")


def cmd_list_contributors(args, config: Dict[str, Any]) -> None:
    """List all contributors."""
    logger.info("Listing contributors...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Get contributors
    contributors = components["contributor_manager"].get_contributors(
        active_only=args.active_only
    )
    
    # Display results
    print(f"\nFound {len(contributors)} contributors:")
    for contributor in contributors:
        print(f"ID: {contributor['contributor_id']}")
        print(f"Name: {contributor['name']}")
        print(f"Email: {contributor['email']}")
        if contributor.get('institution'):
            print(f"Institution: {contributor['institution']}")
        if contributor.get('specialization'):
            print(f"Specialization: {contributor['specialization']}")
        print(f"Active: {'Yes' if contributor['active'] else 'No'}")
        print(f"Join Date: {contributor['join_date']}")
        print(f"Contributions: {len(contributor['contributions'])}")
        print("-" * 50)


def cmd_record_contribution(args, config: Dict[str, Any]) -> None:
    """Record a contribution from a contributor."""
    logger.info(f"Recording contribution from contributor {args.contributor_id}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Record contribution
    success = components["contributor_manager"].record_contribution(
        contributor_id=args.contributor_id,
        contribution_type=args.type,
        description=args.description,
        value=args.value,
        metadata=json.loads(args.metadata) if args.metadata else {}
    )
    
    # Display results
    if success:
        logger.info(f"Contribution recorded successfully")
    else:
        logger.error(f"Failed to record contribution")


def cmd_create_revenue_period(args, config: Dict[str, Any]) -> None:
    """Create a new revenue period."""
    logger.info(f"Creating revenue period {args.name}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Create revenue period
    period = components["revenue_system"].create_revenue_period(
        name=args.name,
        start_date=args.start_date,
        end_date=args.end_date,
        total_revenue=args.total_revenue,
        currency=args.currency,
        metadata=json.loads(args.metadata) if args.metadata else {}
    )
    
    # Display results
    print(f"\nRevenue period created successfully:")
    print(f"ID: {period['period_id']}")
    print(f"Name: {period['name']}")
    print(f"Start Date: {period['start_date']}")
    print(f"End Date: {period['end_date']}")
    print(f"Total Revenue: {period['total_revenue']} {period['currency']}")
    print(f"Status: {period['status']}")


def cmd_calculate_shares(args, config: Dict[str, Any]) -> None:
    """Calculate revenue shares for a period."""
    logger.info(f"Calculating revenue shares for period {args.period_id}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Calculate shares
    shares = components["revenue_system"].calculate_shares(
        period_id=args.period_id,
        detailed=args.detailed
    )
    
    # Display results
    print(f"\nRevenue shares calculated for period {args.period_id}:")
    for share in shares:
        print(f"Contributor: {share['contributor_name']} ({share['contributor_id']})")
        print(f"Share Percentage: {share['percentage']:.2f}%")
        print(f"Amount: {share['amount']} {share['currency']}")
        if args.detailed and 'breakdown' in share:
            print("Breakdown:")
            for item in share['breakdown']:
                print(f"  {item['contribution_type']}: {item['percentage']:.2f}% ({item['amount']} {item['currency']})")
        print("-" * 50)


def cmd_register_model(args, config: Dict[str, Any]) -> None:
    """Register a new model in the distribution system."""
    logger.info(f"Registering model {args.name}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Register model
    model = components["model_distributor"].register_model(
        name=args.name,
        description=args.description,
        model_type=args.type,
        metadata=json.loads(args.metadata) if args.metadata else {}
    )
    
    # Display results
    print(f"\nModel registered successfully:")
    print(f"ID: {model['id']}")
    print(f"Name: {model['name']}")
    print(f"Type: {model['type']}")
    print(f"Created At: {model['created_at']}")


def cmd_create_model_version(args, config: Dict[str, Any]) -> None:
    """Create a new version of a model."""
    logger.info(f"Creating version {args.version} for model {args.model_id}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Create artifacts path
    artifacts_path = f"./data/tmp/model_artifacts/{args.model_id}/{args.version}"
    os.makedirs(artifacts_path, exist_ok=True)
    
    # If artifacts directory provided, copy contents
    if args.artifacts_path and os.path.exists(args.artifacts_path):
        import shutil
        if os.path.isdir(args.artifacts_path):
            for item in os.listdir(args.artifacts_path):
                source = os.path.join(args.artifacts_path, item)
                dest = os.path.join(artifacts_path, item)
                if os.path.isdir(source):
                    shutil.copytree(source, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, dest)
        else:
            shutil.copy2(args.artifacts_path, os.path.join(artifacts_path, os.path.basename(args.artifacts_path)))
    
    # Parse contributors
    contributors = args.contributors.split(',') if args.contributors else []
    
    # Create config
    config_data = {}
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
    
    # Create version
    version = components["model_distributor"].create_version(
        model_id=args.model_id,
        version_number=args.version,
        description=args.description,
        artifacts_path=artifacts_path,
        config=config_data,
        contributors=contributors
    )
    
    # Display results
    if version:
        print(f"\nModel version created successfully:")
        print(f"ID: {version.version_id}")
        print(f"Version: {version.version_number}")
        print(f"Status: {version.status}")
        print(f"Contributors: {', '.join(version.contributors) if version.contributors else 'None'}")
        print(f"Created At: {version.created_at}")
    else:
        logger.error("Failed to create model version")


def cmd_release_model_version(args, config: Dict[str, Any]) -> None:
    """Release a model version."""
    logger.info(f"Releasing model version {args.version_id}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Release version
    version = components["model_distributor"].release_version(args.version_id)
    
    # Display results
    if version:
        print(f"\nModel version released successfully:")
        print(f"ID: {version.version_id}")
        print(f"Version: {version.version_number}")
        print(f"Status: {version.status}")
    else:
        logger.error("Failed to release model version")


def cmd_create_license(args, config: Dict[str, Any]) -> None:
    """Create a license for a model version."""
    logger.info(f"Creating license for version {args.version_id}...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Parse usage limits
    usage_limits = {}
    if args.usage_limits:
        usage_limits = json.loads(args.usage_limits)
    
    # Create license
    license_record = components["model_distributor"].create_license(
        version_id=args.version_id,
        user_id=args.user_id,
        license_type=args.type,
        expiration_date=args.expiration_date,
        usage_limits=usage_limits,
        custom_terms=args.custom_terms
    )
    
    # Display results
    if license_record:
        print(f"\nLicense created successfully:")
        print(f"ID: {license_record['id']}")
        print(f"License Key: {license_record['license_key']}")
        print(f"Type: {license_record['type']}")
        print(f"User: {license_record['user_id']}")
        if license_record.get('expiration_date'):
            print(f"Expires: {license_record['expiration_date']}")
        print(f"Status: {license_record['status']}")
        print(f"Created At: {license_record['created_at']}")
    else:
        logger.error("Failed to create license")


def cmd_list_models(args, config: Dict[str, Any]) -> None:
    """List models in the distribution system."""
    logger.info("Listing models...")
    
    # Initialize components
    components = initialize_system(config)
    
    # Get models
    models = components["model_distributor"].get_models(
        model_type=args.type
    )
    
    # Display results
    print(f"\nFound {len(models)} models:")
    for model in models:
        print(f"ID: {model['id']}")
        print(f"Name: {model['name']}")
        print(f"Type: {model['type']}")
        print(f"Description: {model['description']}")
        print(f"Versions: {len(model['versions'])}")
        print(f"Created At: {model['created_at']}")
        print(f"Updated At: {model['updated_at']}")
        print("-" * 50)


def display_env_status() -> None:
    """Display information about the execution environment."""
    try:
        import platform
        import sys
        import importlib.metadata as metadata
        
        print("\nEnvironment Information:")
        print(f"Python version: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        
        # List key package versions
        key_packages = [
            "openai", "anthropic", "fastapi", "uvicorn", "numpy", 
            "pandas", "scipy", "torch", "transformers", "pillow", "opencv-python"
        ]
        
        print("\nInstalled Packages:")
        for package in key_packages:
            try:
                version = metadata.version(package)
                print(f"  {package}: {version}")
            except metadata.PackageNotFoundError:
                print(f"  {package}: Not installed")
        
        print("\nAPI Keys:")
        for key_name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if os.environ.get(key_name):
                print(f"  {key_name}: Set")
            else:
                print(f"  {key_name}: Not set")
                
    except Exception as e:
        logger.error(f"Error displaying environment status: {str(e)}")


def main():
    """Main entry point for the MediNex AI application."""
    parser = argparse.ArgumentParser(description="MediNex AI - Medical Knowledge Assistant")
    
    # Global options
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to the configuration file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize the MediNex AI system")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    serve_parser.add_argument("--log-level", default="INFO", choices=["debug", "info", "warning", "error"], help="Server log level")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import data into the knowledge base")
    import_parser.add_argument("--directory", required=True, help="Directory containing data to import")
    import_parser.add_argument("--no-recursive", action="store_true", help="Do not recursively import from subdirectories")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the MediNex AI system")
    query_parser.add_argument("query", help="Query to process")
    query_parser.add_argument("--no-kb", action="store_true", help="Do not use the knowledge base")
    query_parser.add_argument("--max-docs", type=int, default=5, help="Maximum number of documents to use for context")
    
    # Analyze image command
    analyze_image_parser = subparsers.add_parser("analyze-image", help="Analyze a medical image")
    analyze_image_parser.add_argument("image_path", help="Path to the image file")
    analyze_image_parser.add_argument("--prompt", help="Additional prompt for the analysis")
    analyze_image_parser.add_argument("--type", default="general", help="Type of analysis to perform")
    
    # List documents command
    list_docs_parser = subparsers.add_parser("list-documents", help="List documents in the knowledge base")
    list_docs_parser.add_argument("--limit", type=int, default=10, help="Maximum number of documents to list")
    list_docs_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    
    # Delete document command
    delete_doc_parser = subparsers.add_parser("delete-document", help="Delete a document from the knowledge base")
    delete_doc_parser.add_argument("doc_id", help="ID of the document to delete")
    
    # =============== Contributor Management Commands ===============
    
    # Register contributor command
    register_contributor_parser = subparsers.add_parser("register-contributor", help="Register a new contributor")
    register_contributor_parser.add_argument("--name", required=True, help="Contributor's name")
    register_contributor_parser.add_argument("--email", required=True, help="Contributor's email")
    register_contributor_parser.add_argument("--institution", help="Contributor's institution")
    register_contributor_parser.add_argument("--specialization", help="Contributor's specialization")
    register_contributor_parser.add_argument("--metadata", help="Additional metadata in JSON format")
    
    # List contributors command
    list_contributors_parser = subparsers.add_parser("list-contributors", help="List all contributors")
    list_contributors_parser.add_argument("--active-only", action="store_true", help="List only active contributors")
    
    # Record contribution command
    record_contribution_parser = subparsers.add_parser("record-contribution", help="Record a contribution from a contributor")
    record_contribution_parser.add_argument("--contributor-id", required=True, help="Contributor ID")
    record_contribution_parser.add_argument("--type", required=True, help="Contribution type")
    record_contribution_parser.add_argument("--description", required=True, help="Contribution description")
    record_contribution_parser.add_argument("--value", type=float, required=True, help="Contribution value")
    record_contribution_parser.add_argument("--metadata", help="Additional metadata in JSON format")
    
    # =============== Revenue Sharing Commands ===============
    
    # Create revenue period command
    create_revenue_period_parser = subparsers.add_parser("create-revenue-period", help="Create a new revenue period")
    create_revenue_period_parser.add_argument("--name", required=True, help="Period name")
    create_revenue_period_parser.add_argument("--start-date", required=True, help="Start date (ISO format)")
    create_revenue_period_parser.add_argument("--end-date", required=True, help="End date (ISO format)")
    create_revenue_period_parser.add_argument("--total-revenue", type=float, required=True, help="Total revenue amount")
    create_revenue_period_parser.add_argument("--currency", default="USD", help="Currency")
    create_revenue_period_parser.add_argument("--metadata", help="Additional metadata in JSON format")
    
    # Calculate shares command
    calculate_shares_parser = subparsers.add_parser("calculate-shares", help="Calculate revenue shares for a period")
    calculate_shares_parser.add_argument("--period-id", required=True, help="Revenue period ID")
    calculate_shares_parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown")
    
    # =============== Model Distribution Commands ===============
    
    # Register model command
    register_model_parser = subparsers.add_parser("register-model", help="Register a new model")
    register_model_parser.add_argument("--name", required=True, help="Model name")
    register_model_parser.add_argument("--description", required=True, help="Model description")
    register_model_parser.add_argument("--type", required=True, help="Model type (e.g., 'rag', 'llm', 'imaging')")
    register_model_parser.add_argument("--metadata", help="Additional metadata in JSON format")
    
    # Create model version command
    create_model_version_parser = subparsers.add_parser("create-model-version", help="Create a new version of a model")
    create_model_version_parser.add_argument("--model-id", required=True, help="Model ID")
    create_model_version_parser.add_argument("--version", required=True, help="Version number (e.g., '1.0.0')")
    create_model_version_parser.add_argument("--description", required=True, help="Version description")
    create_model_version_parser.add_argument("--artifacts-path", help="Path to model artifacts")
    create_model_version_parser.add_argument("--config", help="Path to model configuration file")
    create_model_version_parser.add_argument("--contributors", help="Comma-separated list of contributor IDs")
    
    # Release model version command
    release_model_version_parser = subparsers.add_parser("release-model-version", help="Release a model version")
    release_model_version_parser.add_argument("--version-id", required=True, help="Version ID")
    
    # Create license command
    create_license_parser = subparsers.add_parser("create-license", help="Create a license for a model version")
    create_license_parser.add_argument("--version-id", required=True, help="Version ID")
    create_license_parser.add_argument("--user-id", required=True, help="User or organization ID")
    create_license_parser.add_argument("--type", required=True, help="License type ('standard', 'academic', 'commercial', 'evaluation')")
    create_license_parser.add_argument("--expiration-date", help="Expiration date (ISO format)")
    create_license_parser.add_argument("--usage-limits", help="Usage limitations in JSON format")
    create_license_parser.add_argument("--custom-terms", help="Custom license terms")
    
    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List models in the distribution system")
    list_models_parser.add_argument("--type", help="Filter by model type")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
    if args.command == "init":
        cmd_init(args, config)
    elif args.command == "serve":
        cmd_serve(args, config)
    elif args.command == "import":
        cmd_import(args, config)
    elif args.command == "query":
        cmd_query(args, config)
    elif args.command == "analyze-image":
        cmd_analyze_image(args, config)
    elif args.command == "list-documents":
        cmd_list_documents(args, config)
    elif args.command == "delete-document":
        cmd_delete_document(args, config)
    elif args.command == "register-contributor":
        cmd_register_contributor(args, config)
    elif args.command == "list-contributors":
        cmd_list_contributors(args, config)
    elif args.command == "record-contribution":
        cmd_record_contribution(args, config)
    elif args.command == "create-revenue-period":
        cmd_create_revenue_period(args, config)
    elif args.command == "calculate-shares":
        cmd_calculate_shares(args, config)
    elif args.command == "register-model":
        cmd_register_model(args, config)
    elif args.command == "create-model-version":
        cmd_create_model_version(args, config)
    elif args.command == "release-model-version":
        cmd_release_model_version(args, config)
    elif args.command == "create-license":
        cmd_create_license(args, config)
    elif args.command == "list-models":
        cmd_list_models(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 