"""
MediNex AI API Core Module

This module provides the core FastAPI implementation for the MediNex AI system,
creating endpoints for accessing all functionality including knowledge base
queries, LLM interactions, and medical imaging analysis.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Request/Response Models
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class QueryRequest(BaseModel):
    query: str
    use_knowledge: bool = True
    system_prompt: Optional[str] = None
    clinical_context: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    model: str
    sources: Optional[List[Dict[str, Any]]] = None
    processing_time: float


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class KnowledgeSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float


class DocumentAddRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_id: Optional[str] = None


class DocumentAddResponse(BaseModel):
    success: bool
    doc_id: str
    processing_time: float


class ImageAnalysisResponse(BaseModel):
    success: bool
    modality: str
    analysis_type: str
    vision_analysis: Optional[Dict[str, Any]] = None
    llm_interpretation: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: str


# Contributor Management Models
class ContributorCreate(BaseModel):
    name: str
    email: str
    institution: Optional[str] = None
    specialization: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContributorUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    institution: Optional[str] = None
    specialization: Optional[str] = None
    active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class ContributorResponse(BaseModel):
    contributor_id: str
    name: str
    email: str
    institution: Optional[str] = None
    specialization: Optional[str] = None
    active: bool
    join_date: str
    contributions: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ContributionRecord(BaseModel):
    contributor_id: str
    contribution_type: str
    description: str
    value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Revenue Sharing Models
class RevenuePeriodCreate(BaseModel):
    name: str
    start_date: str
    end_date: str
    total_revenue: float
    currency: str = "USD"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RevenueReportRequest(BaseModel):
    period_id: str
    detailed: bool = False


class RevenueShareResponse(BaseModel):
    success: bool
    period_id: Optional[str] = None
    shares: Optional[List[Dict[str, Any]]] = None
    processing_time: float


# Model Distribution Models
class ModelRegister(BaseModel):
    name: str
    description: str
    model_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelVersionCreate(BaseModel):
    model_id: str
    version_number: str
    description: str
    config: Dict[str, Any]
    contributors: List[str] = Field(default_factory=list)


class LicenseCreate(BaseModel):
    version_id: str
    user_id: str
    license_type: str
    expiration_date: Optional[str] = None
    usage_limits: Dict[str, Any] = Field(default_factory=dict)
    custom_terms: Optional[str] = None


class DeploymentRegister(BaseModel):
    version_id: str
    license_id: str
    deployment_name: str
    environment: str
    endpoint_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeploymentHeartbeat(BaseModel):
    deployment_id: str
    status: str = "active"
    stats: Dict[str, Any] = Field(default_factory=dict)


def create_api(components: Dict[str, Any]) -> FastAPI:
    """
    Create and configure the FastAPI application with all endpoints.
    
    Args:
        components: Dictionary containing initialized system components
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="MediNex AI API",
        description="API for medical knowledge retrieval, LLM-based medical reasoning, and medical imaging analysis",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Extract components
    rag = components.get("rag")
    knowledge_base = components.get("knowledge_base")
    llm_connector = components.get("llm_connector")
    imaging_pipeline = components.get("imaging_pipeline")
    contributor_manager = components.get("contributor_manager")
    revenue_system = components.get("revenue_system")
    model_distributor = components.get("model_distributor")
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check if the API is healthy and return basic system information."""
        return {
            "status": "healthy",
            "version": app.version,
            "timestamp": datetime.now().isoformat()
        }
    
    # LLM query endpoint
    @app.post("/query", response_model=QueryResponse)
    async def query_llm(request: QueryRequest):
        """
        Query the medical LLM with optional knowledge retrieval enhancement.
        """
        if not llm_connector:
            raise HTTPException(status_code=503, detail="LLM service not available")
        
        start_time = time.time()
        
        try:
            if request.use_knowledge and rag:
                # Use RAG for knowledge-enhanced responses
                result = rag.query(
                    query=request.query,
                    system_prompt=request.system_prompt,
                    clinical_context=request.clinical_context
                )
                
                return {
                    "answer": result.get("answer", ""),
                    "model": result.get("model", ""),
                    "sources": result.get("sources", []),
                    "processing_time": time.time() - start_time
                }
            else:
                # Direct LLM query without knowledge enhancement
                response = llm_connector.generate_response(
                    query=request.query,
                    system_prompt=request.system_prompt,
                    context=request.clinical_context
                )
                
                return {
                    "answer": response.get("text", ""),
                    "model": response.get("model", ""),
                    "sources": None,
                    "processing_time": time.time() - start_time
                }
        except Exception as e:
            logger.error(f"Error in LLM query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Knowledge base search endpoint
    @app.post("/knowledge/search", response_model=KnowledgeSearchResponse)
    async def search_knowledge(request: KnowledgeSearchRequest):
        """
        Search the medical knowledge base using semantic search.
        """
        if not knowledge_base:
            raise HTTPException(status_code=503, detail="Knowledge base not available")
        
        start_time = time.time()
        
        try:
            search_results = knowledge_base.search(
                query=request.query,
                top_k=request.top_k
            )
            
            return {
                "results": search_results,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Error in knowledge search: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Add document to knowledge base endpoint
    @app.post("/knowledge/add", response_model=DocumentAddResponse)
    async def add_document(request: DocumentAddRequest):
        """
        Add a document to the medical knowledge base.
        """
        if not knowledge_base:
            raise HTTPException(status_code=503, detail="Knowledge base not available")
        
        start_time = time.time()
        
        try:
            doc_id = knowledge_base.add_document(
                content=request.content,
                metadata=request.metadata,
                doc_id=request.doc_id
            )
            
            return {
                "success": True,
                "doc_id": doc_id,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Medical image analysis endpoint
    @app.post("/imaging/analyze", response_model=ImageAnalysisResponse)
    async def analyze_image(
        file: UploadFile = File(...),
        modality: str = Form(...),
        analysis_type: str = Form("general"),
        clinical_context: Optional[str] = Form(None),
        patient_info: Optional[str] = Form(None)
    ):
        """
        Analyze a medical image and generate an interpretation.
        """
        if not imaging_pipeline:
            raise HTTPException(status_code=503, detail="Imaging analysis not available")
        
        start_time = time.time()
        
        try:
            # Read image file
            contents = await file.read()
            image = BytesIO(contents)
            
            # Parse patient info if provided
            patient_data = json.loads(patient_info) if patient_info else None
            
            # Analyze the image
            result = imaging_pipeline.analyze_image(
                image_path=image,
                modality=modality,
                analysis_type=analysis_type,
                clinical_context=clinical_context,
                patient_info=patient_data
            )
            
            return {
                "success": result.get("success", False),
                "modality": modality,
                "analysis_type": analysis_type,
                "vision_analysis": result.get("vision_analysis"),
                "llm_interpretation": result.get("llm_interpretation"),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # =============== Contributor Management Endpoints ===============
    
    @app.post("/contributors/register", response_model=ContributorResponse)
    async def register_contributor(request: ContributorCreate):
        """
        Register a new contributor to the MediNex AI system.
        """
        if not contributor_manager:
            raise HTTPException(status_code=503, detail="Contributor management not available")
        
        try:
            contributor = contributor_manager.register_contributor(
                name=request.name,
                email=request.email,
                institution=request.institution,
                specialization=request.specialization,
                metadata=request.metadata
            )
            
            return contributor
        except Exception as e:
            logger.error(f"Error registering contributor: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/contributors/{contributor_id}", response_model=ContributorResponse)
    async def get_contributor(contributor_id: str):
        """
        Get information about a specific contributor.
        """
        if not contributor_manager:
            raise HTTPException(status_code=503, detail="Contributor management not available")
        
        try:
            contributor = contributor_manager.get_contributor(contributor_id)
            if not contributor:
                raise HTTPException(status_code=404, detail="Contributor not found")
            
            return contributor
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting contributor: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/contributors/{contributor_id}", response_model=ContributorResponse)
    async def update_contributor(contributor_id: str, request: ContributorUpdate):
        """
        Update information for an existing contributor.
        """
        if not contributor_manager:
            raise HTTPException(status_code=503, detail="Contributor management not available")
        
        try:
            contributor = contributor_manager.update_contributor(
                contributor_id=contributor_id,
                name=request.name,
                email=request.email,
                institution=request.institution,
                specialization=request.specialization,
                active=request.active,
                metadata=request.metadata
            )
            
            if not contributor:
                raise HTTPException(status_code=404, detail="Contributor not found")
            
            return contributor
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating contributor: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/contributors", response_model=List[ContributorResponse])
    async def list_contributors(active_only: bool = Query(False)):
        """
        List all contributors, optionally filtering by active status.
        """
        if not contributor_manager:
            raise HTTPException(status_code=503, detail="Contributor management not available")
        
        try:
            contributors = contributor_manager.get_contributors(active_only=active_only)
            return contributors
        except Exception as e:
            logger.error(f"Error listing contributors: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/contributions/record")
    async def record_contribution(request: ContributionRecord):
        """
        Record a contribution from a contributor.
        """
        if not contributor_manager:
            raise HTTPException(status_code=503, detail="Contributor management not available")
        
        try:
            success = contributor_manager.record_contribution(
                contributor_id=request.contributor_id,
                contribution_type=request.contribution_type,
                description=request.description,
                value=request.value,
                metadata=request.metadata
            )
            
            if not success:
                raise HTTPException(status_code=404, detail="Failed to record contribution")
            
            return {"success": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error recording contribution: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # =============== Revenue Sharing Endpoints ===============
    
    @app.post("/revenue/periods/create", response_model=Dict[str, Any])
    async def create_revenue_period(request: RevenuePeriodCreate):
        """
        Create a new revenue period for calculating contributor shares.
        """
        if not revenue_system:
            raise HTTPException(status_code=503, detail="Revenue sharing system not available")
        
        try:
            period = revenue_system.create_revenue_period(
                name=request.name,
                start_date=request.start_date,
                end_date=request.end_date,
                total_revenue=request.total_revenue,
                currency=request.currency,
                metadata=request.metadata
            )
            
            return period
        except Exception as e:
            logger.error(f"Error creating revenue period: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/revenue/periods", response_model=List[Dict[str, Any]])
    async def list_revenue_periods():
        """
        List all revenue periods.
        """
        if not revenue_system:
            raise HTTPException(status_code=503, detail="Revenue sharing system not available")
        
        try:
            periods = revenue_system.get_revenue_periods()
            return periods
        except Exception as e:
            logger.error(f"Error listing revenue periods: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/revenue/calculate", response_model=RevenueShareResponse)
    async def calculate_revenue_shares(request: RevenueReportRequest):
        """
        Calculate revenue shares for contributors in a specific period.
        """
        if not revenue_system:
            raise HTTPException(status_code=503, detail="Revenue sharing system not available")
        
        start_time = time.time()
        
        try:
            shares = revenue_system.calculate_shares(
                period_id=request.period_id,
                detailed=request.detailed
            )
            
            return {
                "success": True,
                "period_id": request.period_id,
                "shares": shares,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Error calculating revenue shares: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/revenue/periods/{period_id}/finalize")
    async def finalize_revenue_period(period_id: str):
        """
        Finalize a revenue period, locking in the calculations.
        """
        if not revenue_system:
            raise HTTPException(status_code=503, detail="Revenue sharing system not available")
        
        try:
            success = revenue_system.finalize_period(period_id)
            
            if not success:
                raise HTTPException(status_code=400, detail="Failed to finalize period")
            
            return {"success": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error finalizing revenue period: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # =============== Model Distribution Endpoints ===============
    
    @app.post("/models/register", response_model=Dict[str, Any])
    async def register_model(request: ModelRegister):
        """
        Register a new model in the distribution system.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            model = model_distributor.register_model(
                name=request.name,
                description=request.description,
                model_type=request.model_type,
                metadata=request.metadata
            )
            
            return model
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models", response_model=List[Dict[str, Any]])
    async def list_models(model_type: Optional[str] = None):
        """
        List all available models, optionally filtered by type.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            models = model_distributor.get_models(model_type=model_type)
            return models
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models/{model_id}", response_model=Dict[str, Any])
    async def get_model(model_id: str):
        """
        Get details about a specific model.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            model = model_distributor.get_model(model_id)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            return model
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/versions/create", response_model=Dict[str, Any])
    async def create_model_version(request: ModelVersionCreate, background_tasks: BackgroundTasks):
        """
        Create a new version of a model.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            # Check if artifacts directory exists, create a temporary one if not provided
            artifacts_path = f"./data/tmp/model_artifacts/{request.model_id}/{request.version_number}"
            os.makedirs(artifacts_path, exist_ok=True)
            
            version = model_distributor.create_version(
                model_id=request.model_id,
                version_number=request.version_number,
                description=request.description,
                artifacts_path=artifacts_path,
                config=request.config,
                contributors=request.contributors
            )
            
            if not version:
                raise HTTPException(status_code=400, detail="Failed to create version")
            
            return version.to_dict()
        except Exception as e:
            logger.error(f"Error creating model version: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/versions/{version_id}/release")
    async def release_model_version(version_id: str):
        """
        Release a model version, making it available for distribution.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            version = model_distributor.release_version(version_id)
            
            if not version:
                raise HTTPException(status_code=404, detail="Version not found")
            
            return {"success": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error releasing model version: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/versions/{version_id}/package")
    async def package_model_version(
        version_id: str, 
        package_name: Optional[str] = None,
        include_config: bool = True,
        include_readme: bool = True
    ):
        """
        Package a model version for distribution.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            package_path = model_distributor.package_version(
                version_id=version_id,
                package_name=package_name,
                include_config=include_config,
                include_readme=include_readme
            )
            
            if not package_path:
                raise HTTPException(status_code=400, detail="Failed to package version")
            
            return {"success": True, "package_path": package_path}
        except Exception as e:
            logger.error(f"Error packaging model version: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models/packages/{package_name}")
    async def download_model_package(package_name: str):
        """
        Download a packaged model.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            package_path = os.path.join(model_distributor.packages_dir, f"{package_name}.zip")
            
            if not os.path.exists(package_path):
                raise HTTPException(status_code=404, detail="Package not found")
            
            return FileResponse(
                path=package_path,
                filename=f"{package_name}.zip",
                media_type="application/zip"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading model package: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/licenses/create", response_model=Dict[str, Any])
    async def create_license(request: LicenseCreate):
        """
        Create a license for a model version.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            license_record = model_distributor.create_license(
                version_id=request.version_id,
                user_id=request.user_id,
                license_type=request.license_type,
                expiration_date=request.expiration_date,
                usage_limits=request.usage_limits,
                custom_terms=request.custom_terms
            )
            
            if not license_record:
                raise HTTPException(status_code=400, detail="Failed to create license")
            
            return license_record
        except Exception as e:
            logger.error(f"Error creating license: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/licenses/verify")
    async def verify_license(license_key: str, version_id: str):
        """
        Verify a license for a model version.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            verification = model_distributor.verify_license(
                license_key=license_key,
                version_id=version_id
            )
            
            return verification
        except Exception as e:
            logger.error(f"Error verifying license: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/deployments/register", response_model=Dict[str, Any])
    async def register_deployment(request: DeploymentRegister):
        """
        Register a model deployment.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            deployment = model_distributor.register_deployment(
                version_id=request.version_id,
                license_id=request.license_id,
                deployment_name=request.deployment_name,
                environment=request.environment,
                endpoint_url=request.endpoint_url,
                metadata=request.metadata
            )
            
            if not deployment:
                raise HTTPException(status_code=400, detail="Failed to register deployment")
            
            return deployment
        except Exception as e:
            logger.error(f"Error registering deployment: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/deployments/heartbeat")
    async def update_deployment_heartbeat(request: DeploymentHeartbeat):
        """
        Update the heartbeat for a deployment.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            success = model_distributor.update_deployment_heartbeat(
                deployment_id=request.deployment_id,
                status=request.status,
                stats=request.stats
            )
            
            if not success:
                raise HTTPException(status_code=404, detail="Deployment not found")
            
            return {"success": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating deployment heartbeat: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/deployments")
    async def list_deployments(
        version_id: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[str] = None
    ):
        """
        List deployments with optional filtering.
        """
        if not model_distributor:
            raise HTTPException(status_code=503, detail="Model distribution not available")
        
        try:
            deployments = model_distributor.get_deployments(
                version_id=version_id,
                environment=environment,
                status=status
            )
            
            return deployments
        except Exception as e:
            logger.error(f"Error listing deployments: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


if __name__ == "__main__":
    # This is for testing the API in isolation
    from ai.llm.model_connector import MedicalLLMConnector
    from ai.knowledge.medical_rag import MedicalKnowledgeBase, MedicalRAG
    from ai.integrations.imaging_llm_pipeline import MedicalImagingLLMPipeline
    
    # Create mock components
    mock_components = {}
    
    # Start API
    api = create_api(mock_components)
    uvicorn.run(api, host="0.0.0.0", port=8000) 