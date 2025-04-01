"""
MediNex AI Main API Module

This module provides the FastAPI application for the MediNex AI system.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import MediNex AI modules
from ai.llm.model_connector import MedicalLLMConnector
from ai.knowledge.medical_rag import MedicalRAG
from ai.knowledge.medical_knowledge_base import MedicalKnowledgeBase
from ai.clinical.decision_support import ClinicalDecisionSupport
from ai.clinical.integration import ClinicalIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Create FastAPI application
app = FastAPI(
    title="MediNex AI API",
    description="API for the MediNex AI medical assistant system",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration from environment variables
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "openai"),
    "model": os.getenv("LLM_MODEL", "gpt-4"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "api_key": os.getenv("LLM_API_KEY")
}

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge")

# Initialize components
llm_connector = None
knowledge_base = None
medical_rag = None
clinical_integration = None

# Dependency for getting LLM connector
def get_llm_connector():
    global llm_connector
    if llm_connector is None:
        try:
            llm_connector = MedicalLLMConnector(LLM_CONFIG)
            llm_connector.connect()
        except Exception as e:
            logger.error(f"Failed to initialize LLM connector: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize LLM connector: {str(e)}"
            )
    return llm_connector

# Dependency for getting RAG system
def get_rag_system():
    global medical_rag, knowledge_base
    if medical_rag is None:
        try:
            # Initialize knowledge base
            knowledge_base = MedicalKnowledgeBase(base_path=KNOWLEDGE_BASE_PATH)
            
            # Initialize RAG with knowledge base
            medical_rag = MedicalRAG(
                knowledge_base=knowledge_base,
                llm_config=LLM_CONFIG
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize RAG system: {str(e)}"
            )
    return medical_rag

# Dependency for getting clinical integration
def get_clinical_integration():
    global clinical_integration
    if clinical_integration is None:
        try:
            clinical_integration = ClinicalIntegration(
                llm_config=LLM_CONFIG,
                knowledge_base_path=KNOWLEDGE_BASE_PATH
            )
        except Exception as e:
            logger.error(f"Failed to initialize clinical integration: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize clinical integration: {str(e)}"
            )
    return clinical_integration

# Define API models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The medical question or query")
    category: Optional[str] = Field(None, description="Optional medical category to filter by")
    include_sources: bool = Field(True, description="Whether to include source information")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The response to the query")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source information if requested")
    has_relevant_context: bool = Field(..., description="Whether relevant context was found")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")

class PatientInfo(BaseModel):
    name: Optional[str] = Field(None, description="Patient name")
    age: int = Field(..., description="Patient age in years")
    sex: str = Field(..., description="Patient biological sex (male/female)")
    weight: Optional[str] = Field(None, description="Patient weight")
    height: Optional[str] = Field(None, description="Patient height")
    bmi: Optional[float] = Field(None, description="Patient BMI")

class DiagnosisRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of reported symptoms")
    patient_info: PatientInfo = Field(..., description="Patient information")
    medical_history: Optional[List[str]] = Field(None, description="List of medical history items")
    test_results: Optional[Dict[str, Any]] = Field(None, description="Test results if available")

class TreatmentRequest(BaseModel):
    diagnosis: str = Field(..., description="The diagnosis to get treatment for")
    patient_info: PatientInfo = Field(..., description="Patient information")
    medical_history: Optional[List[str]] = Field(None, description="List of medical history items")
    current_medications: Optional[List[str]] = Field(None, description="List of current medications")
    allergies: Optional[List[str]] = Field(None, description="List of allergies")

class RiskAssessmentRequest(BaseModel):
    patient_info: PatientInfo = Field(..., description="Patient information")
    medical_history: Optional[List[str]] = Field(None, description="List of medical history items")
    current_medications: Optional[List[str]] = Field(None, description="List of current medications")
    vitals: Optional[Dict[str, Any]] = Field(None, description="Dictionary of vital signs")
    lab_results: Optional[Dict[str, Any]] = Field(None, description="Dictionary of laboratory results")
    condition: Optional[str] = Field(None, description="Specific condition to assess risk for")

class FollowUpRequest(BaseModel):
    diagnosis: str = Field(..., description="The diagnosis")
    treatment_plan: List[str] = Field(..., description="List of treatment plan items")
    patient_info: PatientInfo = Field(..., description="Patient information")
    visit_notes: Optional[str] = Field(None, description="Clinical notes from the visit")
    time_frame: Optional[str] = Field("short-term", description="Time frame for follow-up")

class ClinicalCaseRequest(BaseModel):
    patient_info: PatientInfo = Field(..., description="Patient information")
    symptoms: List[str] = Field(..., description="List of reported symptoms")
    medical_history: Optional[List[str]] = Field(None, description="List of medical history items")
    current_medications: Optional[List[str]] = Field(None, description="List of current medications")
    allergies: Optional[List[str]] = Field(None, description="List of allergies")
    vitals: Optional[Dict[str, Any]] = Field(None, description="Dictionary of vital signs")
    lab_results: Optional[Dict[str, Any]] = Field(None, description="Dictionary of laboratory results")
    clinical_notes: Optional[str] = Field(None, description="Clinical notes about the case")

class ReportGenerationRequest(BaseModel):
    case_result: Dict[str, Any] = Field(..., description="The result from process_full_clinical_case()")
    report_type: str = Field("comprehensive", description="Type of report")

# Define API endpoints
@app.get("/")
async def root():
    """Root endpoint providing basic API information"""
    return {
        "message": "Welcome to MediNex AI API",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag: MedicalRAG = Depends(get_rag_system)
):
    """
    Query the medical knowledge base using the RAG system
    """
    start_time = datetime.now()
    
    try:
        result = rag.query(
            query=request.query,
            category=request.category,
            include_sources=request.include_sources
        )
        
        sources = result.get("sources", []) if request.include_sources else None
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": result["response"],
            "sources": sources,
            "has_relevant_context": result["has_relevant_context"],
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clinical/diagnosis")
async def get_differential_diagnosis(
    request: DiagnosisRequest,
    clinical: ClinicalIntegration = Depends(get_clinical_integration)
):
    """
    Generate differential diagnosis based on symptoms and patient information
    """
    try:
        result = clinical.cds.get_differential_diagnosis(
            symptoms=request.symptoms,
            patient_info=request.patient_info.dict(exclude_none=True),
            medical_history=request.medical_history,
            test_results=request.test_results
        )
        
        return result
    except Exception as e:
        logger.error(f"Error generating differential diagnosis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clinical/treatment")
async def get_treatment_recommendations(
    request: TreatmentRequest,
    clinical: ClinicalIntegration = Depends(get_clinical_integration)
):
    """
    Generate treatment recommendations based on diagnosis and patient information
    """
    try:
        result = clinical.cds.get_treatment_recommendations(
            diagnosis=request.diagnosis,
            patient_info=request.patient_info.dict(exclude_none=True),
            medical_history=request.medical_history,
            current_medications=request.current_medications,
            allergies=request.allergies
        )
        
        return result
    except Exception as e:
        logger.error(f"Error generating treatment recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clinical/risk")
async def assess_risk(
    request: RiskAssessmentRequest,
    clinical: ClinicalIntegration = Depends(get_clinical_integration)
):
    """
    Assess patient risk based on medical information
    """
    try:
        result = clinical.cds.assess_risk(
            patient_info=request.patient_info.dict(exclude_none=True),
            medical_history=request.medical_history,
            current_medications=request.current_medications,
            vitals=request.vitals,
            lab_results=request.lab_results,
            condition=request.condition
        )
        
        return result
    except Exception as e:
        logger.error(f"Error generating risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clinical/followup")
async def generate_follow_up(
    request: FollowUpRequest,
    clinical: ClinicalIntegration = Depends(get_clinical_integration)
):
    """
    Generate follow-up recommendations based on diagnosis and treatment plan
    """
    try:
        result = clinical.cds.generate_follow_up(
            diagnosis=request.diagnosis,
            treatment_plan=request.treatment_plan,
            patient_info=request.patient_info.dict(exclude_none=True),
            visit_notes=request.visit_notes,
            time_frame=request.time_frame
        )
        
        return result
    except Exception as e:
        logger.error(f"Error generating follow-up recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clinical/case")
async def process_clinical_case(
    request: ClinicalCaseRequest,
    clinical: ClinicalIntegration = Depends(get_clinical_integration)
):
    """
    Process a complete clinical case and provide comprehensive analysis
    """
    try:
        result = clinical.process_full_clinical_case(
            patient_info=request.patient_info.dict(exclude_none=True),
            symptoms=request.symptoms,
            medical_history=request.medical_history,
            current_medications=request.current_medications,
            allergies=request.allergies,
            vitals=request.vitals,
            lab_results=request.lab_results,
            clinical_notes=request.clinical_notes
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing clinical case: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clinical/report")
async def generate_clinical_report(
    request: ReportGenerationRequest,
    clinical: ClinicalIntegration = Depends(get_clinical_integration)
):
    """
    Generate a formatted clinical report based on case results
    """
    try:
        report = clinical.generate_clinical_report(
            case_result=request.case_result,
            report_type=request.report_type
        )
        
        return {"report": report}
    except Exception as e:
        logger.error(f"Error generating clinical report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 