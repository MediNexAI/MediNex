"""
MediNex AI Clinical Decision Support System

This module implements the clinical decision support system for medical diagnosis and treatment recommendations.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..llm.model_connector import MedicalLLMConnector
from ..knowledge.medical_rag import MedicalRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PatientInfo:
    """Patient information."""
    age: int
    gender: str
    symptoms: List[str]
    medical_history: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    vital_signs: Optional[Dict[str, float]] = None
    lab_results: Optional[Dict[str, Any]] = None

@dataclass
class Diagnosis:
    """Medical diagnosis."""
    condition: str
    confidence: float
    evidence: List[str]
    icd_code: Optional[str] = None
    differential_diagnoses: Optional[List[Dict[str, Any]]] = None

@dataclass
class TreatmentPlan:
    """Treatment plan."""
    recommendations: List[str]
    medications: List[Dict[str, Any]]
    follow_up: Dict[str, Any]
    precautions: List[str]
    monitoring: List[str]

class ClinicalDecisionSupport:
    """
    Clinical Decision Support System for medical diagnosis and treatment recommendations.
    """
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        rag_config: Dict[str, Any],
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the clinical decision support system.
        
        Args:
            llm_config: Configuration for the LLM
            rag_config: Configuration for the RAG system
            cache_dir: Directory for caching
        """
        self.llm = MedicalLLMConnector(llm_config)
        self.rag = MedicalRAG(rag_config)
        self.cache_dir = cache_dir
        
        # Create cache directory if needed
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def analyze_symptoms(
        self,
        patient_info: PatientInfo
    ) -> List[Dict[str, Any]]:
        """
        Analyze patient symptoms.
        
        Args:
            patient_info: Patient information
            
        Returns:
            List of potential conditions with confidence scores
        """
        # Prepare context
        context = (
            f"Patient Information:\n"
            f"Age: {patient_info.age}\n"
            f"Gender: {patient_info.gender}\n"
            f"Symptoms: {', '.join(patient_info.symptoms)}\n"
        )
        
        if patient_info.medical_history:
            context += f"Medical History: {', '.join(patient_info.medical_history)}\n"
        if patient_info.medications:
            context += f"Current Medications: {', '.join(patient_info.medications)}\n"
        if patient_info.allergies:
            context += f"Allergies: {', '.join(patient_info.allergies)}\n"
        if patient_info.vital_signs:
            context += "Vital Signs:\n"
            for key, value in patient_info.vital_signs.items():
                context += f"- {key}: {value}\n"
        
        # Query RAG system
        response = self.rag.query(
            f"Based on the following patient information, what are the most likely diagnoses? {context}"
        )
        
        # Process response
        conditions = []
        if response["used_rag"]:
            # Extract structured information from LLM response
            prompt = (
                f"Extract potential diagnoses from this medical analysis:\n{response['answer']}\n\n"
                "Format each diagnosis as a JSON object with fields:\n"
                "- condition: name of the condition\n"
                "- confidence: confidence score (0-1)\n"
                "- evidence: list of supporting evidence\n"
                "- icd_code: ICD-10 code if available"
            )
            
            structured_response = self.llm.generate_text(prompt)
            try:
                conditions = json.loads(structured_response)
            except json.JSONDecodeError:
                logger.error("Failed to parse structured diagnosis information")
                conditions = []
        
        return conditions
    
    def generate_diagnosis(
        self,
        patient_info: PatientInfo,
        conditions: Optional[List[Dict[str, Any]]] = None
    ) -> Diagnosis:
        """
        Generate a diagnosis based on patient information.
        
        Args:
            patient_info: Patient information
            conditions: Optional list of pre-analyzed conditions
            
        Returns:
            Diagnosis object
        """
        if not conditions:
            conditions = self.analyze_symptoms(patient_info)
        
        if not conditions:
            return Diagnosis(
                condition="Unable to determine",
                confidence=0.0,
                evidence=["Insufficient information"],
                differential_diagnoses=[]
            )
        
        # Sort conditions by confidence
        sorted_conditions = sorted(
            conditions,
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )
        
        # Get primary diagnosis
        primary = sorted_conditions[0]
        
        # Create differential diagnoses list
        differential = [
            {
                "condition": c["condition"],
                "confidence": c["confidence"],
                "evidence": c.get("evidence", []),
                "icd_code": c.get("icd_code")
            }
            for c in sorted_conditions[1:4]  # Top 3 alternatives
        ]
        
        return Diagnosis(
            condition=primary["condition"],
            confidence=primary["confidence"],
            evidence=primary.get("evidence", []),
            icd_code=primary.get("icd_code"),
            differential_diagnoses=differential
        )
    
    def recommend_treatment(
        self,
        diagnosis: Diagnosis,
        patient_info: PatientInfo
    ) -> TreatmentPlan:
        """
        Generate treatment recommendations.
        
        Args:
            diagnosis: Diagnosis object
            patient_info: Patient information
            
        Returns:
            TreatmentPlan object
        """
        # Prepare context
        context = (
            f"Generate a treatment plan for:\n"
            f"Diagnosis: {diagnosis.condition}\n"
            f"Patient Age: {patient_info.age}\n"
            f"Patient Gender: {patient_info.gender}\n"
        )
        
        if patient_info.medical_history:
            context += f"Medical History: {', '.join(patient_info.medical_history)}\n"
        if patient_info.medications:
            context += f"Current Medications: {', '.join(patient_info.medications)}\n"
        if patient_info.allergies:
            context += f"Allergies: {', '.join(patient_info.allergies)}\n"
        
        # Query RAG system
        response = self.rag.query(context)
        
        # Process response
        prompt = (
            f"Extract treatment recommendations from this medical analysis:\n{response['answer']}\n\n"
            "Format the treatment plan as a JSON object with fields:\n"
            "- recommendations: list of general recommendations\n"
            "- medications: list of medication objects (name, dosage, frequency, duration)\n"
            "- follow_up: object with follow-up details (timing, tests, specialist referrals)\n"
            "- precautions: list of precautions and warnings\n"
            "- monitoring: list of parameters to monitor"
        )
        
        structured_response = self.llm.generate_text(prompt)
        try:
            plan_data = json.loads(structured_response)
            return TreatmentPlan(
                recommendations=plan_data["recommendations"],
                medications=plan_data["medications"],
                follow_up=plan_data["follow_up"],
                precautions=plan_data["precautions"],
                monitoring=plan_data["monitoring"]
            )
        except (json.JSONDecodeError, KeyError):
            logger.error("Failed to parse treatment plan")
            return TreatmentPlan(
                recommendations=["Unable to generate treatment plan"],
                medications=[],
                follow_up={},
                precautions=[],
                monitoring=[]
            )
    
    def assess_risk(
        self,
        diagnosis: Diagnosis,
        patient_info: PatientInfo,
        treatment_plan: Optional[TreatmentPlan] = None
    ) -> Dict[str, Any]:
        """
        Assess patient risks.
        
        Args:
            diagnosis: Diagnosis object
            patient_info: Patient information
            treatment_plan: Optional treatment plan
            
        Returns:
            Risk assessment dictionary
        """
        # Prepare context
        context = (
            f"Assess risks for patient with:\n"
            f"Diagnosis: {diagnosis.condition}\n"
            f"Age: {patient_info.age}\n"
            f"Gender: {patient_info.gender}\n"
        )
        
        if patient_info.medical_history:
            context += f"Medical History: {', '.join(patient_info.medical_history)}\n"
        if treatment_plan and treatment_plan.medications:
            context += "Prescribed Medications:\n"
            for med in treatment_plan.medications:
                context += f"- {med['name']}: {med['dosage']}, {med['frequency']}\n"
        
        # Query RAG system
        response = self.rag.query(context)
        
        # Process response
        prompt = (
            f"Extract risk assessment from this medical analysis:\n{response['answer']}\n\n"
            "Format the assessment as a JSON object with fields:\n"
            "- overall_risk_level: string (low/medium/high)\n"
            "- specific_risks: list of risk objects (risk, likelihood, severity, mitigation)\n"
            "- monitoring_recommendations: list of parameters to monitor\n"
            "- warning_signs: list of warning signs to watch for"
        )
        
        structured_response = self.llm.generate_text(prompt)
        try:
            return json.loads(structured_response)
        except json.JSONDecodeError:
            logger.error("Failed to parse risk assessment")
            return {
                "overall_risk_level": "unknown",
                "specific_risks": [],
                "monitoring_recommendations": [],
                "warning_signs": []
            }
    
    def generate_report(
        self,
        patient_info: PatientInfo,
        diagnosis: Diagnosis,
        treatment_plan: TreatmentPlan,
        risk_assessment: Dict[str, Any]
    ) -> str:
        """
        Generate a clinical report.
        
        Args:
            patient_info: Patient information
            diagnosis: Diagnosis object
            treatment_plan: Treatment plan
            risk_assessment: Risk assessment
            
        Returns:
            Formatted clinical report
        """
        report = [
            "CLINICAL REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            
            "\nPATIENT INFORMATION",
            "-" * 80,
            f"Age: {patient_info.age}",
            f"Gender: {patient_info.gender}",
            f"Presenting Symptoms: {', '.join(patient_info.symptoms)}",
        ]
        
        if patient_info.medical_history:
            report.append(f"Medical History: {', '.join(patient_info.medical_history)}")
        if patient_info.medications:
            report.append(f"Current Medications: {', '.join(patient_info.medications)}")
        if patient_info.allergies:
            report.append(f"Allergies: {', '.join(patient_info.allergies)}")
        
        report.extend([
            "\nDIAGNOSIS",
            "-" * 80,
            f"Primary Diagnosis: {diagnosis.condition}",
            f"Confidence: {diagnosis.confidence:.1%}",
            "\nSupporting Evidence:",
            *[f"- {e}" for e in diagnosis.evidence],
            
            "\nDifferential Diagnoses:",
            *[f"- {d['condition']} (Confidence: {d['confidence']:.1%})"
              for d in (diagnosis.differential_diagnoses or [])]
        ])
        
        report.extend([
            "\nTREATMENT PLAN",
            "-" * 80,
            "\nRecommendations:",
            *[f"- {r}" for r in treatment_plan.recommendations],
            
            "\nMedications:",
            *[f"- {m['name']}: {m['dosage']}, {m['frequency']}"
              for m in treatment_plan.medications],
            
            "\nFollow-up Plan:",
            *[f"- {k}: {v}" for k, v in treatment_plan.follow_up.items()],
            
            "\nPrecautions:",
            *[f"- {p}" for p in treatment_plan.precautions],
            
            "\nMonitoring:",
            *[f"- {m}" for m in treatment_plan.monitoring]
        ])
        
        report.extend([
            "\nRISK ASSESSMENT",
            "-" * 80,
            f"Overall Risk Level: {risk_assessment['overall_risk_level'].upper()}",
            
            "\nSpecific Risks:",
            *[f"- {r['risk']} (Likelihood: {r['likelihood']}, Severity: {r['severity']})"
              for r in risk_assessment['specific_risks']],
            
            "\nWarning Signs:",
            *[f"- {w}" for w in risk_assessment['warning_signs']]
        ])
        
        return "\n".join(report)
    
    def save_case(
        self,
        patient_info: PatientInfo,
        diagnosis: Diagnosis,
        treatment_plan: TreatmentPlan,
        risk_assessment: Dict[str, Any],
        output_dir: str
    ) -> str:
        """
        Save case information.
        
        Args:
            patient_info: Patient information
            diagnosis: Diagnosis object
            treatment_plan: Treatment plan
            risk_assessment: Risk assessment
            output_dir: Output directory
            
        Returns:
            Path to saved case file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate case ID
        case_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"case_{case_id}.json")
        
        # Prepare case data
        case_data = {
            "timestamp": datetime.now().isoformat(),
            "patient_info": {
                "age": patient_info.age,
                "gender": patient_info.gender,
                "symptoms": patient_info.symptoms,
                "medical_history": patient_info.medical_history,
                "medications": patient_info.medications,
                "allergies": patient_info.allergies,
                "vital_signs": patient_info.vital_signs,
                "lab_results": patient_info.lab_results
            },
            "diagnosis": {
                "condition": diagnosis.condition,
                "confidence": diagnosis.confidence,
                "evidence": diagnosis.evidence,
                "icd_code": diagnosis.icd_code,
                "differential_diagnoses": diagnosis.differential_diagnoses
            },
            "treatment_plan": {
                "recommendations": treatment_plan.recommendations,
                "medications": treatment_plan.medications,
                "follow_up": treatment_plan.follow_up,
                "precautions": treatment_plan.precautions,
                "monitoring": treatment_plan.monitoring
            },
            "risk_assessment": risk_assessment
        }
        
        # Save case data
        with open(output_path, "w") as f:
            json.dump(case_data, f, indent=2)
        
        return output_path 