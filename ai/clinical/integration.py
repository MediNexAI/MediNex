"""
MediNex AI Clinical Integration Module

This module provides integration between clinical decision support and other MediNex AI components.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

from ..llm.model_connector import MedicalLLMConnector
from ..knowledge.medical_rag import MedicalRAG
from .decision_support import ClinicalDecisionSupport
from ..models.image_analysis import MedicalImageAnalysis


class ClinicalIntegration:
    """
    Integration between clinical decision support and other MediNex AI components.
    
    This class connects:
    - Clinical decision support
    - Medical image analysis
    - Medical RAG system
    - LLM connector
    
    To provide comprehensive clinical workflows.
    """
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        knowledge_base_path: Optional[str] = None,
        image_model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the clinical integration.
        
        Args:
            llm_config: Configuration for the LLM connector
            knowledge_base_path: Optional path to the knowledge base directory
            image_model_config: Optional configuration for image analysis models
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM connector
        try:
            self.llm = MedicalLLMConnector(llm_config)
            self.llm.connect()
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM connector: {str(e)}")
            raise
        
        # Initialize RAG system if knowledge base path is provided
        self.rag = None
        if knowledge_base_path:
            try:
                from ..knowledge.medical_knowledge_base import MedicalKnowledgeBase
                
                # Initialize knowledge base
                knowledge_base = MedicalKnowledgeBase(base_path=knowledge_base_path)
                
                # Initialize RAG with knowledge base
                self.rag = MedicalRAG(
                    knowledge_base=knowledge_base,
                    llm_config=llm_config
                )
                
                self.logger.info(f"Initialized RAG system with knowledge base at {knowledge_base_path}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize RAG system: {str(e)}")
        
        # Initialize clinical decision support
        self.cds = ClinicalDecisionSupport(
            llm_connector=self.llm,
            medical_rag=self.rag
        )
        
        # Initialize image analysis if config is provided
        self.image_analysis = None
        if image_model_config:
            try:
                self.image_analysis = MedicalImageAnalysis(image_model_config)
                self.logger.info("Initialized medical image analysis")
            except Exception as e:
                self.logger.warning(f"Failed to initialize image analysis: {str(e)}")
    
    def process_medical_images(
        self,
        image_paths: List[str],
        modality: Optional[str] = None,
        anatomy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process medical images and extract findings.
        
        Args:
            image_paths: List of paths to medical images
            modality: Optional imaging modality (e.g., "X-ray", "MRI", "CT")
            anatomy: Optional anatomical region (e.g., "chest", "brain", "abdomen")
            
        Returns:
            Dictionary containing image analysis results
        """
        if not self.image_analysis:
            raise ValueError("Image analysis not initialized")
        
        # Process images
        try:
            results = {}
            
            for image_path in image_paths:
                # Analyze image
                analysis_result = self.image_analysis.analyze_image(
                    image_path=image_path,
                    modality=modality,
                    anatomy=anatomy
                )
                
                # Add to results
                results[image_path] = analysis_result
            
            return {
                "status": "success",
                "image_count": len(image_paths),
                "results": results
            }
        except Exception as e:
            self.logger.error(f"Error processing medical images: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "image_count": len(image_paths)
            }
    
    def generate_diagnosis_from_images(
        self,
        image_paths: List[str],
        patient_info: Dict[str, Any],
        clinical_notes: Optional[str] = None,
        modality: Optional[str] = None,
        anatomy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate differential diagnosis based on medical images and patient information.
        
        Args:
            image_paths: List of paths to medical images
            patient_info: Patient demographic and basic information
            clinical_notes: Optional clinical notes about the case
            modality: Optional imaging modality
            anatomy: Optional anatomical region
            
        Returns:
            Dictionary containing diagnosis results
        """
        try:
            # Step 1: Process images
            image_results = self.process_medical_images(
                image_paths=image_paths,
                modality=modality,
                anatomy=anatomy
            )
            
            if image_results["status"] == "error":
                return image_results
            
            # Step 2: Extract findings from image analysis
            findings = []
            for path, result in image_results["results"].items():
                if "findings" in result:
                    findings.extend(result["findings"])
            
            # Step 3: Format image findings as symptoms
            symptoms = []
            for finding in findings:
                if isinstance(finding, dict) and "description" in finding:
                    symptoms.append(finding["description"])
                elif isinstance(finding, str):
                    symptoms.append(finding)
            
            # Add any clinical notes as additional symptoms/findings
            if clinical_notes:
                # Extract key findings from clinical notes
                notes_prompt = (
                    "Extract the key clinical findings and symptoms from the following clinical notes. "
                    "Return only a list of the findings, one per line:\n\n"
                    f"{clinical_notes}"
                )
                
                extracted_findings = self.llm.generate_text(notes_prompt)
                extracted_symptoms = [s.strip() for s in extracted_findings.split('\n') if s.strip()]
                symptoms.extend(extracted_symptoms)
            
            # Step 4: Generate differential diagnosis
            diagnosis_result = self.cds.get_differential_diagnosis(
                symptoms=symptoms,
                patient_info=patient_info
            )
            
            # Step 5: Add image metadata to result
            diagnosis_result["image_metadata"] = {
                "image_count": len(image_paths),
                "image_paths": image_paths,
                "modality": modality,
                "anatomy": anatomy
            }
            
            return diagnosis_result
            
        except Exception as e:
            self.logger.error(f"Error generating diagnosis from images: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_full_clinical_case(
        self,
        patient_info: Dict[str, Any],
        symptoms: List[str],
        medical_history: Optional[List[str]] = None,
        current_medications: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        vitals: Optional[Dict[str, Any]] = None,
        lab_results: Optional[Dict[str, Any]] = None,
        image_paths: Optional[List[str]] = None,
        clinical_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a full clinical case and provide comprehensive analysis and recommendations.
        
        Args:
            patient_info: Patient demographic and basic information
            symptoms: List of reported symptoms
            medical_history: Optional list of medical history items
            current_medications: Optional list of current medications
            allergies: Optional list of allergies
            vitals: Optional dictionary of vital signs
            lab_results: Optional dictionary of laboratory results
            image_paths: Optional list of paths to medical images
            clinical_notes: Optional clinical notes about the case
            
        Returns:
            Dictionary containing comprehensive clinical analysis
        """
        try:
            results = {
                "patient_info": patient_info,
                "case_summary": {}
            }
            
            # Step 1: Process any available images
            if image_paths and self.image_analysis:
                image_results = self.process_medical_images(image_paths)
                results["imaging"] = image_results
                
                # Extract findings from images and add to symptoms
                if image_results["status"] == "success":
                    for path, img_result in image_results["results"].items():
                        if "findings" in img_result:
                            for finding in img_result["findings"]:
                                if isinstance(finding, dict) and "description" in finding:
                                    symptoms.append(finding["description"])
                                elif isinstance(finding, str):
                                    symptoms.append(finding)
            
            # Step 2: Generate differential diagnosis
            diagnosis_result = self.cds.get_differential_diagnosis(
                symptoms=symptoms,
                patient_info=patient_info,
                medical_history=medical_history,
                test_results=lab_results
            )
            results["differential_diagnosis"] = diagnosis_result
            
            # Get the most likely diagnosis
            most_likely_diagnosis = None
            if "diagnoses" in diagnosis_result and diagnosis_result["diagnoses"]:
                # Sort by confidence
                sorted_diagnoses = sorted(
                    diagnosis_result["diagnoses"], 
                    key=lambda d: d.get("confidence", 0), 
                    reverse=True
                )
                most_likely_diagnosis = sorted_diagnoses[0]["name"]
                
                # Add to case summary
                results["case_summary"]["most_likely_diagnosis"] = most_likely_diagnosis
                results["case_summary"]["confidence"] = sorted_diagnoses[0].get("confidence", 0)
            
            # Step 3: Generate treatment recommendations for the most likely diagnosis
            if most_likely_diagnosis:
                treatment_result = self.cds.get_treatment_recommendations(
                    diagnosis=most_likely_diagnosis,
                    patient_info=patient_info,
                    medical_history=medical_history,
                    current_medications=current_medications,
                    allergies=allergies
                )
                results["treatment_recommendations"] = treatment_result
                
                # Extract first-line treatments for the case summary
                if "first_line_treatments" in treatment_result:
                    results["case_summary"]["recommended_treatments"] = treatment_result["first_line_treatments"]
                
                # Step 4: Perform risk assessment
                risk_result = self.cds.assess_risk(
                    patient_info=patient_info,
                    medical_history=medical_history,
                    current_medications=current_medications,
                    vitals=vitals,
                    lab_results=lab_results,
                    condition=most_likely_diagnosis
                )
                results["risk_assessment"] = risk_result
                
                # Add risk level to case summary
                if "risk_level" in risk_result:
                    results["case_summary"]["risk_level"] = risk_result["risk_level"]
                
                # Step 5: Generate follow-up recommendations
                if "first_line_treatments" in treatment_result:
                    follow_up_result = self.cds.generate_follow_up(
                        diagnosis=most_likely_diagnosis,
                        treatment_plan=treatment_result["first_line_treatments"],
                        patient_info=patient_info,
                        visit_notes=clinical_notes
                    )
                    results["follow_up_recommendations"] = follow_up_result
            
            # Step 6: Generate a comprehensive case summary using LLM
            if clinical_notes or most_likely_diagnosis:
                summary_prompt = (
                    "Generate a concise clinical case summary based on the following information:\n\n"
                )
                
                if clinical_notes:
                    summary_prompt += f"Clinical Notes: {clinical_notes}\n\n"
                
                summary_prompt += f"Patient: {patient_info.get('age', 'Unknown')} year old {patient_info.get('sex', 'Unknown')}\n"
                summary_prompt += f"Symptoms: {', '.join(symptoms)}\n"
                
                if medical_history:
                    summary_prompt += f"Medical History: {', '.join(medical_history)}\n"
                
                if most_likely_diagnosis:
                    summary_prompt += f"Most Likely Diagnosis: {most_likely_diagnosis}\n"
                
                if "recommended_treatments" in results["case_summary"]:
                    treatments = results["case_summary"]["recommended_treatments"]
                    summary_prompt += f"Recommended Treatments: {', '.join(treatments)}\n"
                
                summary_result = self.llm.generate_text(summary_prompt)
                results["case_summary"]["narrative"] = summary_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing clinical case: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_clinical_report(
        self,
        case_result: Dict[str, Any],
        report_type: str = "comprehensive"
    ) -> str:
        """
        Generate a formatted clinical report based on case results.
        
        Args:
            case_result: The result from process_full_clinical_case()
            report_type: Type of report to generate (comprehensive, summary, technical)
            
        Returns:
            Formatted clinical report text
        """
        try:
            # Extract relevant information from the case result
            patient_info = case_result.get("patient_info", {})
            diagnosis = case_result.get("differential_diagnosis", {})
            treatment = case_result.get("treatment_recommendations", {})
            risk = case_result.get("risk_assessment", {})
            follow_up = case_result.get("follow_up_recommendations", {})
            case_summary = case_result.get("case_summary", {})
            
            # Create the report prompt based on report type
            if report_type == "summary":
                prompt = (
                    "Generate a concise clinical summary report for healthcare providers. "
                    "Include only the most important information in a brief format."
                )
            elif report_type == "technical":
                prompt = (
                    "Generate a detailed technical medical report with comprehensive clinical details "
                    "and technical medical terminology appropriate for specialists."
                )
            else:  # comprehensive (default)
                prompt = (
                    "Generate a comprehensive clinical report with all relevant patient information, "
                    "findings, assessments, and recommendations in a well-structured format."
                )
            
            # Add patient information
            prompt += "\n\n## Patient Information\n"
            for key, value in patient_info.items():
                prompt += f"{key.capitalize()}: {value}\n"
            
            # Add case summary if available
            if case_summary:
                prompt += "\n## Case Summary\n"
                if "narrative" in case_summary:
                    prompt += f"{case_summary['narrative']}\n"
                else:
                    for key, value in case_summary.items():
                        if key != "narrative":
                            prompt += f"{key.replace('_', ' ').capitalize()}: {value}\n"
            
            # Add diagnosis information
            if diagnosis and "diagnoses" in diagnosis:
                prompt += "\n## Differential Diagnosis\n"
                for idx, diag in enumerate(diagnosis["diagnoses"][:3], 1):  # Top 3 diagnoses
                    prompt += f"{idx}. {diag.get('name', 'Unknown')} "
                    prompt += f"(Confidence: {diag.get('confidence', 0):.2f})\n"
                    prompt += f"   Explanation: {diag.get('explanation', 'N/A')}\n"
                    if "supporting_evidence" in diag:
                        prompt += f"   Evidence: {diag.get('supporting_evidence', 'N/A')}\n"
            
            # Add treatment recommendations
            if treatment:
                prompt += "\n## Treatment Recommendations\n"
                
                if "first_line_treatments" in treatment:
                    prompt += "First-line Treatments:\n"
                    for item in treatment["first_line_treatments"]:
                        prompt += f"- {item}\n"
                
                if "special_considerations" in treatment:
                    prompt += "Special Considerations:\n"
                    for item in treatment["special_considerations"]:
                        prompt += f"- {item}\n"
                
                if "contraindications" in treatment:
                    prompt += "Contraindications:\n"
                    for item in treatment["contraindications"]:
                        prompt += f"- {item}\n"
            
            # Add risk assessment
            if risk:
                prompt += "\n## Risk Assessment\n"
                prompt += f"Risk Level: {risk.get('risk_level', 'Unknown')}\n"
                
                if "risk_factors" in risk:
                    prompt += "Risk Factors:\n"
                    risk_factors = risk["risk_factors"]
                    if isinstance(risk_factors, list):
                        for factor in risk_factors:
                            if isinstance(factor, dict):
                                prompt += f"- {factor.get('factor', 'Unknown')}\n"
                            else:
                                prompt += f"- {factor}\n"
                    elif isinstance(risk_factors, dict):
                        for factor, value in risk_factors.items():
                            prompt += f"- {factor}\n"
                
                if "recommendations" in risk:
                    prompt += "Risk Mitigation:\n"
                    for item in risk["recommendations"]:
                        prompt += f"- {item}\n"
            
            # Add follow-up recommendations
            if follow_up:
                prompt += "\n## Follow-up Recommendations\n"
                
                if "timeline" in follow_up:
                    prompt += "Timeline:\n"
                    timeline = follow_up["timeline"]
                    if isinstance(timeline, list):
                        for item in timeline:
                            if isinstance(item, dict) and "time" in item and "event" in item:
                                prompt += f"- {item['time']}: {item['event']}\n"
                            elif isinstance(item, str):
                                prompt += f"- {item}\n"
                    elif isinstance(timeline, dict):
                        for time, event in timeline.items():
                            prompt += f"- {time}: {event}\n"
                
                if "warning_signs" in follow_up:
                    prompt += "Warning Signs:\n"
                    for item in follow_up["warning_signs"]:
                        prompt += f"- {item}\n"
            
            # Add final formatting instructions
            prompt += (
                "\nFormat the report in a professional clinical style with clear sections and headings. "
                "Use medical terminology appropriate for healthcare providers. "
                "Make sure to highlight any critical information that requires immediate attention."
            )
            
            # Generate the report
            report = self.llm.generate_text(prompt)
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating clinical report: {str(e)}")
            return f"Error generating report: {str(e)}" 