"""
Unit tests for the MediNex AI Clinical Decision Support module
"""

import unittest
import json
from unittest.mock import MagicMock, patch
import logging

from ai.clinical.decision_support import ClinicalDecisionSupport
from ai.llm.model_connector import MedicalLLMConnector


class TestClinicalDecisionSupport(unittest.TestCase):
    """Test cases for the ClinicalDecisionSupport class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock LLM connector
        self.mock_llm = MagicMock(spec=MedicalLLMConnector)
        
        # Configure the mock to return a valid response
        self.mock_llm.generate_text.return_value = json.dumps({
            "diagnoses": [
                {
                    "name": "Common Cold",
                    "confidence": 0.85,
                    "explanation": "The symptoms align with an upper respiratory viral infection.",
                    "supporting_evidence": "Runny nose, sore throat, mild fever.",
                    "suggested_tests": ["None needed, symptomatic treatment recommended"]
                },
                {
                    "name": "Seasonal Allergies",
                    "confidence": 0.65,
                    "explanation": "Symptoms can mimic a cold but are triggered by allergens.",
                    "supporting_evidence": "Runny nose, itchy eyes, seasonal pattern.",
                    "suggested_tests": ["Allergy panel", "IgE levels"]
                }
            ]
        })
        
        # Create the clinical decision support with the mock LLM
        self.cds = ClinicalDecisionSupport(self.mock_llm)
        
        # Test patient data
        self.test_patient = {
            "name": "John Doe",
            "age": 45,
            "sex": "male",
            "weight": "180 lbs",
            "height": "5'10\"",
            "bmi": 25.8
        }
        
        # Test symptoms
        self.test_symptoms = ["headache", "fever", "cough"]
        
        # Silence the logger for tests
        logging.getLogger('ai.clinical.decision_support').setLevel(logging.CRITICAL)
    
    @patch('ai.llm.model_connector.MedicalLLMConnector')
    def test_init(self, mock_llm_class):
        """Test initialization of the ClinicalDecisionSupport class"""
        # Configure the mock
        mock_instance = MagicMock()
        mock_llm_class.return_value = mock_instance
        
        # Create with confidence threshold
        cds = ClinicalDecisionSupport(mock_instance, confidence_threshold=0.8)
        
        # Check that attributes are set correctly
        self.assertEqual(cds.llm, mock_instance)
        self.assertEqual(cds.confidence_threshold, 0.8)
        self.assertIsNone(cds.rag)
        
        # Verify that connect was called
        mock_instance.connect.assert_called_once()
    
    def test_get_differential_diagnosis(self):
        """Test getting differential diagnosis"""
        # Call the method
        result = self.cds.get_differential_diagnosis(
            symptoms=self.test_symptoms,
            patient_info=self.test_patient
        )
        
        # Check that LLM was called with appropriate prompt
        args, _ = self.mock_llm.generate_text.call_args
        prompt = args[0]
        
        # Verify the prompt contains important elements
        self.assertIn("headache, fever, cough", prompt)
        self.assertIn("Age: 45", prompt)
        self.assertIn("Sex: male", prompt)
        
        # Check the result structure
        self.assertIn("diagnoses", result)
        self.assertIn("metadata", result)
        
        # Check the diagnoses
        diagnoses = result["diagnoses"]
        self.assertEqual(len(diagnoses), 2)
        self.assertEqual(diagnoses[0]["name"], "Common Cold")
        self.assertEqual(diagnoses[0]["confidence"], 0.85)
        
        # Check metadata
        metadata = result["metadata"]
        self.assertEqual(metadata["symptoms"], self.test_symptoms)
        self.assertEqual(metadata["confidence_threshold"], 0.7)
    
    def test_format_patient_info(self):
        """Test patient info formatting"""
        formatted = self.cds._format_patient_info(self.test_patient)
        
        # Check that all patient info is included
        self.assertIn("Name: John Doe", formatted)
        self.assertIn("Age: 45", formatted)
        self.assertIn("Sex: male", formatted)
        self.assertIn("Weight: 180 lbs", formatted)
        self.assertIn("Height: 5'10\"", formatted)
        self.assertIn("Bmi: 25.8", formatted)
    
    def test_get_treatment_recommendations(self):
        """Test getting treatment recommendations"""
        # Configure mock to return treatment recommendations
        self.mock_llm.generate_text.return_value = json.dumps({
            "first_line_treatments": [
                "Acetaminophen for fever and pain",
                "Rest and hydration"
            ],
            "alternative_treatments": [
                "NSAIDs if acetaminophen is not effective"
            ],
            "non_pharmacological": [
                "Increase fluid intake",
                "Humidifier for congestion"
            ],
            "special_considerations": [
                "Avoid NSAIDs if patient has history of gastric ulcers"
            ],
            "contraindications": [
                "No aspirin for children due to risk of Reye's syndrome"
            ],
            "follow_up": [
                "Follow up in 1 week if symptoms persist"
            ]
        })
        
        # Call the method
        result = self.cds.get_treatment_recommendations(
            diagnosis="Common Cold",
            patient_info=self.test_patient
        )
        
        # Check the prompt
        args, _ = self.mock_llm.generate_text.call_args
        prompt = args[0]
        
        # Verify the prompt contains important elements
        self.assertIn("Common Cold", prompt)
        self.assertIn("Age: 45", prompt)
        
        # Check the result
        self.assertIn("first_line_treatments", result)
        self.assertEqual(len(result["first_line_treatments"]), 2)
        self.assertIn("alternative_treatments", result)
        self.assertIn("non_pharmacological", result)
    
    def test_error_handling(self):
        """Test error handling in clinical decision support"""
        # Configure mock to raise an exception
        self.mock_llm.generate_text.side_effect = Exception("API Error")
        
        # Call the method and expect an error response
        result = self.cds.get_differential_diagnosis(
            symptoms=self.test_symptoms,
            patient_info=self.test_patient
        )
        
        # Check that the error is properly formatted
        self.assertIn("error", result)
        self.assertIn("Failed to generate differential diagnosis", result["error"])
    
    def test_json_parsing_error(self):
        """Test handling of invalid JSON responses"""
        # Configure mock to return invalid JSON
        self.mock_llm.generate_text.return_value = "This is not JSON"
        
        # Call the method
        result = self.cds.get_differential_diagnosis(
            symptoms=self.test_symptoms,
            patient_info=self.test_patient
        )
        
        # Check that the error is properly handled
        self.assertIn("error", result)
        self.assertEqual("Failed to parse response", result["error"])
        self.assertIn("raw_response", result)
        self.assertEqual("This is not JSON", result["raw_response"])
    
    def test_assess_risk(self):
        """Test the risk assessment functionality"""
        # Configure mock to return risk assessment
        self.mock_llm.generate_text.return_value = json.dumps({
            "risk_level": "moderate",
            "risk_factors": [
                {"factor": "Age over 40", "significance": 0.4},
                {"factor": "Persistent cough", "significance": 0.6},
                {"factor": "Fever", "significance": 0.5}
            ],
            "potential_complications": [
                "Pneumonia",
                "Dehydration"
            ],
            "recommendations": [
                "Monitor temperature",
                "Increase fluid intake"
            ],
            "monitoring": [
                "Check temperature every 4-6 hours",
                "Watch for worsening cough or difficulty breathing"
            ]
        })
        
        # Test data
        vitals = {
            "temperature": "38.2°C",
            "heart_rate": "88 bpm",
            "blood_pressure": "125/82 mmHg",
            "respiratory_rate": "18/min",
            "oxygen_saturation": "97%"
        }
        
        # Call the method
        result = self.cds.assess_risk(
            patient_info=self.test_patient,
            vitals=vitals,
            condition="Respiratory Infection"
        )
        
        # Check the prompt
        args, _ = self.mock_llm.generate_text.call_args
        prompt = args[0]
        
        # Verify the prompt contains important elements
        self.assertIn("Respiratory Infection", prompt)
        self.assertIn("temperature: 38.2°C", prompt)
        
        # Check the result
        self.assertEqual(result["risk_level"], "moderate")
        self.assertEqual(len(result["risk_factors"]), 3)
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["condition"], "Respiratory Infection")
        self.assertTrue(result["metadata"]["has_vitals"])
    
    def test_generate_follow_up(self):
        """Test follow-up recommendations generation"""
        # Configure mock to return follow-up recommendations
        self.mock_llm.generate_text.return_value = json.dumps({
            "timeline": [
                {"time": "48 hours", "event": "Check if fever has resolved"},
                {"time": "1 week", "event": "Follow-up appointment if symptoms persist"},
                {"time": "2 weeks", "event": "Full recovery expected"}
            ],
            "monitoring_parameters": [
                "Body temperature",
                "Respiratory symptoms",
                "Energy levels"
            ],
            "warning_signs": [
                "Fever above 39°C",
                "Difficulty breathing",
                "Severe chest pain"
            ],
            "success_criteria": [
                "Resolution of fever within 3-5 days",
                "Gradual improvement in cough and congestion",
                "Return to normal activities within 1-2 weeks"
            ],
            "additional_testing": [
                "None required unless symptoms worsen"
            ],
            "self_management": [
                "Rest and adequate hydration",
                "Over-the-counter medications as prescribed",
                "Avoid strenuous activities until recovered"
            ]
        })
        
        # Test treatment plan
        treatment_plan = [
            "Acetaminophen 500mg every 6 hours for fever",
            "Increase fluid intake to at least 2L per day",
            "Rest for at least 3 days"
        ]
        
        # Call the method
        result = self.cds.generate_follow_up(
            diagnosis="Common Cold",
            treatment_plan=treatment_plan,
            patient_info=self.test_patient,
            time_frame="short-term"
        )
        
        # Check the prompt
        args, _ = self.mock_llm.generate_text.call_args
        prompt = args[0]
        
        # Verify the prompt contains important elements
        self.assertIn("Common Cold", prompt)
        self.assertIn("Acetaminophen 500mg", prompt)
        self.assertIn("short-term", prompt)
        
        # Check the result
        self.assertIn("timeline", result)
        self.assertEqual(len(result["timeline"]), 3)
        self.assertIn("monitoring_parameters", result)
        self.assertIn("warning_signs", result)
        
        # Check metadata
        metadata = result["metadata"]
        self.assertEqual(metadata["diagnosis"], "Common Cold")
        self.assertEqual(metadata["time_frame"], "short-term")
        self.assertEqual(metadata["treatment_count"], 3)


if __name__ == "__main__":
    unittest.main() 