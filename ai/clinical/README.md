# Clinical Decision Support Module

## Overview

The Clinical Decision Support module provides AI-powered clinical reasoning and decision support for healthcare professionals. This module leverages large language models (LLMs) and medical knowledge to assist with diagnosis, treatment planning, risk assessment, and follow-up recommendations.

## Features

- **Differential Diagnosis**: Generate ranked differential diagnoses based on patient symptoms, medical history, and test results
- **Treatment Recommendations**: Suggest evidence-based treatments for diagnosed conditions
- **Risk Assessment**: Evaluate patient risks based on medical information and provide risk mitigation strategies
- **Follow-up Planning**: Generate structured follow-up timelines and monitoring recommendations
- **Clinical Visualization**: Create visualizations of clinical data to aid in decision-making
- **Integration with Medical Imaging**: Combine image analysis findings with clinical reasoning

## Core Components

### ClinicalDecisionSupport

The main class that provides the core clinical decision support functionality:

```python
from ai.clinical.decision_support import ClinicalDecisionSupport
from ai.llm.model_connector import MedicalLLMConnector
from ai.knowledge.medical_rag import MedicalRAG

# Initialize LLM connector
llm_config = {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.1
}
llm = MedicalLLMConnector(llm_config)

# Initialize clinical decision support
cds = ClinicalDecisionSupport(
    llm_connector=llm,
    medical_rag=medical_rag,  # Optional
    confidence_threshold=0.7
)

# Generate differential diagnosis
diagnosis = cds.get_differential_diagnosis(
    symptoms=["fever", "cough", "fatigue"],
    patient_info={"age": 45, "sex": "male"}
)
```

### ClinicalVisualization

Utilities for visualizing clinical decision support results:

```python
from ai.clinical.visualization import ClinicalVisualization

# Initialize visualization
viz = ClinicalVisualization(output_dir="output/visualizations")

# Visualize differential diagnosis
viz.visualize_differential_diagnosis(
    diagnosis_result=diagnosis,
    title="Differential Diagnosis"
)
```

### ClinicalIntegration

Integration between clinical decision support and other MediNex AI components:

```python
from ai.clinical.integration import ClinicalIntegration

# Initialize clinical integration
integration = ClinicalIntegration(
    llm_config=llm_config,
    knowledge_base_path="data/knowledge",
    image_model_config=image_config
)

# Process a full clinical case
case_result = integration.process_full_clinical_case(
    patient_info=patient_info,
    symptoms=symptoms,
    medical_history=medical_history,
    image_paths=image_paths
)

# Generate a clinical report
report = integration.generate_clinical_report(
    case_result=case_result,
    report_type="comprehensive"
)
```

## Example Use Cases

### Generating Differential Diagnosis

```python
diagnosis = cds.get_differential_diagnosis(
    symptoms=["headache", "fever", "neck stiffness"],
    patient_info={
        "age": 35,
        "sex": "female",
        "weight": "70 kg",
        "height": "165 cm"
    },
    medical_history=["Migraine", "Hypertension"],
    test_results={"Blood pressure": "140/90 mmHg", "Temperature": "39.2°C"}
)
```

### Treatment Recommendations

```python
treatment = cds.get_treatment_recommendations(
    diagnosis="Bacterial Meningitis",
    patient_info=patient_info,
    medical_history=medical_history,
    current_medications=["Lisinopril 10mg daily"],
    allergies=["Penicillin"]
)
```

### Risk Assessment

```python
risk = cds.assess_risk(
    patient_info=patient_info,
    medical_history=medical_history,
    current_medications=current_medications,
    vitals={
        "temperature": "39.2°C",
        "heart_rate": "110 bpm",
        "blood_pressure": "140/90 mmHg",
        "respiratory_rate": "22/min",
        "oxygen_saturation": "96%"
    },
    lab_results={
        "WBC": "15.2 x 10^9/L",
        "CRP": "85 mg/L"
    },
    condition="Bacterial Meningitis"
)
```

### Follow-up Recommendations

```python
follow_up = cds.generate_follow_up(
    diagnosis="Bacterial Meningitis",
    treatment_plan=[
        "Ceftriaxone 2g IV every 12 hours",
        "Vancomycin 15 mg/kg IV every 8 hours",
        "Dexamethasone 10 mg IV every 6 hours"
    ],
    patient_info=patient_info,
    time_frame="short-term"
)
```

## Integration with Knowledge Base

The clinical decision support system can be enhanced with a medical knowledge base for retrieval-augmented generation (RAG):

```python
from ai.knowledge.medical_knowledge_base import MedicalKnowledgeBase
from ai.knowledge.medical_rag import MedicalRAG

# Initialize knowledge base
kb = MedicalKnowledgeBase(base_path="data/knowledge")

# Initialize RAG
rag = MedicalRAG(
    knowledge_base=kb,
    llm_config=llm_config
)

# Initialize clinical decision support with RAG
cds = ClinicalDecisionSupport(
    llm_connector=llm,
    medical_rag=rag
)
```

## Dependencies

- `ai.llm.model_connector`: LLM integration module
- `ai.knowledge.medical_rag`: Medical knowledge base and RAG module
- `matplotlib`: For visualization capabilities
- `numpy`: For data processing

## Unit Testing

Run the unit tests to ensure functionality:

```bash
pytest tests/clinical/test_decision_support.py -v
``` 