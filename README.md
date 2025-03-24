# MediNex AI - Advanced Medical Knowledge Assistant

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logos/medinex_logo.svg" width="300">
    <source media="(prefers-color-scheme: light)" srcset="assets/logos/medinex_logo.svg" width="300">
    <img alt="MediNex AI Logo" src="assets/logos/medinex_logo.svg" width="300">
  </picture>
  
  <!-- Note: If the SVG logo doesn't display properly, please view it directly at assets/logos/medinex_logo.svg -->
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.92.0-green)](https://fastapi.tiangolo.com/)
</div>

## [KEY] Overview

MediNex AI is an advanced medical knowledge assistant powered by large language models (LLMs) and retrieval-augmented generation (RAG). The system integrates expert medical knowledge with state-of-the-art AI models to provide accurate, contextualized medical information and reasoning.

### Key Features

- **Medical Knowledge Base**: Vector-based storage and retrieval of medical information
- **Retrieval-Augmented Generation**: Enhances LLM responses with relevant medical knowledge
- **Medical Image Analysis**: Vision-language capabilities for medical imaging interpretation
- **Multi-Provider Support**: Connects to various LLM providers (OpenAI, Anthropic, etc.)
- **Contributor Management**: System for tracking contributions and revenue sharing
- **Model Distribution**: Framework for versioning and distributing medical AI models

## [ARCHITECTURE] System Architecture

MediNex AI employs a modular architecture designed for performance, accuracy, and extensibility.

### High-Level Architecture

```
+-------------------------------------------------------------+
|                        Client Applications                   |
|                                                             |
|   +---------------+   +---------------+   +---------------+ |
|   |   Web Interface|   |   CLI Interface|   |   API Clients | |
|   |               |   |               |   |               | |
|   +---------------+   +---------------+   +---------------+ |
+------------------------------+--------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                           API Layer                          |
|                                                             |
|   +---------------+   +---------------+   +---------------+ |
|   |  Query Endpoints|   |  KB Management|   |  Image Analysis| |
|   |               |   |               |   |               | |
|   +---------------+   +---------------+   +---------------+ |
+------------------------------+--------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Core Processing Modules                  |
|                                                             |
| +-----------+ +-----------+ +-----------+ +-----------+    |
| | Medical RAG| | Knowledge | | LLM       | | Imaging   |    |
| | System    | | Base      | | Connector | | Pipeline  |    |
| +-----------+ +-----------+ +-----------+ +-----------+    |
|                                                             |
| +-----------+ +-----------+ +-----------+ +-----------+    |
| | Data      | | Contributor| | Revenue   | | Model     |    |
| | Importer  | | Management| | Sharing   | | Distribution|    |
| +-----------+ +-----------+ +-----------+ +-----------+    |
+------------------------------+--------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       External Integrations                  |
|                                                             |
|   +---------------+   +---------------+   +---------------+ |
|   |  LLM Providers|   |  Medical Data |   |  Vision Models| |
|   |  (OpenAI, etc.)|   |  Sources     |   |               | |
|   +---------------+   +---------------+   +---------------+ |
+------------------------------+--------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                         Data Layer                           |
|                                                             |
|   +---------------+   +---------------+   +---------------+ |
|   |   Knowledge   |   |   Models &    |   | Contributor & | |
|   |   Base Storage|   |   Cache       |   | Revenue Data  | |
|   +---------------+   +---------------+   +---------------+ |
+-------------------------------------------------------------+
```

### Data Flow

```
+-------------+     +-------------+     +-------------+
| User Query  |---->| Medical RAG |---->| Knowledge   |
+-------------+     | System      |<----| Base        |
                    +------+------+     | Retrieval   |
                           |            +-------------+
                           v
+-------------+     +-------------+     +-------------+
| Enhanced    |<----| LLM         |<----| Context     |
| Response    |     | Processing  |     | Formatting  |
+-------------+     | & Generation|     +-------------+
                    +-------------+
```

## [TECH] Technical Stack

MediNex AI is built with a modern technology stack:

### Core Components
- **Language**: Python 3.9+
- **API Framework**: FastAPI
- **Vector Storage**: FAISS (Facebook AI Similarity Search)
- **Data Processing**: NumPy, SciPy, Pandas
- **Image Processing**: OpenCV, PIL

### LLM Integration
- **Model Providers**: OpenAI, Anthropic, HuggingFace 
- **Local Models**: Support for locally hosted models
- **Embeddings**: State-of-the-art text embeddings for knowledge retrieval

### Medical Imaging
- **Image Analysis**: Computer vision for medical imaging
- **Modalities**: X-ray, MRI, CT, Ultrasound, Pathology, Dermatology
- **Vision-Language**: Integrated vision-language capabilities

## [GETTING-STARTED] Getting Started

### Prerequisites
- Python 3.9+
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/MediNexAI/MediNex.git
cd MediNex

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ai/requirements.txt

# Set up environment variables
cp .env.example .env.development
# Edit .env.development with your API keys and configuration

# Initialize the system
python app.py init
```

## [COMPONENTS] Core Components

### Medical Knowledge Base

The knowledge base stores and retrieves medical information with semantic search capabilities:

```python
# Initialize knowledge base
kb = MedicalKnowledgeBase(
    knowledge_dir="./data/knowledge",
    llm_config={"provider": "openai", "model": "text-embedding-3-small"}
)

# Add a document
doc_id = kb.add_document(
    text="Hypertension is a condition...",
    metadata=DocumentMetadata(
        source="Medical Encyclopedia",
        title="Hypertension",
        category="cardiovascular"
    )
)

# Search the knowledge base
results = kb.search("treatments for high blood pressure", limit=5)
```

### Retrieval-Augmented Generation (RAG)

The RAG system enhances LLM responses with medical knowledge:

```python
# Initialize the RAG system
rag = MedicalRAG(
    knowledge_base=kb,
    llm_config={"provider": "openai", "model": "gpt-4"}
)

# Query the system
response = rag.query(
    query="What are the latest treatments for hypertension?",
    category="cardiovascular",
    include_sources=True
)

# Access the enhanced response and sources
answer = response["answer"]
sources = response["sources"]
```

### Medical Imaging Pipeline

The imaging pipeline analyzes medical images with computer vision and LLM interpretation:

```python
# Initialize the imaging pipeline
imaging = MedicalImagingPipeline(
    llm_connector=llm_connector
)

# Analyze a medical image
result = imaging.analyze_image(
    image_path="path/to/xray.jpg",
    modality="xray",
    analysis_type="diagnostic",
    clinical_context="Patient has persistent cough"
)

# Access the analysis results
findings = result["findings"]
impression = result["impression"]
recommendations = result["recommendations"]
```

## [CLI] Command-Line Interface

MediNex AI provides a comprehensive CLI for system interaction:

```bash
# Initialize the system
python app.py init

# Start the API server
python app.py serve --port 8000

# Query the system
python app.py query "What are the symptoms of diabetes?"

# Import data into the knowledge base
python app.py import --directory ./data/sample/medical_papers

# Analyze a medical image
python app.py analyze-image ./data/sample/images/xray.jpg --modality xray

# Manage the knowledge base
python app.py list-documents --limit 10
python app.py delete-document <doc_id>
```

## [API] API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/llm/query` | POST | Generate a response to a medical query |
| `/knowledge/search` | POST | Search the medical knowledge base |
| `/knowledge/add` | POST | Add a document to the knowledge base |
| `/imaging/analyze` | POST | Analyze a medical image |
| `/contributors/register` | POST | Register a new contributor |
| `/models/versions` | GET | List available model versions |

## [CONFIG] Configuration

The system can be configured via environment variables or the `config.json` file:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.3
  },
  "knowledge_base": {
    "storage_path": "./data/knowledge",
    "chunk_size": 500,
    "chunk_overlap": 50
  },
  "imaging": {
    "models_dir": "./models/imaging",
    "cache_dir": "./cache/imaging"
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "info"
  }
}
```

## [SECURITY] Security Framework

MediNex AI implements several security measures:

1. **Data Protection**
   - Encryption for sensitive medical data
   - Access control mechanisms
   - Data minimization principles

2. **API Security**
   - Authentication and authorization
   - Rate limiting
   - Input validation

3. **LLM Safety**
   - Content moderation
   - Medical accuracy verification
   - Prompt engineering safeguards

4. **Privacy Compliance**
   - HIPAA compatibility considerations
   - Patient data anonymization
   - Audit trails for sensitive operations

## [STRUCTURE] Project Structure

```
MediNex AI
+-- ai/
|   +-- api/                  # API implementation
|   +-- contributors/         # Contributor management
|   +-- distribution/         # Model distribution
|   +-- evaluation/           # Evaluation utilities
|   +-- integrations/         # External integrations
|   |   +-- imaging_llm_pipeline.py # Medical imaging analysis
|   +-- knowledge/            # Knowledge base and RAG
|   |   +-- medical_knowledge_base.py # Vector knowledge base
|   |   +-- medical_rag.py    # Retrieval-augmented generation
|   +-- llm/                  # LLM connectivity
|   |   +-- model_connector.py # Model provider interface
|   +-- models/               # Model definitions
|   +-- serialization/        # Data serialization utilities
|   +-- training/             # Model training utilities
|   +-- requirements.txt      # Core dependencies
+-- app.py                    # Application entry point
+-- config.json               # System configuration
+-- data/                     # Data storage
|   +-- knowledge/            # Knowledge base storage
|   +-- contributors/         # Contributor data
|   +-- revenue/              # Revenue sharing data
|   +-- models/               # Model versioning and packaging
+-- models/                   # Model storage
|   +-- imaging/              # Medical imaging models
|   +-- local_model/          # Local LLM models
+-- cache/                    # Caching directory
```

## [LICENSE] License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## [CONTRIBUTING] Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## [CONTACT] Contact

- Website: [medinex.life](http://medinex.life/)
- Twitter: [MediNex_AI](https://x.com/MediNex_AI)
- GitHub: [MediNexAI/MediNex](https://github.com/MediNexAI/MediNex)
- Email: info@medinex.life

Built with love by the MediNex AI Team 