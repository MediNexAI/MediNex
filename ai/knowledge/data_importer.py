"""
Medical Data Importer

This module provides functionality to import medical data from various sources
into the MediNex AI knowledge base, supporting formats like CSV, JSON, PDF,
and plain text.
"""

import os
import csv
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import re

from .medical_rag import MedicalKnowledgeBase

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalDataImporter:
    """
    Imports medical data from various sources into the knowledge base.
    
    This class provides methods to import data from files (CSV, JSON, PDF,
    plain text) and directories, preprocessing them and adding them to the
    MediNex AI knowledge base.
    """
    
    def __init__(self, knowledge_base):
        """
        Initialize the medical data importer.
        
        Args:
            knowledge_base: The MedicalKnowledgeBase instance
        """
        self.knowledge_base = knowledge_base
        
        # Common metadata fields to extract from various sources
        self.common_metadata_fields = [
            "title", "source", "author", "date", "category", 
            "specialty", "keywords", "type", "url", "doi"
        ]
        
        logger.info("Initialized medical data importer")
    
    def import_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Import all supported files from a directory into the knowledge base.
        
        Args:
            directory_path: Path to the directory containing files
            recursive: Whether to search subdirectories recursively
            
        Returns:
            Summary of import operation
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Track import statistics
        stats = {
            "total_files": 0,
            "successful_imports": 0,
            "failed_imports": 0,
            "by_type": {
                "csv": 0,
                "json": 0,
                "txt": 0,
                "pdf": 0,
                "md": 0,
                "other": 0
            }
        }
        
        logger.info(f"Starting import from directory: {directory_path}")
        
        # Get all files
        if recursive:
            files = list(directory.glob("**/*"))
        else:
            files = list(directory.glob("*"))
        
        # Filter for regular files
        files = [f for f in files if f.is_file()]
        stats["total_files"] = len(files)
        
        # Process each file
        for file_path in files:
            try:
                suffix = file_path.suffix.lower()
                
                if suffix == ".csv":
                    self.import_csv(str(file_path))
                    stats["by_type"]["csv"] += 1
                    stats["successful_imports"] += 1
                elif suffix in [".json", ".jsonl"]:
                    self.import_json(str(file_path))
                    stats["by_type"]["json"] += 1
                    stats["successful_imports"] += 1
                elif suffix in [".txt", ".md", ".pdf"]:
                    self.import_text_file(str(file_path))
                    if suffix == ".txt":
                        stats["by_type"]["txt"] += 1
                    elif suffix == ".md":
                        stats["by_type"]["md"] += 1
                    elif suffix == ".pdf":
                        stats["by_type"]["pdf"] += 1
                    stats["successful_imports"] += 1
                else:
                    stats["by_type"]["other"] += 1
                    logger.warning(f"Skipping unsupported file type: {file_path}")
                    
            except Exception as e:
                stats["failed_imports"] += 1
                logger.error(f"Error importing file {file_path}: {str(e)}")
        
        logger.info(f"Directory import complete. Imported {stats['successful_imports']} of {stats['total_files']} files.")
        return stats
    
    def import_csv(self, file_path: str, content_column: Optional[str] = None,
                   delimiter: str = ",", encoding: str = "utf-8") -> int:
        """
        Import data from a CSV file into the knowledge base.
        
        Args:
            file_path: Path to the CSV file
            content_column: Column containing the main content to store
                           (if None, autodetect or use all columns)
            delimiter: CSV delimiter character
            encoding: File encoding
            
        Returns:
            Number of documents added
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Track the number of imported documents
        added_documents = 0
        
        logger.info(f"Importing CSV from {file_path}")
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                
                # Get the headers
                headers = reader.fieldnames
                if not headers:
                    raise ValueError("CSV file has no headers")
                
                # Auto-detect content column if not specified
                if content_column is None:
                    # Look for common content column names
                    content_column_candidates = ["content", "text", "body", "description", "information", "data"]
                    for candidate in content_column_candidates:
                        if candidate in headers:
                            content_column = candidate
                            break
                    
                    # If still not found, use the column with the longest average content
                    if content_column is None:
                        # Read a sample of rows to determine average content length
                        sample_rows = []
                        for i, row in enumerate(reader):
                            sample_rows.append(row)
                            if i >= 10:  # Sample size of 10 rows
                                break
                        
                        # Reset file pointer to start
                        f.seek(0)
                        next(reader)  # Skip header row
                        
                        # Calculate average content length per column
                        if sample_rows:
                            avg_lengths = {}
                            for header in headers:
                                avg_lengths[header] = sum(len(str(row.get(header, ""))) for row in sample_rows) / len(sample_rows)
                            
                            # Use column with longest average content
                            content_column = max(avg_lengths, key=avg_lengths.get)
                
                # Process each row
                for row in reader:
                    # Extract content
                    if content_column in row:
                        content = row[content_column]
                    else:
                        # If content column not found, use all columns
                        content = "\n".join([f"{header}: {value}" for header, value in row.items()])
                    
                    # Extract metadata
                    metadata = {}
                    for header in headers:
                        if header != content_column:
                            metadata[header] = row[header]
                    
                    # Add source and file information to metadata
                    metadata["source"] = f"CSV Import: {os.path.basename(file_path)}"
                    metadata["import_time"] = time.time()
                    
                    # Add document to knowledge base
                    doc_id = self.knowledge_base.add_document(content=content, metadata=metadata)
                    added_documents += 1
            
            logger.info(f"Successfully imported {added_documents} documents from CSV file")
            return added_documents
            
        except Exception as e:
            logger.error(f"Error importing CSV file: {str(e)}")
            raise
    
    def import_json(self, file_path: str, content_key: Optional[str] = None,
                   metadata_keys: Optional[List[str]] = None) -> int:
        """
        Import data from a JSON file into the knowledge base.
        
        Args:
            file_path: Path to the JSON file
            content_key: Key containing the main content to store
                        (if None, autodetect)
            metadata_keys: Keys to extract as metadata
                          (if None, extract all except content_key)
            
        Returns:
            Number of documents added
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Track the number of imported documents
        added_documents = 0
        
        logger.info(f"Importing JSON from {file_path}")
        
        try:
            # Check if it's a JSONL file
            is_jsonl = file_path.lower().endswith(".jsonl")
            
            if is_jsonl:
                # Process JSONL (JSON Lines) file - one JSON object per line
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            json_obj = json.loads(line)
                            # Process the JSON object
                            added = self._process_json_object(json_obj, content_key, metadata_keys, file_path)
                            added_documents += added
                        except Exception as e:
                            logger.error(f"Error processing JSON line {line_num + 1}: {str(e)}")
            else:
                # Regular JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    # List of objects
                    for item in data:
                        if isinstance(item, dict):
                            added = self._process_json_object(item, content_key, metadata_keys, file_path)
                            added_documents += added
                        else:
                            # Simple value in a list
                            doc_id = self.knowledge_base.add_document(
                                content=str(item),
                                metadata={
                                    "source": f"JSON Import: {os.path.basename(file_path)}",
                                    "import_time": time.time()
                                }
                            )
                            added_documents += 1
                elif isinstance(data, dict):
                    # Single object
                    added = self._process_json_object(data, content_key, metadata_keys, file_path)
                    added_documents += added
                else:
                    # Primitive value
                    doc_id = self.knowledge_base.add_document(
                        content=str(data),
                        metadata={
                            "source": f"JSON Import: {os.path.basename(file_path)}",
                            "import_time": time.time()
                        }
                    )
                    added_documents += 1
            
            logger.info(f"Successfully imported {added_documents} documents from JSON file")
            return added_documents
            
        except Exception as e:
            logger.error(f"Error importing JSON file: {str(e)}")
            raise
    
    def _process_json_object(self, obj: Dict[str, Any], content_key: Optional[str],
                            metadata_keys: Optional[List[str]], file_path: str) -> int:
        """
        Process a JSON object and add it to the knowledge base.
        
        Args:
            obj: JSON object (as dict)
            content_key: Key for content
            metadata_keys: Keys for metadata
            file_path: Source file path
            
        Returns:
            Number of documents added (0 or 1)
        """
        if not isinstance(obj, dict):
            return 0
            
        # Auto-detect content key if not specified
        if content_key is None:
            # Look for common content keys
            content_key_candidates = ["content", "text", "body", "description", "information", "data"]
            for candidate in content_key_candidates:
                if candidate in obj:
                    content_key = candidate
                    break
            
            # If still not found, use the longest string value
            if content_key is None:
                max_length = 0
                for key, value in obj.items():
                    if isinstance(value, str) and len(value) > max_length:
                        max_length = len(value)
                        content_key = key
        
        # Extract content
        if content_key in obj:
            content = obj[content_key]
        else:
            # If no content key found, concatenate all string values
            content_parts = []
            for key, value in obj.items():
                if isinstance(value, str):
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, (int, float, bool)):
                    content_parts.append(f"{key}: {value}")
            content = "\n".join(content_parts)
        
        # Convert content to string if it's not already
        if not isinstance(content, str):
            content = json.dumps(content)
        
        # Extract metadata
        metadata = {}
        if metadata_keys:
            # Extract specified keys
            for key in metadata_keys:
                if key in obj and key != content_key:
                    metadata[key] = obj[key]
        else:
            # Extract all keys except content key
            for key, value in obj.items():
                if key != content_key and (isinstance(value, (str, int, float, bool)) or value is None):
                    metadata[key] = value
        
        # Add source and file information to metadata
        metadata["source"] = f"JSON Import: {os.path.basename(file_path)}"
        metadata["import_time"] = time.time()
        
        # Add document to knowledge base
        doc_id = self.knowledge_base.add_document(content=content, metadata=metadata)
        return 1
    
    def import_text_file(self, file_path: str, encoding: str = "utf-8",
                        chunk_size: Optional[int] = None) -> int:
        """
        Import data from a text file (TXT, MD, PDF) into the knowledge base.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding
            chunk_size: Size of chunks to split the text into
                      (if None, use knowledge base default)
            
        Returns:
            Number of documents added
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"Importing text file from {file_path}")
        
        try:
            # Extract content based on file type
            if file_ext == ".pdf":
                content = self._extract_pdf_content(file_path)
            else:
                # Regular text file
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
            
            # Extract basic metadata
            metadata = {
                "source": f"File Import: {os.path.basename(file_path)}",
                "file_type": file_ext[1:],  # Remove leading dot
                "file_name": os.path.basename(file_path),
                "import_time": time.time()
            }
            
            # Extract additional metadata from file content
            extracted_metadata = self._extract_metadata_from_text(content)
            metadata.update(extracted_metadata)
            
            # Add document to knowledge base
            doc_id = self.knowledge_base.add_document(
                content=content,
                metadata=metadata,
                chunk_size=chunk_size
            )
            
            logger.info(f"Successfully imported document from text file")
            return 1
            
        except Exception as e:
            logger.error(f"Error importing text file: {str(e)}")
            raise
    
    def _extract_pdf_content(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            # Try to use PyPDF2
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                return text
            except ImportError:
                logger.warning("PyPDF2 not installed. Trying pdfplumber...")
                
            # Try to use pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n\n"
                    return text
            except ImportError:
                logger.warning("pdfplumber not installed. Using basic text extraction...")
                
            # Fallback to basic text extraction
            with open(file_path, 'rb') as f:
                content = f.read().decode('latin-1', errors='ignore')
                # Remove non-printable characters
                content = ''.join(c if c.isprintable() or c in ['\n', '\t'] else ' ' for c in content)
                return content
                
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            raise
    
    def _extract_metadata_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract metadata from text content.
        
        Args:
            text: Text content
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        # Extract title (first non-empty line or # heading in markdown)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # Check for markdown title
                if line.startswith('# '):
                    metadata['title'] = line[2:].strip()
                    break
                # Otherwise use first non-empty line
                metadata['title'] = line
                break
        
        # Look for common metadata patterns in the text
        patterns = {
            'author': r'(?:author|by)[:\s]+([^\n]+)',
            'date': r'(?:date|published)[:\s]+([^\n]+)',
            'doi': r'(?:doi)[:\s]+([^\n]+)',
            'keywords': r'(?:keywords|tags)[:\s]+([^\n]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()
        
        return metadata


# Example usage
if __name__ == "__main__":
    from ai.knowledge.medical_rag import MedicalKnowledgeBase
    
    # Initialize knowledge base
    kb = MedicalKnowledgeBase(storage_path="./data/knowledge")
    
    # Initialize importer
    importer = MedicalDataImporter(kb)
    
    # Import example data
    importer.import_directory("./data/sample") 