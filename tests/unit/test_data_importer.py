"""
Unit tests for the Medical Data Importer module.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import csv
import json
import io

from ai.knowledge.data_importer import MedicalDataImporter


class TestMedicalDataImporter:
    """Test cases for the MedicalDataImporter class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock knowledge base
        self.mock_kb = MagicMock()
        self.mock_kb.add_document.return_value = "test-doc-id"
        
        # Create importer with mock knowledge base
        self.importer = MedicalDataImporter(knowledge_base=self.mock_kb)

    def test_initialization(self):
        """Test that the importer initializes correctly."""
        assert self.importer.knowledge_base == self.mock_kb
        assert hasattr(self.importer, 'supported_extensions')

    def test_extract_metadata_from_text(self):
        """Test metadata extraction from text content."""
        text = """Title: Diabetes Overview
        Author: Dr. Jane Smith
        Date: 2023-05-15
        
        Diabetes is a chronic condition that affects how the body processes blood sugar.
        """
        
        metadata = self.importer._extract_metadata_from_text(text)
        
        assert metadata["title"] == "Diabetes Overview"
        assert metadata["author"] == "Dr. Jane Smith"
        assert metadata["date"] == "2023-05-15"

    @patch('os.path.exists')
    @patch('os.path.isfile')
    @patch('ai.knowledge.data_importer.MedicalDataImporter.import_csv')
    @patch('ai.knowledge.data_importer.MedicalDataImporter.import_json')
    @patch('ai.knowledge.data_importer.MedicalDataImporter.import_text_file')
    def test_import_file(self, mock_import_text, mock_import_json, mock_import_csv, 
                         mock_isfile, mock_exists):
        """Test importing a single file based on extension."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        
        # Test CSV file import
        self.importer.import_file("test.csv")
        mock_import_csv.assert_called_once_with("test.csv")
        mock_import_csv.reset_mock()
        
        # Test JSON file import
        self.importer.import_file("test.json")
        mock_import_json.assert_called_once_with("test.json")
        mock_import_json.reset_mock()
        
        # Test text file import
        self.importer.import_file("test.txt")
        mock_import_text.assert_called_once_with("test.txt")
        mock_import_text.reset_mock()
        
        # Test PDF file import
        self.importer.import_file("test.pdf")
        mock_import_text.assert_called_once_with("test.pdf")
        mock_import_text.reset_mock()
        
        # Test unsupported file extension
        with pytest.raises(ValueError):
            self.importer.import_file("test.unknown")

    @patch('os.path.exists')
    @patch('os.walk')
    @patch('ai.knowledge.data_importer.MedicalDataImporter.import_file')
    def test_import_directory(self, mock_import_file, mock_walk, mock_exists):
        """Test importing files from a directory."""
        # Setup mocks
        mock_exists.return_value = True
        mock_walk.return_value = [
            ("/test/dir", [], ["test1.csv", "test2.json", "test3.txt", "test4.unknown"])
        ]
        
        # Define side effect for import_file to simulate success/failure
        def import_file_side_effect(filepath):
            if filepath.endswith('.unknown'):
                raise ValueError("Unsupported file type")
            return "test-doc-id"
        
        mock_import_file.side_effect = import_file_side_effect
        
        # Import directory
        results = self.importer.import_directory("/test/dir")
        
        # Verify expected calls and results
        assert mock_import_file.call_count == 4
        assert results["total_files"] == 4
        assert results["successful"] == 3
        assert results["failed"] == 1
        assert len(results["errors"]) == 1
        assert "Unsupported file type" in results["errors"][0]

    @patch('csv.reader')
    @patch('builtins.open', new_callable=mock_open)
    def test_import_csv(self, mock_file, mock_csv_reader):
        """Test importing data from a CSV file."""
        # Setup CSV mock data
        mock_csv_reader.return_value = [
            ["Title", "Content", "Author", "Source"],
            ["Diabetes Overview", "Diabetes is a chronic condition.", "Dr. Smith", "Medical Journal"],
            ["Heart Disease", "Heart disease is a leading cause of death.", "Dr. Jones", "Health Magazine"]
        ]
        
        # Call import_csv
        self.importer.import_csv("test.csv")
        
        # Verify knowledge base was called correctly
        assert self.mock_kb.add_document.call_count == 2
        
        # Check first document
        call_args1 = self.mock_kb.add_document.call_args_list[0][1]
        assert call_args1["content"] == "Diabetes is a chronic condition."
        assert call_args1["metadata"]["title"] == "Diabetes Overview"
        assert call_args1["metadata"]["author"] == "Dr. Smith"
        assert call_args1["metadata"]["source"] == "Medical Journal"
        
        # Check second document
        call_args2 = self.mock_kb.add_document.call_args_list[1][1]
        assert call_args2["content"] == "Heart disease is a leading cause of death."
        assert call_args2["metadata"]["title"] == "Heart Disease"
        assert call_args2["metadata"]["author"] == "Dr. Jones"
        assert call_args2["metadata"]["source"] == "Health Magazine"

    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_import_json_standard(self, mock_file, mock_json_load):
        """Test importing data from a standard JSON file."""
        # Setup JSON mock data (standard format)
        mock_json_load.return_value = [
            {
                "title": "Diabetes Overview",
                "content": "Diabetes is a chronic condition.",
                "metadata": {
                    "author": "Dr. Smith",
                    "source": "Medical Journal"
                }
            },
            {
                "title": "Heart Disease",
                "content": "Heart disease is a leading cause of death.",
                "metadata": {
                    "author": "Dr. Jones",
                    "source": "Health Magazine"
                }
            }
        ]
        
        # Call import_json
        self.importer.import_json("test.json")
        
        # Verify knowledge base was called correctly
        assert self.mock_kb.add_document.call_count == 2
        
        # Check first document
        call_args1 = self.mock_kb.add_document.call_args_list[0][1]
        assert call_args1["content"] == "Diabetes is a chronic condition."
        assert call_args1["metadata"]["title"] == "Diabetes Overview"
        assert call_args1["metadata"]["author"] == "Dr. Smith"
        assert call_args1["metadata"]["source"] == "Medical Journal"
        
        # Check second document
        call_args2 = self.mock_kb.add_document.call_args_list[1][1]
        assert call_args2["content"] == "Heart disease is a leading cause of death."
        assert call_args2["metadata"]["title"] == "Heart Disease"
        assert call_args2["metadata"]["author"] == "Dr. Jones"
        assert call_args2["metadata"]["source"] == "Health Magazine"

    @patch('builtins.open', new_callable=mock_open)
    @patch('ai.knowledge.data_importer.PyPDF2')
    def test_import_text_file_pdf(self, mock_pypdf2, mock_file):
        """Test importing data from a PDF file."""
        # Setup PDF mock data
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(), MagicMock()]
        mock_reader.pages[0].extract_text.return_value = "Title: Diabetes Overview\nAuthor: Dr. Smith\n\nPage 1 content."
        mock_reader.pages[1].extract_text.return_value = "Page 2 content."
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        # Call import_text_file
        self.importer.import_text_file("test.pdf")
        
        # Verify knowledge base was called correctly
        self.mock_kb.add_document.assert_called_once()
        call_args = self.mock_kb.add_document.call_args[1]
        assert call_args["content"] == "Title: Diabetes Overview\nAuthor: Dr. Smith\n\nPage 1 content.\nPage 2 content."
        assert call_args["metadata"]["title"] == "Diabetes Overview"
        assert call_args["metadata"]["author"] == "Dr. Smith"
        assert call_args["metadata"]["file_source"] == "test.pdf"

    @patch('builtins.open', new_callable=mock_open)
    def test_import_text_file_txt(self, mock_file):
        """Test importing data from a text file."""
        # Setup text file mock data
        mock_file.return_value.read.return_value = "Title: Diabetes Overview\nAuthor: Dr. Smith\n\nDiabetes is a chronic condition."
        
        # Call import_text_file
        self.importer.import_text_file("test.txt")
        
        # Verify knowledge base was called correctly
        self.mock_kb.add_document.assert_called_once()
        call_args = self.mock_kb.add_document.call_args[1]
        assert call_args["content"] == "Title: Diabetes Overview\nAuthor: Dr. Smith\n\nDiabetes is a chronic condition."
        assert call_args["metadata"]["title"] == "Diabetes Overview"
        assert call_args["metadata"]["author"] == "Dr. Smith"
        assert call_args["metadata"]["file_source"] == "test.txt" 