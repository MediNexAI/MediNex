"""
MediNex AI Contributor Manager

This module provides functionality for managing contributors to the MediNex AI system,
including registration, contribution tracking, and permissions management.
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContributorManager:
    """
    Manages contributors to the MediNex AI system.
    
    This class provides methods to register contributors, track their contributions,
    manage permissions, and calculate contribution metrics for revenue sharing.
    """
    
    def __init__(self, storage_path: str = "./data/contributors"):
        """
        Initialize the contributor manager.
        
        Args:
            storage_path: Path to store contributor data
        """
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Path to contributor and contribution files
        self.contributors_file = os.path.join(storage_path, "contributors.json")
        self.contributions_file = os.path.join(storage_path, "contributions.json")
        
        # Load existing data or initialize empty data structures
        self.contributors = self._load_contributors()
        self.contributions = self._load_contributions()
        
        # Contribution types and their weights for revenue calculation
        self.contribution_types = {
            "medical_data": {"description": "Medical data contribution", "weight": 0.5},
            "model_improvement": {"description": "Model improvement or fine-tuning", "weight": 0.3},
            "code_contribution": {"description": "Code or algorithm contribution", "weight": 0.15},
            "medical_review": {"description": "Medical content review or validation", "weight": 0.05}
        }
        
        logger.info("Initialized contributor manager")
    
    def _load_contributors(self) -> Dict[str, Dict[str, Any]]:
        """
        Load contributors from storage.
        
        Returns:
            Dictionary of contributors
        """
        if os.path.exists(self.contributors_file):
            try:
                with open(self.contributors_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading contributors: {str(e)}")
                return {}
        else:
            return {}
    
    def _load_contributions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load contribution records from storage.
        
        Returns:
            Dictionary of contributions by contributor ID
        """
        if os.path.exists(self.contributions_file):
            try:
                with open(self.contributions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading contributions: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_contributors(self) -> bool:
        """
        Save contributors to storage.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            with open(self.contributors_file, 'w') as f:
                json.dump(self.contributors, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving contributors: {str(e)}")
            return False
    
    def _save_contributions(self) -> bool:
        """
        Save contribution records to storage.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            with open(self.contributions_file, 'w') as f:
                json.dump(self.contributions, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving contributions: {str(e)}")
            return False
    
    def register_contributor(
        self,
        name: str,
        email: str,
        organization: Optional[str] = None,
        role: Optional[str] = None,
        specialties: Optional[List[str]] = None,
        payment_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a new contributor.
        
        Args:
            name: Contributor's name
            email: Contributor's email
            organization: Contributor's organization
            role: Contributor's role (e.g., "Doctor", "Researcher")
            specialties: List of medical specialties
            payment_info: Payment information for revenue sharing
            
        Returns:
            Contributor record with ID
        """
        # Check if contributor already exists with this email
        for contrib_id, contrib in self.contributors.items():
            if contrib["email"] == email:
                logger.warning(f"Contributor with email {email} already exists")
                return contrib
        
        # Generate a unique ID for the contributor
        contributor_id = str(uuid.uuid4())
        
        # Create contributor record
        contributor = {
            "id": contributor_id,
            "name": name,
            "email": email,
            "organization": organization,
            "role": role,
            "specialties": specialties or [],
            "payment_info": payment_info or {},
            "status": "active",
            "reputation_score": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Add to contributors dictionary
        self.contributors[contributor_id] = contributor
        
        # Initialize contributions list for this contributor
        self.contributions[contributor_id] = []
        
        # Save changes
        self._save_contributors()
        self._save_contributions()
        
        logger.info(f"Registered new contributor: {name} (ID: {contributor_id})")
        
        return contributor
    
    def update_contributor(
        self,
        contributor_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        organization: Optional[str] = None,
        role: Optional[str] = None,
        specialties: Optional[List[str]] = None,
        payment_info: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update a contributor's information.
        
        Args:
            contributor_id: Contributor's ID
            name: New name (if updating)
            email: New email (if updating)
            organization: New organization (if updating)
            role: New role (if updating)
            specialties: New specialties (if updating)
            payment_info: New payment information (if updating)
            status: New status (if updating)
            
        Returns:
            Updated contributor record or None if not found
        """
        if contributor_id not in self.contributors:
            logger.error(f"Contributor not found: {contributor_id}")
            return None
        
        contributor = self.contributors[contributor_id]
        
        # Update fields if provided
        if name is not None:
            contributor["name"] = name
        if email is not None:
            contributor["email"] = email
        if organization is not None:
            contributor["organization"] = organization
        if role is not None:
            contributor["role"] = role
        if specialties is not None:
            contributor["specialties"] = specialties
        if payment_info is not None:
            contributor["payment_info"] = payment_info
        if status is not None:
            contributor["status"] = status
        
        # Update timestamp
        contributor["updated_at"] = datetime.now().isoformat()
        
        # Save changes
        self._save_contributors()
        
        logger.info(f"Updated contributor: {contributor['name']} (ID: {contributor_id})")
        
        return contributor
    
    def get_contributor(self, contributor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a contributor by ID.
        
        Args:
            contributor_id: Contributor's ID
            
        Returns:
            Contributor record or None if not found
        """
        return self.contributors.get(contributor_id)
    
    def get_contributors(
        self,
        status: Optional[str] = "active",
        specialty: Optional[str] = None,
        role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get contributors with optional filtering.
        
        Args:
            status: Filter by status (None for all)
            specialty: Filter by specialty (None for all)
            role: Filter by role (None for all)
            
        Returns:
            List of contributor records
        """
        contributors = list(self.contributors.values())
        
        # Apply filters
        if status is not None:
            contributors = [c for c in contributors if c["status"] == status]
        
        if specialty is not None:
            contributors = [c for c in contributors if specialty in c["specialties"]]
        
        if role is not None:
            contributors = [c for c in contributors if c["role"] == role]
        
        return contributors
    
    def record_contribution(
        self,
        contributor_id: str,
        contribution_type: str,
        description: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Record a contribution from a contributor.
        
        Args:
            contributor_id: Contributor's ID
            contribution_type: Type of contribution (must match defined types)
            description: Description of the contribution
            data: Contribution data (content, references, etc.)
            metadata: Additional metadata about the contribution
            
        Returns:
            Contribution record or None if failed
        """
        if contributor_id not in self.contributors:
            logger.error(f"Contributor not found: {contributor_id}")
            return None
        
        if contribution_type not in self.contribution_types:
            logger.error(f"Invalid contribution type: {contribution_type}")
            return None
        
        if self.contributors[contributor_id]["status"] != "active":
            logger.error(f"Contributor is not active: {contributor_id}")
            return None
        
        # Generate a unique ID for the contribution
        contribution_id = str(uuid.uuid4())
        
        # Create contribution record
        contribution = {
            "id": contribution_id,
            "contributor_id": contributor_id,
            "type": contribution_type,
            "description": description,
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "review_status": "pending",
            "usage_count": 0,
            "value_score": 0
        }
        
        # Add to contributions dictionary
        if contributor_id not in self.contributions:
            self.contributions[contributor_id] = []
        
        self.contributions[contributor_id].append(contribution)
        
        # Save changes
        self._save_contributions()
        
        logger.info(f"Recorded contribution from {contributor_id}: {description} (ID: {contribution_id})")
        
        return contribution
    
    def update_contribution_status(
        self,
        contributor_id: str,
        contribution_id: str,
        status: str,
        review_notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update the status of a contribution.
        
        Args:
            contributor_id: Contributor's ID
            contribution_id: Contribution's ID
            status: New status ("approved", "rejected", "pending")
            review_notes: Notes from the reviewer
            
        Returns:
            Updated contribution record or None if not found
        """
        if contributor_id not in self.contributions:
            logger.error(f"Contributor not found: {contributor_id}")
            return None
        
        # Find the contribution by ID
        found = False
        for contribution in self.contributions[contributor_id]:
            if contribution["id"] == contribution_id:
                contribution["status"] = status
                contribution["review_status"] = status
                
                if review_notes:
                    if "review" not in contribution["metadata"]:
                        contribution["metadata"]["review"] = {}
                    
                    contribution["metadata"]["review"]["notes"] = review_notes
                    contribution["metadata"]["review"]["timestamp"] = datetime.now().isoformat()
                
                found = True
                break
        
        if not found:
            logger.error(f"Contribution not found: {contribution_id}")
            return None
        
        # Save changes
        self._save_contributions()
        
        logger.info(f"Updated contribution status: {contribution_id} to {status}")
        
        return contribution
    
    def get_contributions(
        self,
        contributor_id: Optional[str] = None,
        contribution_type: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get contributions with optional filtering.
        
        Args:
            contributor_id: Filter by contributor ID (None for all)
            contribution_type: Filter by contribution type (None for all)
            status: Filter by status (None for all)
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            List of contribution records
        """
        all_contributions = []
        
        # Collect all contributions or just for a specific contributor
        if contributor_id is not None:
            if contributor_id in self.contributions:
                all_contributions = self.contributions[contributor_id]
        else:
            for contrib_list in self.contributions.values():
                all_contributions.extend(contrib_list)
        
        # Apply filters
        filtered_contributions = all_contributions
        
        if contribution_type is not None:
            filtered_contributions = [c for c in filtered_contributions if c["type"] == contribution_type]
        
        if status is not None:
            filtered_contributions = [c for c in filtered_contributions if c["status"] == status]
        
        if start_date is not None:
            start_datetime = datetime.fromisoformat(start_date)
            filtered_contributions = [
                c for c in filtered_contributions 
                if datetime.fromisoformat(c["timestamp"]) >= start_datetime
            ]
        
        if end_date is not None:
            end_datetime = datetime.fromisoformat(end_date)
            filtered_contributions = [
                c for c in filtered_contributions 
                if datetime.fromisoformat(c["timestamp"]) <= end_datetime
            ]
        
        return filtered_contributions
    
    def calculate_contributor_metrics(self, contributor_id: str) -> Dict[str, Any]:
        """
        Calculate metrics for a contributor.
        
        Args:
            contributor_id: Contributor's ID
            
        Returns:
            Dictionary with contributor metrics
        """
        if contributor_id not in self.contributors:
            logger.error(f"Contributor not found: {contributor_id}")
            return {}
        
        if contributor_id not in self.contributions:
            return {
                "contributor_id": contributor_id,
                "total_contributions": 0,
                "approved_contributions": 0,
                "contribution_by_type": {},
                "total_usage": 0,
                "total_value": 0
            }
        
        # Get all contributions for this contributor
        contributions = self.contributions[contributor_id]
        
        # Calculate metrics
        total_contributions = len(contributions)
        approved_contributions = sum(1 for c in contributions if c["status"] == "approved")
        
        # Count by type
        contribution_by_type = {}
        for c in contributions:
            c_type = c["type"]
            if c_type not in contribution_by_type:
                contribution_by_type[c_type] = 0
            contribution_by_type[c_type] += 1
        
        # Calculate usage and value
        total_usage = sum(c.get("usage_count", 0) for c in contributions)
        total_value = sum(c.get("value_score", 0) for c in contributions)
        
        return {
            "contributor_id": contributor_id,
            "total_contributions": total_contributions,
            "approved_contributions": approved_contributions,
            "contribution_by_type": contribution_by_type,
            "total_usage": total_usage,
            "total_value": total_value
        }
    
    def update_usage_statistics(
        self, 
        contribution_ids: List[str], 
        usage_type: str
    ) -> None:
        """
        Update usage statistics for contributions.
        
        Args:
            contribution_ids: List of contribution IDs that were used
            usage_type: Type of usage (e.g., "query", "training")
        """
        for contributor_id, contributions in self.contributions.items():
            for contribution in contributions:
                if contribution["id"] in contribution_ids:
                    # Increment usage count
                    contribution["usage_count"] = contribution.get("usage_count", 0) + 1
                    
                    # Add usage record
                    if "usage_history" not in contribution:
                        contribution["usage_history"] = []
                    
                    contribution["usage_history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "type": usage_type
                    })
        
        # Save changes
        self._save_contributions()


# Example usage
if __name__ == "__main__":
    # Initialize contributor manager
    manager = ContributorManager()
    
    # Register a contributor
    contributor = manager.register_contributor(
        name="Dr. Jane Smith",
        email="jane.smith@example.com",
        organization="General Hospital",
        role="Doctor",
        specialties=["Cardiology", "Internal Medicine"],
        payment_info={"type": "bank_transfer", "account": "XXX-YYY-ZZZ"}
    )
    
    # Record a contribution
    contribution = manager.record_contribution(
        contributor_id=contributor["id"],
        contribution_type="medical_data",
        description="Cardiology diagnostic guidelines",
        data={
            "content": "Guidelines for diagnosing cardiovascular conditions...",
            "source": "Hospital Internal Guidelines",
            "version": "2.0"
        },
        metadata={
            "format": "text",
            "language": "en",
            "keywords": ["cardiology", "diagnosis", "guidelines"]
        }
    )
    
    # Approve the contribution
    manager.update_contribution_status(
        contributor_id=contributor["id"],
        contribution_id=contribution["id"],
        status="approved",
        review_notes="Excellent contribution. Very comprehensive guidelines."
    ) 