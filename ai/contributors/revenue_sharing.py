"""
MediNex AI Revenue Sharing System

This module provides functionality for calculating and distributing revenue
to contributors based on their contributions to the MediNex AI system.
"""

import os
import json
import logging
import time
import uuid
import csv
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .contributor_manager import ContributorManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RevenueCalculator:
    """
    Helper class to calculate revenue shares based on contribution metrics.
    """
    
    def __init__(self, contributor_manager: ContributorManager):
        """
        Initialize the revenue calculator.
        
        Args:
            contributor_manager: Instance of ContributorManager
        """
        self.contributor_manager = contributor_manager
        
        # Default weights for different contribution types
        self.weights = contributor_manager.contribution_types
    
    def calculate_contribution_value(
        self, 
        contribution: Dict[str, Any],
        base_value: float = 1.0
    ) -> float:
        """
        Calculate the value of a single contribution.
        
        Args:
            contribution: Contribution record
            base_value: Base value per usage
            
        Returns:
            Calculated value
        """
        if contribution["status"] != "approved":
            return 0.0
        
        contribution_type = contribution["type"]
        weight = self.weights.get(contribution_type, {}).get("weight", 0.0)
        usage_count = contribution.get("usage_count", 0)
        
        # Simple formula: base_value * weight * usage_count
        value = base_value * weight * usage_count
        
        # Apply quality multiplier based on review score if available
        review_score = contribution.get("metadata", {}).get("review", {}).get("score", 0.5)
        value *= (0.5 + review_score)  # Score range: 0.5 to 1.5
        
        return value
    
    def calculate_contributor_share(
        self,
        contributor_id: str,
        total_revenue: float,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate a contributor's share of the revenue.
        
        Args:
            contributor_id: Contributor's ID
            total_revenue: Total revenue to distribute
            period_start: Start date of the revenue period (ISO format)
            period_end: End date of the revenue period (ISO format)
            
        Returns:
            Dictionary with share calculation
        """
        # Get contributions for this contributor in the specified period
        contributions = self.contributor_manager.get_contributions(
            contributor_id=contributor_id,
            status="approved",
            start_date=period_start,
            end_date=period_end
        )
        
        if not contributions:
            return {
                "contributor_id": contributor_id,
                "revenue_share": 0.0,
                "total_value": 0.0,
                "contribution_count": 0
            }
        
        # Calculate total value of these contributions
        total_value = 0.0
        for contribution in contributions:
            value = self.calculate_contribution_value(contribution)
            total_value += value
        
        # For now, just return the value (actual share calculation will
        # be done at the RevenueShareSystem level with all contributors)
        return {
            "contributor_id": contributor_id,
            "total_value": total_value,
            "contribution_count": len(contributions),
            "contributions": [c["id"] for c in contributions]
        }
    
    def calculate_contribution_type_distribution(
        self,
        contributions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate the distribution of value by contribution type.
        
        Args:
            contributions: List of contribution records
            
        Returns:
            Dictionary mapping contribution types to their total value
        """
        distribution = {}
        
        for c_type in self.weights.keys():
            distribution[c_type] = 0.0
        
        for contribution in contributions:
            if contribution["status"] == "approved":
                c_type = contribution["type"]
                value = self.calculate_contribution_value(contribution)
                distribution[c_type] += value
        
        return distribution


class RevenueShareSystem:
    """
    Manages the calculation and distribution of revenue to contributors
    based on their contributions to the MediNex AI system.
    """
    
    def __init__(
        self,
        contributor_manager: ContributorManager,
        storage_path: str = "./data/revenue",
        platform_fee_percent: float = 15.0
    ):
        """
        Initialize the revenue share system.
        
        Args:
            contributor_manager: Instance of ContributorManager
            storage_path: Path to store revenue data
            platform_fee_percent: Platform fee percentage
        """
        self.contributor_manager = contributor_manager
        self.storage_path = storage_path
        self.platform_fee_percent = platform_fee_percent
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Path to revenue period and payment files
        self.periods_file = os.path.join(storage_path, "revenue_periods.json")
        self.payments_file = os.path.join(storage_path, "payments.json")
        
        # Load existing data or initialize empty data structures
        self.revenue_periods = self._load_revenue_periods()
        self.payments = self._load_payments()
        
        # Initialize revenue calculator
        self.calculator = RevenueCalculator(contributor_manager)
        
        logger.info("Initialized revenue share system")
    
    def _load_revenue_periods(self) -> Dict[str, Dict[str, Any]]:
        """
        Load revenue periods from storage.
        
        Returns:
            Dictionary of revenue periods
        """
        if os.path.exists(self.periods_file):
            try:
                with open(self.periods_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading revenue periods: {str(e)}")
                return {}
        else:
            return {}
    
    def _load_payments(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load payment records from storage.
        
        Returns:
            Dictionary of payments by contributor ID
        """
        if os.path.exists(self.payments_file):
            try:
                with open(self.payments_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading payments: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_revenue_periods(self) -> bool:
        """
        Save revenue periods to storage.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            with open(self.periods_file, 'w') as f:
                json.dump(self.revenue_periods, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving revenue periods: {str(e)}")
            return False
    
    def _save_payments(self) -> bool:
        """
        Save payment records to storage.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            with open(self.payments_file, 'w') as f:
                json.dump(self.payments, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving payments: {str(e)}")
            return False
    
    def create_revenue_period(
        self,
        start_date: str,
        end_date: str,
        total_revenue: float,
        currency: str = "USD",
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new revenue period for calculation.
        
        Args:
            start_date: Start date of the period (ISO format)
            end_date: End date of the period (ISO format)
            total_revenue: Total revenue to distribute
            currency: Currency code
            name: Name of the revenue period
            description: Description of the revenue period
            
        Returns:
            Revenue period record
        """
        # Generate a unique ID for the revenue period
        period_id = str(uuid.uuid4())
        
        # Create default name if not provided
        if name is None:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            name = f"Revenue Period {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
        
        # Create revenue period record
        period = {
            "id": period_id,
            "name": name,
            "description": description,
            "start_date": start_date,
            "end_date": end_date,
            "total_revenue": total_revenue,
            "currency": currency,
            "platform_fee_percent": self.platform_fee_percent,
            "platform_fee_amount": (total_revenue * self.platform_fee_percent / 100),
            "distributable_revenue": (total_revenue * (100 - self.platform_fee_percent) / 100),
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "calculated_at": None,
            "finalized_at": None,
            "shares": {}
        }
        
        # Add to revenue periods dictionary
        self.revenue_periods[period_id] = period
        
        # Save changes
        self._save_revenue_periods()
        
        logger.info(f"Created new revenue period: {name} (ID: {period_id})")
        
        return period
    
    def calculate_revenue_shares(self, period_id: str) -> Dict[str, Any]:
        """
        Calculate revenue shares for all contributors for a period.
        
        Args:
            period_id: Revenue period ID
            
        Returns:
            Updated revenue period record
        """
        if period_id not in self.revenue_periods:
            logger.error(f"Revenue period not found: {period_id}")
            return {}
        
        period = self.revenue_periods[period_id]
        
        if period["status"] not in ["created", "calculated"]:
            logger.error(f"Revenue period has invalid status for calculation: {period['status']}")
            return period
        
        # Get all active contributors
        contributors = self.contributor_manager.get_contributors(status="active")
        
        # Calculate share for each contributor
        total_value = 0.0
        contributor_values = {}
        
        for contributor in contributors:
            contributor_id = contributor["id"]
            share_data = self.calculator.calculate_contributor_share(
                contributor_id=contributor_id,
                total_revenue=period["distributable_revenue"],
                period_start=period["start_date"],
                period_end=period["end_date"]
            )
            
            if share_data["total_value"] > 0:
                contributor_values[contributor_id] = share_data
                total_value += share_data["total_value"]
        
        # Calculate actual revenue share based on relative value
        shares = {}
        if total_value > 0:
            for contributor_id, data in contributor_values.items():
                relative_value = data["total_value"] / total_value
                revenue_share = period["distributable_revenue"] * relative_value
                
                shares[contributor_id] = {
                    "contributor_id": contributor_id,
                    "contributor_name": self.contributor_manager.get_contributor(contributor_id)["name"],
                    "relative_value": relative_value,
                    "revenue_share": revenue_share,
                    "total_value": data["total_value"],
                    "contribution_count": data["contribution_count"],
                    "contributions": data["contributions"]
                }
        
        # Update the revenue period
        period["shares"] = shares
        period["total_value"] = total_value
        period["contributor_count"] = len(shares)
        period["status"] = "calculated"
        period["calculated_at"] = datetime.now().isoformat()
        
        # Save changes
        self._save_revenue_periods()
        
        logger.info(f"Calculated revenue shares for period: {period['name']} (ID: {period_id})")
        
        return period
    
    def finalize_revenue_period(self, period_id: str) -> Dict[str, Any]:
        """
        Finalize a revenue period, making it ready for payments.
        
        Args:
            period_id: Revenue period ID
            
        Returns:
            Updated revenue period record
        """
        if period_id not in self.revenue_periods:
            logger.error(f"Revenue period not found: {period_id}")
            return {}
        
        period = self.revenue_periods[period_id]
        
        if period["status"] != "calculated":
            logger.error(f"Revenue period must be calculated before finalizing: {period['status']}")
            return period
        
        # Update the status
        period["status"] = "finalized"
        period["finalized_at"] = datetime.now().isoformat()
        
        # Save changes
        self._save_revenue_periods()
        
        logger.info(f"Finalized revenue period: {period['name']} (ID: {period_id})")
        
        return period
    
    def get_revenue_period(self, period_id: str) -> Dict[str, Any]:
        """
        Get a revenue period by ID.
        
        Args:
            period_id: Revenue period ID
            
        Returns:
            Revenue period record
        """
        return self.revenue_periods.get(period_id, {})
    
    def get_revenue_periods(
        self,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get revenue periods with optional filtering.
        
        Args:
            status: Filter by status (None for all)
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            List of revenue period records
        """
        periods = list(self.revenue_periods.values())
        
        # Apply filters
        if status is not None:
            periods = [p for p in periods if p["status"] == status]
        
        if start_date is not None:
            start_datetime = datetime.fromisoformat(start_date)
            periods = [
                p for p in periods 
                if datetime.fromisoformat(p["start_date"]) >= start_datetime
            ]
        
        if end_date is not None:
            end_datetime = datetime.fromisoformat(end_date)
            periods = [
                p for p in periods 
                if datetime.fromisoformat(p["end_date"]) <= end_datetime
            ]
        
        # Sort by start date
        periods.sort(key=lambda p: p["start_date"], reverse=True)
        
        return periods
    
    def record_payment(
        self,
        period_id: str,
        contributor_id: str,
        amount: float,
        payment_method: str,
        transaction_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a payment to a contributor.
        
        Args:
            period_id: Revenue period ID
            contributor_id: Contributor's ID
            amount: Payment amount
            payment_method: Payment method used
            transaction_id: External transaction ID
            notes: Payment notes
            
        Returns:
            Payment record
        """
        if period_id not in self.revenue_periods:
            logger.error(f"Revenue period not found: {period_id}")
            return {}
        
        period = self.revenue_periods[period_id]
        
        if period["status"] != "finalized":
            logger.error(f"Revenue period must be finalized before recording payments: {period['status']}")
            return {}
        
        if contributor_id not in period["shares"]:
            logger.error(f"Contributor {contributor_id} not found in revenue period shares")
            return {}
        
        # Check if there's a share for this contributor
        share = period["shares"][contributor_id]
        
        # Generate a unique ID for the payment
        payment_id = str(uuid.uuid4())
        
        # Create payment record
        payment = {
            "id": payment_id,
            "contributor_id": contributor_id,
            "period_id": period_id,
            "amount": amount,
            "currency": period["currency"],
            "payment_method": payment_method,
            "transaction_id": transaction_id,
            "notes": notes,
            "status": "completed",
            "created_at": datetime.now().isoformat()
        }
        
        # Add to payments dictionary
        if contributor_id not in self.payments:
            self.payments[contributor_id] = []
        
        self.payments[contributor_id].append(payment)
        
        # Save changes
        self._save_payments()
        
        logger.info(f"Recorded payment to contributor {contributor_id}: {amount} {period['currency']}")
        
        return payment
    
    def get_contributor_payments(
        self,
        contributor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get payments for a contributor.
        
        Args:
            contributor_id: Contributor's ID
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            List of payment records
        """
        if contributor_id not in self.payments:
            return []
        
        payments = self.payments[contributor_id]
        
        # Apply date filters if provided
        if start_date is not None:
            start_datetime = datetime.fromisoformat(start_date)
            payments = [
                p for p in payments 
                if datetime.fromisoformat(p["created_at"]) >= start_datetime
            ]
        
        if end_date is not None:
            end_datetime = datetime.fromisoformat(end_date)
            payments = [
                p for p in payments 
                if datetime.fromisoformat(p["created_at"]) <= end_datetime
            ]
        
        # Sort by date
        payments.sort(key=lambda p: p["created_at"], reverse=True)
        
        return payments
    
    def generate_period_report(self, period_id: str, file_path: str) -> bool:
        """
        Generate a CSV report for a revenue period.
        
        Args:
            period_id: Revenue period ID
            file_path: Path to save the CSV report
            
        Returns:
            True if successful, False otherwise
        """
        if period_id not in self.revenue_periods:
            logger.error(f"Revenue period not found: {period_id}")
            return False
        
        period = self.revenue_periods[period_id]
        
        try:
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = [
                    'contributor_id', 'contributor_name', 'contribution_count',
                    'total_value', 'relative_value', 'revenue_share', 'currency'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for share in period["shares"].values():
                    writer.writerow({
                        'contributor_id': share['contributor_id'],
                        'contributor_name': share['contributor_name'],
                        'contribution_count': share['contribution_count'],
                        'total_value': share['total_value'],
                        'relative_value': share['relative_value'],
                        'revenue_share': share['revenue_share'],
                        'currency': period['currency']
                    })
            
            logger.info(f"Generated revenue period report: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating revenue period report: {str(e)}")
            return False
    
    def generate_contributor_report(
        self,
        contributor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        file_path: str
    ) -> bool:
        """
        Generate a CSV report for a contributor.
        
        Args:
            contributor_id: Contributor's ID
            start_date: Start date for the report (ISO format)
            end_date: End date for the report (ISO format)
            file_path: Path to save the CSV report
            
        Returns:
            True if successful, False otherwise
        """
        contributor = self.contributor_manager.get_contributor(contributor_id)
        if not contributor:
            logger.error(f"Contributor not found: {contributor_id}")
            return False
        
        try:
            # Get payments for the contributor
            payments = self.get_contributor_payments(
                contributor_id, start_date, end_date
            )
            
            # Get all revenue periods
            periods = self.get_revenue_periods()
            
            # Collect revenue shares for this contributor
            shares = []
            for period in periods:
                if contributor_id in period["shares"]:
                    if start_date and period["end_date"] < start_date:
                        continue
                    if end_date and period["start_date"] > end_date:
                        continue
                    
                    share = period["shares"][contributor_id]
                    share["period_name"] = period["name"]
                    share["start_date"] = period["start_date"]
                    share["end_date"] = period["end_date"]
                    share["currency"] = period["currency"]
                    shares.append(share)
            
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = [
                    'period_name', 'start_date', 'end_date', 'contribution_count',
                    'total_value', 'relative_value', 'revenue_share', 'currency'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for share in shares:
                    writer.writerow({
                        'period_name': share['period_name'],
                        'start_date': share['start_date'],
                        'end_date': share['end_date'],
                        'contribution_count': share['contribution_count'],
                        'total_value': share['total_value'],
                        'relative_value': share['relative_value'],
                        'revenue_share': share['revenue_share'],
                        'currency': share['currency']
                    })
            
            logger.info(f"Generated contributor report: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating contributor report: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize contributor manager
    contrib_manager = ContributorManager()
    
    # Initialize revenue share system
    revenue_system = RevenueShareSystem(contrib_manager)
    
    # Create a revenue period
    today = datetime.now()
    month_ago = today - timedelta(days=30)
    
    period = revenue_system.create_revenue_period(
        start_date=month_ago.isoformat(),
        end_date=today.isoformat(),
        total_revenue=10000.00,
        currency="USD",
        name="April 2024 Revenue",
        description="Monthly revenue for April 2024"
    )
    
    # Calculate shares
    revenue_system.calculate_revenue_shares(period["id"])
    
    # Finalize the period
    revenue_system.finalize_revenue_period(period["id"])
    
    # Record payments
    for contributor_id, share in period["shares"].items():
        revenue_system.record_payment(
            period_id=period["id"],
            contributor_id=contributor_id,
            amount=share["revenue_share"],
            payment_method="bank_transfer",
            notes="April 2024 revenue share"
        )
    
    # Generate reports
    revenue_system.generate_period_report(period["id"], "april_2024_revenue_report.csv") 