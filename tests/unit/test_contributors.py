"""
Unit tests for the Contributors module.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
import tempfile
from datetime import datetime

from ai.contributors.contributor_manager import ContributorManager
from ai.contributors.revenue_sharing import RevenueCalculator, RevenueShareSystem


class TestContributorManager:
    """Test cases for the ContributorManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temp file for contributor data
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        
        # Initialize test contributors data
        self.test_contributors = {
            "user1": {
                "name": "Test User 1",
                "email": "user1@example.com",
                "join_date": "2023-01-01",
                "permissions": ["read", "write"],
                "contributions": {
                    "data": 100,
                    "code": 5,
                    "review": 10
                }
            },
            "user2": {
                "name": "Test User 2",
                "email": "user2@example.com",
                "join_date": "2023-02-01",
                "permissions": ["read"],
                "contributions": {
                    "data": 50,
                    "code": 0,
                    "review": 5
                }
            }
        }
        
        # Write test data to file
        with open(self.temp_file.name, 'w') as f:
            json.dump(self.test_contributors, f)
        
        # Create contributor manager with test file
        self.manager = ContributorManager(data_file=self.temp_file.name)

    def teardown_method(self):
        """Tear down test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test that the contributor manager initializes correctly."""
        assert self.manager.data_file == self.temp_file.name
        assert isinstance(self.manager.contributors, dict)
        assert len(self.manager.contributors) == 2
        assert "user1" in self.manager.contributors
        assert "user2" in self.manager.contributors

    def test_register_contributor(self):
        """Test registering a new contributor."""
        # Register a new contributor
        user_id = self.manager.register_contributor(
            name="Test User 3",
            email="user3@example.com",
            permissions=["read"]
        )
        
        # Verify contributor was added
        assert user_id in self.manager.contributors
        assert self.manager.contributors[user_id]["name"] == "Test User 3"
        assert self.manager.contributors[user_id]["email"] == "user3@example.com"
        assert set(self.manager.contributors[user_id]["permissions"]) == {"read"}
        assert "join_date" in self.manager.contributors[user_id]
        assert "contributions" in self.manager.contributors[user_id]
        
        # Verify data was saved
        with open(self.temp_file.name, 'r') as f:
            saved_data = json.load(f)
            assert user_id in saved_data

    def test_get_contributor(self):
        """Test getting a contributor by ID."""
        # Get existing contributor
        contributor = self.manager.get_contributor("user1")
        
        # Verify contributor data
        assert contributor is not None
        assert contributor["name"] == "Test User 1"
        assert contributor["email"] == "user1@example.com"
        assert contributor["join_date"] == "2023-01-01"
        
        # Get non-existent contributor
        with pytest.raises(ValueError):
            self.manager.get_contributor("non_existent")

    def test_update_contributor(self):
        """Test updating a contributor's information."""
        # Update contributor
        self.manager.update_contributor(
            user_id="user1",
            name="Updated User 1",
            email="updated1@example.com",
            permissions=["read", "write", "admin"]
        )
        
        # Verify contributor was updated
        contributor = self.manager.get_contributor("user1")
        assert contributor["name"] == "Updated User 1"
        assert contributor["email"] == "updated1@example.com"
        assert set(contributor["permissions"]) == {"read", "write", "admin"}
        
        # Verify data was saved
        with open(self.temp_file.name, 'r') as f:
            saved_data = json.load(f)
            assert saved_data["user1"]["name"] == "Updated User 1"

    def test_record_contribution(self):
        """Test recording a contribution."""
        # Record a contribution
        self.manager.record_contribution(
            user_id="user1",
            contribution_type="data",
            amount=50,
            details="Added 50 medical documents"
        )
        
        # Verify contribution was recorded
        contributor = self.manager.get_contributor("user1")
        assert contributor["contributions"]["data"] == 150  # 100 + 50
        
        # Record a new type of contribution
        self.manager.record_contribution(
            user_id="user1",
            contribution_type="model",
            amount=1,
            details="Contributed custom model"
        )
        
        # Verify new contribution type was recorded
        contributor = self.manager.get_contributor("user1")
        assert contributor["contributions"]["model"] == 1
        
        # Verify data was saved
        with open(self.temp_file.name, 'r') as f:
            saved_data = json.load(f)
            assert saved_data["user1"]["contributions"]["data"] == 150
            assert saved_data["user1"]["contributions"]["model"] == 1

    def test_get_contributors(self):
        """Test getting all contributors."""
        # Get all contributors
        contributors = self.manager.get_contributors()
        
        # Verify all contributors are returned
        assert len(contributors) == 2
        assert "user1" in contributors
        assert "user2" in contributors

    def test_remove_contributor(self):
        """Test removing a contributor."""
        # Remove a contributor
        self.manager.remove_contributor("user2")
        
        # Verify contributor was removed
        assert "user2" not in self.manager.contributors
        
        # Verify data was saved
        with open(self.temp_file.name, 'r') as f:
            saved_data = json.load(f)
            assert "user2" not in saved_data
        
        # Try to remove non-existent contributor
        with pytest.raises(ValueError):
            self.manager.remove_contributor("non_existent")

    def test_get_contributor_stats(self):
        """Test getting contributor statistics."""
        # Get contributor stats
        stats = self.manager.get_contributor_stats("user1")
        
        # Verify stats
        assert stats["total_contributions"] == 115  # 100 + 5 + 10
        assert stats["data_percentage"] == (100 / 115) * 100
        assert stats["code_percentage"] == (5 / 115) * 100
        assert stats["review_percentage"] == (10 / 115) * 100


class TestRevenueCalculator:
    """Test cases for the RevenueCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test contributor data
        self.test_contributors = {
            "user1": {
                "contributions": {
                    "data": 100,
                    "code": 50,
                    "review": 20
                }
            },
            "user2": {
                "contributions": {
                    "data": 50,
                    "code": 10,
                    "review": 30
                }
            },
            "user3": {
                "contributions": {
                    "data": 25,
                    "code": 5,
                    "review": 10
                }
            }
        }
        
        # Create calculator with test weights
        self.calculator = RevenueCalculator(
            weights={
                "data": 1.0,
                "code": 2.0,
                "review": 0.5
            }
        )

    def test_calculate_weighted_contributions(self):
        """Test calculating weighted contributions."""
        # Calculate weighted contributions
        weighted = self.calculator.calculate_weighted_contributions(self.test_contributors)
        
        # Verify weighted calculations
        assert weighted["user1"] == 100*1.0 + 50*2.0 + 20*0.5 == 210
        assert weighted["user2"] == 50*1.0 + 10*2.0 + 30*0.5 == 85
        assert weighted["user3"] == 25*1.0 + 5*2.0 + 10*0.5 == 40
        
        # Test with different weights
        calculator = RevenueCalculator(
            weights={
                "data": 2.0,
                "code": 1.0,
                "review": 1.0
            }
        )
        weighted = calculator.calculate_weighted_contributions(self.test_contributors)
        assert weighted["user1"] == 100*2.0 + 50*1.0 + 20*1.0 == 270

    def test_calculate_share_percentages(self):
        """Test calculating share percentages."""
        # Calculate share percentages
        percentages = self.calculator.calculate_share_percentages(self.test_contributors)
        
        total_weighted = 210 + 85 + 40  # From previous test
        
        # Verify percentages
        assert percentages["user1"] == (210 / total_weighted) * 100
        assert percentages["user2"] == (85 / total_weighted) * 100
        assert percentages["user3"] == (40 / total_weighted) * 100
        assert sum(percentages.values()) == pytest.approx(100.0)

    def test_calculate_revenue_shares(self):
        """Test calculating revenue shares from a total amount."""
        # Calculate revenue shares
        total_revenue = 10000.0
        shares = self.calculator.calculate_revenue_shares(self.test_contributors, total_revenue)
        
        total_weighted = 210 + 85 + 40  # From previous test
        
        # Verify shares
        assert shares["user1"] == (210 / total_weighted) * total_revenue
        assert shares["user2"] == (85 / total_weighted) * total_revenue
        assert shares["user3"] == (40 / total_weighted) * total_revenue
        assert sum(shares.values()) == pytest.approx(total_revenue)


class TestRevenueShareSystem:
    """Test cases for the RevenueShareSystem class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temp file for revenue data
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        
        # Create mock contributor manager
        self.mock_manager = MagicMock()
        self.mock_manager.get_contributors.return_value = {
            "user1": {
                "name": "Test User 1",
                "contributions": {
                    "data": 100,
                    "code": 50,
                    "review": 20
                }
            },
            "user2": {
                "name": "Test User 2",
                "contributions": {
                    "data": 50,
                    "code": 10,
                    "review": 30
                }
            }
        }
        
        # Create revenue system
        self.revenue_system = RevenueShareSystem(
            contributor_manager=self.mock_manager,
            data_file=self.temp_file.name,
            calculator=RevenueCalculator()
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test that the revenue system initializes correctly."""
        assert self.revenue_system.data_file == self.temp_file.name
        assert self.revenue_system.contributor_manager == self.mock_manager
        assert isinstance(self.revenue_system.calculator, RevenueCalculator)
        assert isinstance(self.revenue_system.periods, dict)

    def test_create_revenue_period(self):
        """Test creating a new revenue period."""
        # Create a revenue period
        period_id = self.revenue_system.create_revenue_period(
            name="Q1 2023",
            start_date="2023-01-01",
            end_date="2023-03-31",
            total_revenue=50000.0
        )
        
        # Verify period was created
        assert period_id in self.revenue_system.periods
        period = self.revenue_system.periods[period_id]
        assert period["name"] == "Q1 2023"
        assert period["start_date"] == "2023-01-01"
        assert period["end_date"] == "2023-03-31"
        assert period["total_revenue"] == 50000.0
        assert period["status"] == "open"
        assert "shares" not in period

    def test_calculate_period_shares(self):
        """Test calculating shares for a period."""
        # Create a period first
        period_id = self.revenue_system.create_revenue_period(
            name="Q1 2023",
            start_date="2023-01-01",
            end_date="2023-03-31",
            total_revenue=10000.0
        )
        
        # Calculate shares
        self.revenue_system.calculate_period_shares(period_id)
        
        # Verify shares were calculated
        period = self.revenue_system.periods[period_id]
        assert "shares" in period
        assert "user1" in period["shares"]
        assert "user2" in period["shares"]
        assert sum(period["shares"].values()) == pytest.approx(10000.0)
        
        # Test with non-existent period
        with pytest.raises(ValueError):
            self.revenue_system.calculate_period_shares("non_existent")

    def test_finalize_period(self):
        """Test finalizing a revenue period."""
        # Create and calculate a period first
        period_id = self.revenue_system.create_revenue_period(
            name="Q1 2023",
            start_date="2023-01-01",
            end_date="2023-03-31",
            total_revenue=10000.0
        )
        self.revenue_system.calculate_period_shares(period_id)
        
        # Finalize the period
        self.revenue_system.finalize_period(period_id)
        
        # Verify period was finalized
        period = self.revenue_system.periods[period_id]
        assert period["status"] == "finalized"
        assert "finalized_date" in period
        
        # Verify we can't modify a finalized period
        with pytest.raises(ValueError):
            self.revenue_system.calculate_period_shares(period_id)

    def test_record_payment(self):
        """Test recording a payment to a contributor."""
        # Create, calculate, and finalize a period first
        period_id = self.revenue_system.create_revenue_period(
            name="Q1 2023",
            start_date="2023-01-01",
            end_date="2023-03-31",
            total_revenue=10000.0
        )
        self.revenue_system.calculate_period_shares(period_id)
        self.revenue_system.finalize_period(period_id)
        
        # Record a payment
        payment_id = self.revenue_system.record_payment(
            period_id=period_id,
            user_id="user1",
            amount=5000.0,
            payment_method="bank_transfer",
            transaction_id="tx123"
        )
        
        # Verify payment was recorded
        period = self.revenue_system.periods[period_id]
        assert "payments" in period
        assert payment_id in period["payments"]
        payment = period["payments"][payment_id]
        assert payment["user_id"] == "user1"
        assert payment["amount"] == 5000.0
        assert payment["payment_method"] == "bank_transfer"
        assert payment["transaction_id"] == "tx123"
        assert "date" in payment
        
        # Try to record payment for non-existent period
        with pytest.raises(ValueError):
            self.revenue_system.record_payment(
                period_id="non_existent",
                user_id="user1",
                amount=1000.0,
                payment_method="bank_transfer"
            )

    def test_generate_period_report(self):
        """Test generating a report for a revenue period."""
        # Create, calculate, and finalize a period first
        period_id = self.revenue_system.create_revenue_period(
            name="Q1 2023",
            start_date="2023-01-01",
            end_date="2023-03-31",
            total_revenue=10000.0
        )
        self.revenue_system.calculate_period_shares(period_id)
        self.revenue_system.finalize_period(period_id)
        
        # Record some payments
        self.revenue_system.record_payment(
            period_id=period_id,
            user_id="user1",
            amount=5000.0,
            payment_method="bank_transfer"
        )
        
        # Generate report
        report = self.revenue_system.generate_period_report(period_id)
        
        # Verify report structure
        assert "period_name" in report
        assert "total_revenue" in report
        assert "contributors" in report
        assert "user1" in report["contributors"]
        assert "user2" in report["contributors"]
        assert "total_paid" in report
        assert "total_pending" in report
        assert report["total_paid"] == 5000.0
        assert report["total_pending"] == pytest.approx(5000.0)  # Total - paid 