# MediNex AI Contract Architecture

This document outlines the architecture of the Solana blockchain smart contracts used in the MediNex AI platform.

## Overview

The MediNex AI smart contracts serve several critical purposes:

1. **Model Registration and Verification**: Record and verify AI model information
2. **Contribution Tracking**: Record and attribute contributions to AI models
3. **Revenue Distribution**: Manage token distributions based on contributions
4. **Medical Data Validation**: Ensure the integrity and provenance of medical data

## Contract Architecture

```
                      +-------------------+
                      |                   |
                      |  MediNexToken     |
                      |                   |
                      +---------+---------+
                                |
                                v
                    +---------------------+
                    |                     |
                    |  ModelRegistry      |
                    |                     |
                    +----+----------+-----+
                         |          |
            +------------+          +------------+
            v                                    v
+---------------------+               +---------------------+
|                     |               |                     |
|  ContributionTracker|               |  DataValidation     |
|                     |               |                     |
+-----------+---------+               +---------------------+
            |
            v
+---------------------+
|                     |
|  RevenueDistributor |
|                     |
+---------------------+
```

## Smart Contracts

### 1. MediNexToken

**Purpose**: Implements the MDNX token functionality according to the SPL token standard.

**Key Functions**:
- Token minting and burning
- Token transfers
- Token balance management
- Token allowances

### 2. ModelRegistry

**Purpose**: Records AI model information and manages model versions.

**Key Functions**:
- Register new AI models
- Update model metadata
- Record model versions and improvements
- Link models to their contributors
- Verify model integrity through hashing

**Key Structures**:
```rust
pub struct Model {
    pub id: Pubkey,
    pub name: String,
    pub description: String,
    pub version: String,
    pub model_type: ModelType,
    pub hash: [u8; 32],  // SHA-256 hash of the model
    pub creator: Pubkey,
    pub created_at: i64,
    pub updated_at: i64,
}

pub enum ModelType {
    MedicalImaging,
    MedicalText,
    MedicalDiagnostic,
    MedicalGeneral,
}
```

### 3. ContributionTracker

**Purpose**: Records and verifies contributions made to AI models.

**Key Functions**:
- Register new contributions
- Record contribution types (code, data, validation, etc.)
- Link contributions to specific models
- Verify contribution authenticity
- Track contribution metrics

**Key Structures**:
```rust
pub struct Contribution {
    pub id: Pubkey,
    pub contributor: Pubkey,
    pub model_id: Pubkey,
    pub contribution_type: ContributionType,
    pub description: String,
    pub metadata: String,  // JSON metadata
    pub timestamp: i64,
    pub verification_status: VerificationStatus,
}

pub enum ContributionType {
    ModelCode,
    TrainingData,
    Validation,
    Testing,
    Documentation,
}

pub enum VerificationStatus {
    Pending,
    Verified,
    Rejected,
}
```

### 4. RevenueDistributor

**Purpose**: Manages token distributions based on contributions.

**Key Functions**:
- Calculate revenue shares based on contribution metrics
- Distribute tokens to contributors
- Record distribution history
- Manage vesting schedules
- Handle distribution disputes

**Key Structures**:
```rust
pub struct Distribution {
    pub id: Pubkey,
    pub model_id: Pubkey,
    pub total_amount: u64,
    pub distribution_date: i64,
    pub status: DistributionStatus,
}

pub enum DistributionStatus {
    Pending,
    Completed,
    Failed,
    Disputed,
}

pub struct DistributionShare {
    pub distribution_id: Pubkey,
    pub contributor: Pubkey,
    pub amount: u64,
    pub percentage: u8,  // Percentage multiplied by 100 (e.g., 45.5% = 4550)
    pub claimed: bool,
    pub claim_date: Option<i64>,
}
```

### 5. DataValidation

**Purpose**: Ensures the integrity and provenance of medical data used in AI training.

**Key Functions**:
- Record data sources and metadata
- Verify data integrity through hashing
- Manage data access permissions
- Track data usage in model training
- Ensure compliance with privacy regulations

**Key Structures**:
```rust
pub struct DataSource {
    pub id: Pubkey,
    pub name: String,
    pub source_type: DataSourceType,
    pub hash: [u8; 32],  // SHA-256 hash of the data
    pub metadata: String,  // JSON metadata
    pub owner: Pubkey,
    pub created_at: i64,
    pub privacy_level: PrivacyLevel,
}

pub enum DataSourceType {
    Imaging,
    LabResults,
    ClinicalNotes,
    ResearchPaper,
    AnonymizedPatientData,
}

pub enum PrivacyLevel {
    Public,
    Restricted,
    Private,
    Sensitive,
}
```

## Cross-Contract Interactions

1. **Model Registration Flow**:
   - MediNexToken -> ModelRegistry: Verify token stake for model registration
   - ModelRegistry -> ContributionTracker: Register initial contribution for model creator
   - ModelRegistry -> DataValidation: Link training data sources to the model

2. **Contribution Flow**:
   - ContributionTracker -> ModelRegistry: Update model version on significant contributions
   - ContributionTracker -> RevenueDistributor: Update contribution metrics for revenue calculation

3. **Revenue Distribution Flow**:
   - RevenueDistributor -> ContributionTracker: Fetch contribution metrics
   - RevenueDistributor -> MediNexToken: Execute token transfers to contributors

4. **Data Validation Flow**:
   - DataValidation -> ModelRegistry: Verify data usage in models
   - DataValidation -> ContributionTracker: Register data contributions

## Security Considerations

- All contract functions implement proper access control
- Critical functions require multiple signatures
- Sensitive operations use time-locks
- Contract upgrades follow a governance process
- Regular security audits are performed

## Future Enhancements

1. Integration with decentralized identity systems
2. Implementation of quadratic funding mechanisms
3. Support for cross-chain interoperability
4. Enhanced governance mechanisms for community decision-making