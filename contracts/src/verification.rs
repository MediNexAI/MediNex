use anchor_lang::prelude::*;
use crate::errors::ErrorCode;

/// Verification data structure
#[account]
pub struct Verification {
    /// Type of verification
    pub verification_type: VerificationType,
    
    /// Data or analysis hash (SHA-256)
    pub data_hash: String,
    
    /// Verification method used
    pub verification_method: String,
    
    /// Confidence score (0.0-1.0)
    pub confidence_score: f64,
    
    /// Verifier's public key
    pub verifier: Pubkey,
    
    /// Model used for verification (if applicable)
    pub model: Option<Pubkey>,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Additional metadata (JSON string)
    pub metadata: String,
    
    /// Verification result details
    pub result_details: String,
}

/// Type of verification
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum VerificationType {
    /// Medical data verification
    MedicalData,
    
    /// Analysis result verification
    AnalysisResult,
    
    /// Model output verification
    ModelOutput,
    
    /// Expert verification
    ExpertReview,
}

impl Verification {
    pub const LEN: usize = 8 + // discriminator
        4 + // verification_type (enum)
        64 + // data_hash (string)
        64 + // verification_method (string)
        8 + // confidence_score (f64)
        32 + // verifier
        33 + // model (Option<Pubkey>)
        8 + // created_at
        512 + // metadata (string)
        512; // result_details (string)
}

/// Verification operation implementations
pub mod verification_operations {
    use super::*;
    
    /// Verify medical data
    pub fn verify_data(
        ctx: Context<crate::VerifyData>,
        data_hash: String,
        verification_method: String,
        confidence_score: f64,
        metadata: String,
        result_details: String,
    ) -> Result<()> {
        let verification = &mut ctx.accounts.verification;
        let verifier = &ctx.accounts.verifier;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate inputs
        if data_hash.len() < 16 {
            return Err(ErrorCode::InvalidDataHash.into());
        }
        
        if verification_method.is_empty() {
            return Err(ErrorCode::InvalidVerificationMethod.into());
        }
        
        if confidence_score < 0.0 || confidence_score > 1.0 {
            return Err(ErrorCode::InvalidConfidenceScore.into());
        }
        
        // Initialize verification
        verification.verification_type = VerificationType::MedicalData;
        verification.data_hash = data_hash;
        verification.verification_method = verification_method;
        verification.confidence_score = confidence_score;
        verification.verifier = verifier.key();
        verification.model = ctx.accounts.model.as_ref().map(|model| model.key());
        verification.created_at = current_timestamp;
        verification.metadata = metadata;
        verification.result_details = result_details;
        
        // If model is provided, update model verification count
        if let Some(model_account) = &ctx.accounts.model {
            let model = &mut model_account;
            model.verification_count += 1;
        }
        
        msg!("Medical data verified: {}", data_hash);
        Ok(())
    }
    
    /// Verify analysis results
    pub fn verify_analysis(
        ctx: Context<crate::VerifyAnalysis>,
        data_hash: String,
        verification_method: String,
        confidence_score: f64,
        metadata: String,
        result_details: String,
    ) -> Result<()> {
        let verification = &mut ctx.accounts.verification;
        let verifier = &ctx.accounts.verifier;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate inputs
        if data_hash.len() < 16 {
            return Err(ErrorCode::InvalidDataHash.into());
        }
        
        if verification_method.is_empty() {
            return Err(ErrorCode::InvalidVerificationMethod.into());
        }
        
        if confidence_score < 0.0 || confidence_score > 1.0 {
            return Err(ErrorCode::InvalidConfidenceScore.into());
        }
        
        // Initialize verification
        verification.verification_type = VerificationType::AnalysisResult;
        verification.data_hash = data_hash;
        verification.verification_method = verification_method;
        verification.confidence_score = confidence_score;
        verification.verifier = verifier.key();
        verification.model = ctx.accounts.model.as_ref().map(|model| model.key());
        verification.created_at = current_timestamp;
        verification.metadata = metadata;
        verification.result_details = result_details;
        
        // If model is provided, update model verification count
        if let Some(model_account) = &ctx.accounts.model {
            let model = &mut model_account;
            model.verification_count += 1;
        }
        
        msg!("Analysis result verified: {}", data_hash);
        Ok(())
    }
    
    /// Verify model output
    pub fn verify_model_output(
        ctx: Context<VerifyModelOutput>,
        data_hash: String,
        verification_method: String,
        confidence_score: f64,
        metadata: String,
        result_details: String,
    ) -> Result<()> {
        let verification = &mut ctx.accounts.verification;
        let verifier = &ctx.accounts.verifier;
        let model = &mut ctx.accounts.model;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate inputs
        if data_hash.len() < 16 {
            return Err(ErrorCode::InvalidDataHash.into());
        }
        
        if verification_method.is_empty() {
            return Err(ErrorCode::InvalidVerificationMethod.into());
        }
        
        if confidence_score < 0.0 || confidence_score > 1.0 {
            return Err(ErrorCode::InvalidConfidenceScore.into());
        }
        
        // Initialize verification
        verification.verification_type = VerificationType::ModelOutput;
        verification.data_hash = data_hash;
        verification.verification_method = verification_method;
        verification.confidence_score = confidence_score;
        verification.verifier = verifier.key();
        verification.model = Some(model.key());
        verification.created_at = current_timestamp;
        verification.metadata = metadata;
        verification.result_details = result_details;
        
        // Update model verification count
        model.verification_count += 1;
        
        msg!("Model output verified: {}", data_hash);
        Ok(())
    }
    
    /// Expert review verification
    pub fn expert_verification(
        ctx: Context<ExpertVerification>,
        data_hash: String,
        verification_method: String,
        confidence_score: f64,
        metadata: String,
        result_details: String,
    ) -> Result<()> {
        let verification = &mut ctx.accounts.verification;
        let verifier = &ctx.accounts.verifier;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate expert status (in a real system, would validate against a registry of experts)
        // For demonstration, we just validate input parameters
        
        // Validate inputs
        if data_hash.len() < 16 {
            return Err(ErrorCode::InvalidDataHash.into());
        }
        
        if verification_method.is_empty() {
            return Err(ErrorCode::InvalidVerificationMethod.into());
        }
        
        if confidence_score < 0.0 || confidence_score > 1.0 {
            return Err(ErrorCode::InvalidConfidenceScore.into());
        }
        
        // Initialize verification
        verification.verification_type = VerificationType::ExpertReview;
        verification.data_hash = data_hash;
        verification.verification_method = verification_method;
        verification.confidence_score = confidence_score;
        verification.verifier = verifier.key();
        verification.model = ctx.accounts.model.as_ref().map(|model| model.key());
        verification.created_at = current_timestamp;
        verification.metadata = metadata;
        verification.result_details = result_details;
        
        // If model is provided, update model verification count
        if let Some(model_account) = &ctx.accounts.model {
            let model = &mut model_account;
            model.verification_count += 1;
        }
        
        msg!("Expert verification completed: {}", data_hash);
        Ok(())
    }
}

/// Context for verifying model output
#[derive(Accounts)]
pub struct VerifyModelOutput<'info> {
    /// The verification account to create
    #[account(init, payer = verifier, space = Verification::LEN)]
    pub verification: Account<'info, Verification>,
    
    /// The model that generated the output
    #[account(mut)]
    pub model: Account<'info, crate::model_registry::ModelRegistry>,
    
    /// The verifier (payer)
    #[account(mut)]
    pub verifier: Signer<'info>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
}

/// Context for expert verification
#[derive(Accounts)]
pub struct ExpertVerification<'info> {
    /// The verification account to create
    #[account(init, payer = verifier, space = Verification::LEN)]
    pub verification: Account<'info, Verification>,
    
    /// The model used (optional)
    pub model: Option<Account<'info, crate::model_registry::ModelRegistry>>,
    
    /// The expert verifier (payer)
    #[account(mut)]
    pub verifier: Signer<'info>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
} 