use anchor_lang::prelude::*;
use crate::errors::ErrorCode;

/// Model Registry data structure
#[account]
pub struct ModelRegistry {
    /// Model name
    pub name: String,
    
    /// Model description
    pub description: String,
    
    /// Version identifier
    pub version: String,
    
    /// Model type (e.g., "medical_imaging", "symptom_analysis")
    pub model_type: String,
    
    /// Model hash (SHA-256 of model file)
    pub model_hash: String,
    
    /// Model accuracy
    pub accuracy: f64,
    
    /// Performance metrics (JSON string with additional metrics)
    pub performance_metrics: String,
    
    /// Model authority
    pub authority: Pubkey,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Last update timestamp
    pub updated_at: i64,
    
    /// Number of contributions
    pub contribution_count: u64,
    
    /// Total number of verifications
    pub verification_count: u64,
    
    /// Average confidence score
    pub avg_confidence_score: f64,
    
    /// Number of uses
    pub usage_count: u64,
    
    /// Is the model verified
    pub is_verified: bool,
    
    /// Original parent model (if derived from another model)
    pub parent_model: Option<Pubkey>,
}

impl ModelRegistry {
    pub const LEN: usize = 8 + // discriminator
        64 + // name (string)
        256 + // description (string)
        32 + // version (string)
        32 + // model_type (string)
        64 + // model_hash (string)
        8 + // accuracy (f64)
        512 + // performance_metrics (string)
        32 + // authority
        8 + // created_at
        8 + // updated_at
        8 + // contribution_count
        8 + // verification_count
        8 + // avg_confidence_score
        8 + // usage_count
        1 + // is_verified
        33; // parent_model (Option<Pubkey>)
}

/// Model operation implementations
pub mod model_operations {
    use super::*;
    
    /// Register a new AI model
    pub fn register_model(
        ctx: Context<crate::RegisterModel>,
        name: String,
        description: String,
        version: String,
        model_type: String,
        model_hash: String,
        accuracy: f64,
        performance_metrics: String,
    ) -> Result<()> {
        let model = &mut ctx.accounts.model_registry;
        let authority = &ctx.accounts.authority;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate inputs
        if model_hash.len() < 16 {
            return Err(ErrorCode::InvalidModelHash.into());
        }
        
        if accuracy < 0.0 || accuracy > 1.0 {
            return Err(ErrorCode::InvalidAccuracyValue.into());
        }
        
        // Initialize model
        model.name = name;
        model.description = description;
        model.version = version;
        model.model_type = model_type;
        model.model_hash = model_hash;
        model.accuracy = accuracy;
        model.performance_metrics = performance_metrics;
        model.authority = authority.key();
        model.created_at = current_timestamp;
        model.updated_at = current_timestamp;
        model.contribution_count = 0;
        model.verification_count = 0;
        model.avg_confidence_score = 0.0;
        model.usage_count = 0;
        model.is_verified = false;
        model.parent_model = None;
        
        msg!("Model registered: {} v{}", model.name, model.version);
        Ok(())
    }
    
    /// Update model information
    pub fn update_model(
        ctx: Context<crate::UpdateModel>,
        name: Option<String>,
        description: Option<String>,
        version: Option<String>,
        model_hash: Option<String>,
        accuracy: Option<f64>,
        performance_metrics: Option<String>,
    ) -> Result<()> {
        let model = &mut ctx.accounts.model_registry;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Update fields if provided
        if let Some(name_val) = name {
            model.name = name_val;
        }
        
        if let Some(description_val) = description {
            model.description = description_val;
        }
        
        if let Some(version_val) = version {
            model.version = version_val;
        }
        
        if let Some(model_hash_val) = model_hash {
            if model_hash_val.len() < 16 {
                return Err(ErrorCode::InvalidModelHash.into());
            }
            model.model_hash = model_hash_val;
        }
        
        if let Some(accuracy_val) = accuracy {
            if accuracy_val < 0.0 || accuracy_val > 1.0 {
                return Err(ErrorCode::InvalidAccuracyValue.into());
            }
            model.accuracy = accuracy_val;
        }
        
        if let Some(performance_metrics_val) = performance_metrics {
            model.performance_metrics = performance_metrics_val;
        }
        
        // Update timestamp
        model.updated_at = current_timestamp;
        
        msg!("Model updated: {} v{}", model.name, model.version);
        Ok(())
    }
    
    /// Verify a model
    pub fn verify_model(
        ctx: Context<VerifyModel>,
    ) -> Result<()> {
        let model = &mut ctx.accounts.model_registry;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Set model as verified
        model.is_verified = true;
        model.updated_at = current_timestamp;
        
        msg!("Model verified: {} v{}", model.name, model.version);
        Ok(())
    }
    
    /// Record model usage
    pub fn record_usage(
        ctx: Context<RecordModelUsage>,
        confidence_score: f64,
    ) -> Result<()> {
        let model = &mut ctx.accounts.model_registry;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate confidence score
        if confidence_score < 0.0 || confidence_score > 1.0 {
            return Err(ErrorCode::InvalidConfidenceScore.into());
        }
        
        // Update usage statistics
        model.usage_count += 1;
        
        // Update average confidence score
        let old_avg = model.avg_confidence_score;
        let old_count = model.usage_count - 1;
        
        if old_count == 0 {
            model.avg_confidence_score = confidence_score;
        } else {
            model.avg_confidence_score = (old_avg * (old_count as f64) + confidence_score) / (model.usage_count as f64);
        }
        
        model.updated_at = current_timestamp;
        
        msg!("Model usage recorded for {} v{}, new usage count: {}", 
             model.name, model.version, model.usage_count);
        Ok(())
    }
    
    /// Create a derived model
    pub fn create_derived_model(
        ctx: Context<CreateDerivedModel>,
        name: String,
        description: String,
        version: String,
        model_type: String,
        model_hash: String,
        accuracy: f64,
        performance_metrics: String,
    ) -> Result<()> {
        let model = &mut ctx.accounts.derived_model;
        let parent_model = &ctx.accounts.parent_model;
        let authority = &ctx.accounts.authority;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate inputs
        if model_hash.len() < 16 {
            return Err(ErrorCode::InvalidModelHash.into());
        }
        
        if accuracy < 0.0 || accuracy > 1.0 {
            return Err(ErrorCode::InvalidAccuracyValue.into());
        }
        
        // Initialize model
        model.name = name;
        model.description = description;
        model.version = version;
        model.model_type = model_type;
        model.model_hash = model_hash;
        model.accuracy = accuracy;
        model.performance_metrics = performance_metrics;
        model.authority = authority.key();
        model.created_at = current_timestamp;
        model.updated_at = current_timestamp;
        model.contribution_count = 0;
        model.verification_count = 0;
        model.avg_confidence_score = 0.0;
        model.usage_count = 0;
        model.is_verified = false;
        model.parent_model = Some(parent_model.key());
        
        msg!("Derived model created: {} v{} from parent {}", 
             model.name, model.version, parent_model.key());
        Ok(())
    }
}

/// Context for verifying a model
#[derive(Accounts)]
pub struct VerifyModel<'info> {
    /// Model to verify
    #[account(mut)]
    pub model_registry: Account<'info, ModelRegistry>,
    
    /// Verifier with permission to verify models
    pub verifier: Signer<'info>,
    
    /// The MDNX token (used to check if verifier has authority)
    pub mdnx_token: Account<'info, crate::token::MdnxToken>,
}

/// Context for recording model usage
#[derive(Accounts)]
pub struct RecordModelUsage<'info> {
    /// Model being used
    #[account(mut)]
    pub model_registry: Account<'info, ModelRegistry>,
    
    /// User of the model
    pub user: Signer<'info>,
}

/// Context for creating a derived model
#[derive(Accounts)]
#[instruction(
    name: String, 
    description: String, 
    version: String, 
    model_type: String, 
    model_hash: String
)]
pub struct CreateDerivedModel<'info> {
    /// The new derived model
    #[account(init, payer = authority, space = ModelRegistry::LEN)]
    pub derived_model: Account<'info, ModelRegistry>,
    
    /// The parent model
    pub parent_model: Account<'info, ModelRegistry>,
    
    /// Model authority (payer)
    #[account(mut)]
    pub authority: Signer<'info>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
} 