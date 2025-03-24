use anchor_lang::prelude::*;
use anchor_spl::token::{self, Mint, Token, TokenAccount};
use solana_program::system_program;

// Import project modules
pub mod errors;
pub mod token;
pub mod model_registry;
pub mod contribution;
pub mod verification;

// Re-export key components
pub use errors::*;
pub use token::*;
pub use model_registry::*;
pub use contribution::*;
pub use verification::*;

declare_id!("MdNxToKenxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

#[program]
pub mod medinex_ai {
    use super::*;
    
    /// Initialize the MDNX token
    pub fn initialize_token(
        ctx: Context<InitializeToken>,
        name: String,
        symbol: String,
        uri: String,
        total_supply: u64,
    ) -> Result<()> {
        token_operations::initialize_token(ctx, name, symbol, uri, total_supply)
    }
    
    /// Register a new AI model
    pub fn register_model(
        ctx: Context<RegisterModel>,
        name: String,
        description: String,
        version: String,
        model_type: String,
        model_hash: String,
        accuracy: f64,
        performance_metrics: String,
    ) -> Result<()> {
        model_operations::register_model(
            ctx, name, description, version, model_type, model_hash, accuracy, performance_metrics
        )
    }
    
    /// Update model information
    pub fn update_model(
        ctx: Context<UpdateModel>,
        name: Option<String>,
        description: Option<String>,
        version: Option<String>,
        model_hash: Option<String>,
        accuracy: Option<f64>,
        performance_metrics: Option<String>,
    ) -> Result<()> {
        model_operations::update_model(
            ctx, name, description, version, model_hash, accuracy, performance_metrics
        )
    }
    
    /// Record a model contribution
    pub fn record_contribution(
        ctx: Context<RecordContribution>,
        description: String,
        contribution_type: String,
        accuracy_improvement: f64,
        performance_improvement: String,
        contribution_hash: String,
    ) -> Result<()> {
        contribution_operations::record_contribution(
            ctx,
            description,
            contribution_type,
            accuracy_improvement,
            performance_improvement,
            contribution_hash
        )
    }
    
    /// Approve a contribution and distribute rewards
    pub fn approve_contribution(
        ctx: Context<ApproveContribution>,
        reward_amount: u64,
    ) -> Result<()> {
        contribution_operations::approve_contribution(ctx, reward_amount)
    }
    
    /// Verify medical data
    pub fn verify_data(
        ctx: Context<VerifyData>,
        data_hash: String,
        verification_method: String,
        confidence_score: f64,
        metadata: String,
        result_details: String,
    ) -> Result<()> {
        verification_operations::verify_data(
            ctx,
            data_hash,
            verification_method,
            confidence_score,
            metadata,
            result_details
        )
    }
    
    /// Verify analysis results
    pub fn verify_analysis(
        ctx: Context<VerifyAnalysis>,
        data_hash: String,
        verification_method: String,
        confidence_score: f64,
        metadata: String,
        result_details: String,
    ) -> Result<()> {
        verification_operations::verify_analysis(
            ctx,
            data_hash,
            verification_method,
            confidence_score,
            metadata,
            result_details
        )
    }
    
    /// Verify model output
    pub fn verify_model_output(
        ctx: Context<verification::VerifyModelOutput>,
        data_hash: String,
        verification_method: String,
        confidence_score: f64,
        metadata: String,
        result_details: String,
    ) -> Result<()> {
        verification_operations::verify_model_output(
            ctx,
            data_hash,
            verification_method,
            confidence_score,
            metadata,
            result_details
        )
    }
    
    /// Expert verification
    pub fn expert_verification(
        ctx: Context<verification::ExpertVerification>,
        data_hash: String,
        verification_method: String,
        confidence_score: f64,
        metadata: String,
        result_details: String,
    ) -> Result<()> {
        verification_operations::expert_verification(
            ctx,
            data_hash,
            verification_method,
            confidence_score,
            metadata,
            result_details
        )
    }
}

/// Context for initializing the MDNX token
#[derive(Accounts)]
#[instruction(name: String, symbol: String, uri: String, total_supply: u64)]
pub struct InitializeToken<'info> {
    /// The MDNX token account
    #[account(init, payer = authority, space = 8 + MdnxToken::LEN)]
    pub mdnx_token: Account<'info, MdnxToken>,
    
    /// The token mint
    #[account(init, payer = authority, mint::decimals = 9, mint::authority = authority)]
    pub mint: Account<'info, Mint>,
    
    /// Authority account (payer)
    #[account(mut)]
    pub authority: Signer<'info>,
    
    /// Authority token account to receive initial supply
    #[account(
        init_if_needed,
        payer = authority,
        associated_token::mint = mint,
        associated_token::authority = authority,
    )]
    pub authority_token_account: Account<'info, TokenAccount>,
    
    /// Token program
    pub token_program: Program<'info, Token>,
    
    /// Associated token program
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
}

/// Context for registering an AI model
#[derive(Accounts)]
#[instruction(
    name: String, 
    description: String, 
    version: String, 
    model_type: String, 
    model_hash: String
)]
pub struct RegisterModel<'info> {
    /// Initialize a new model registry account
    #[account(init, payer = authority, space = ModelRegistry::LEN)]
    pub model_registry: Account<'info, ModelRegistry>,
    
    /// Model authority (payer)
    #[account(mut)]
    pub authority: Signer<'info>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
}

/// Context for updating an AI model
#[derive(Accounts)]
pub struct UpdateModel<'info> {
    /// Model registry to update
    #[account(mut, has_one = authority @ ErrorCode::UnauthorizedAccess)]
    pub model_registry: Account<'info, ModelRegistry>,
    
    /// Model authority
    pub authority: Signer<'info>,
}

/// Context for recording a contribution
#[derive(Accounts)]
#[instruction(
    description: String, 
    contribution_type: String,
    accuracy_improvement: f64,
    performance_improvement: String,
    contribution_hash: String
)]
pub struct RecordContribution<'info> {
    /// Initialize a new contribution record
    #[account(init, payer = contributor, space = Contribution::LEN)]
    pub contribution: Account<'info, Contribution>,
    
    /// The model being contributed to
    #[account(mut)]
    pub model_registry: Account<'info, ModelRegistry>,
    
    /// Contributor (payer)
    #[account(mut)]
    pub contributor: Signer<'info>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
}

/// Context for approving a contribution
#[derive(Accounts)]
pub struct ApproveContribution<'info> {
    /// Contribution to approve
    #[account(
        mut,
        constraint = contribution.model == model_registry.key() @ ErrorCode::ModelMismatch
    )]
    pub contribution: Account<'info, Contribution>,
    
    /// The model referenced by the contribution
    #[account(
        mut,
        constraint = model_registry.authority == authority.key() @ ErrorCode::Unauthorized
    )]
    pub model_registry: Account<'info, ModelRegistry>,
    
    /// The MDNX token account
    pub mdnx_token: Account<'info, MdnxToken>,
    
    /// Treasury token account (source of rewards)
    #[account(
        mut,
        constraint = treasury.mint == mdnx_token.mint @ ErrorCode::InvalidTokenAccount,
        constraint = treasury.key() == mdnx_token.treasury @ ErrorCode::InvalidTokenAccount
    )]
    pub treasury: Account<'info, TokenAccount>,
    
    /// Contributor's token account (destination for rewards)
    #[account(mut)]
    pub contributor_token_account: Account<'info, TokenAccount>,
    
    /// Authority (must be model owner)
    pub authority: Signer<'info>,
    
    /// Token program
    pub token_program: Program<'info, Token>,
}

/// Context for verifying medical data
#[derive(Accounts)]
#[instruction(
    data_hash: String,
    verification_method: String,
    confidence_score: f64,
    metadata: String,
    result_details: String
)]
pub struct VerifyData<'info> {
    /// Initialize a new data verification record
    #[account(init, payer = verifier, space = Verification::LEN)]
    pub verification: Account<'info, Verification>,
    
    /// The model used for verification (optional)
    pub model: Option<Account<'info, ModelRegistry>>,
    
    /// The verifier (payer)
    #[account(mut)]
    pub verifier: Signer<'info>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
}

/// Context for verifying analysis results
#[derive(Accounts)]
#[instruction(
    data_hash: String,
    verification_method: String,
    confidence_score: f64,
    metadata: String,
    result_details: String
)]
pub struct VerifyAnalysis<'info> {
    /// Initialize a new analysis verification record
    #[account(init, payer = verifier, space = Verification::LEN)]
    pub verification: Account<'info, Verification>,
    
    /// The model used for analysis (optional)
    pub model: Option<Account<'info, ModelRegistry>>,
    
    /// The verifier (payer)
    #[account(mut)]
    pub verifier: Signer<'info>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
} 