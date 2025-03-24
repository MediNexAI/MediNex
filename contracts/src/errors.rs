use anchor_lang::prelude::*;

#[error_code]
pub enum ErrorCode {
    #[msg("Unauthorized access")]
    UnauthorizedAccess,
    
    #[msg("Invalid token supply")]
    InvalidTokenSupply,
    
    #[msg("Token already initialized")]
    TokenAlreadyInitialized,
    
    #[msg("Model already registered")]
    ModelAlreadyRegistered,
    
    #[msg("Invalid model hash")]
    InvalidModelHash,
    
    #[msg("Model not found")]
    ModelNotFound,
    
    #[msg("Invalid accuracy value")]
    InvalidAccuracyValue,
    
    #[msg("Contribution already processed")]
    ContributionAlreadyProcessed,
    
    #[msg("Invalid contribution improvement value")]
    InvalidContributionValue,
    
    #[msg("Model mismatch")]
    ModelMismatch,
    
    #[msg("Invalid data hash")]
    InvalidDataHash,
    
    #[msg("Data already verified")]
    DataAlreadyVerified,
    
    #[msg("Invalid verification method")]
    InvalidVerificationMethod,
    
    #[msg("Invalid confidence score")]
    InvalidConfidenceScore,
    
    #[msg("Insufficient token balance")]
    InsufficientTokenBalance,
    
    #[msg("Invalid token account")]
    InvalidTokenAccount,
    
    #[msg("Operation rate limited")]
    RateLimited,
    
    #[msg("Invalid authority transfer state")]
    InvalidAuthorityTransferState,
    
    #[msg("Authority transfer expired")]
    AuthorityTransferExpired,
} 