use anchor_lang::prelude::*;
use crate::errors::ErrorCode;

/// Contribution data structure
#[account]
pub struct Contribution {
    /// Model that received the contribution
    pub model: Pubkey,
    
    /// Contributor's public key
    pub contributor: Pubkey,
    
    /// Contribution description
    pub description: String,
    
    /// Contribution type (e.g., "data_contribution", "code_improvement", "validation")
    pub contribution_type: String,
    
    /// Accuracy improvement (percentage as decimal 0.0-1.0)
    pub accuracy_improvement: f64,
    
    /// Performance improvement details (JSON string)
    pub performance_improvement: String,
    
    /// Contribution status
    pub status: ContributionStatus,
    
    /// Reward amount in MDNX tokens (if approved)
    pub reward_amount: u64,
    
    /// When the contribution was submitted
    pub created_at: i64,
    
    /// When the contribution was last updated
    pub updated_at: i64,
    
    /// When the contribution was processed (approved/rejected)
    pub processed_at: Option<i64>,
    
    /// Hash of the contribution data (for verification)
    pub contribution_hash: String,
    
    /// Notes about the contribution (for reviewers)
    pub notes: String,
}

/// Status of a contribution
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum ContributionStatus {
    /// Contribution has been submitted but not yet reviewed
    Pending,
    
    /// Contribution is under review
    InReview,
    
    /// Contribution has been approved
    Approved,
    
    /// Contribution has been rejected
    Rejected,
}

impl Contribution {
    pub const LEN: usize = 8 + // discriminator
        32 + // model
        32 + // contributor
        256 + // description (string)
        32 + // contribution_type (string)
        8 + // accuracy_improvement (f64)
        256 + // performance_improvement (string)
        4 + // status (enum)
        8 + // reward_amount
        8 + // created_at
        8 + // updated_at
        9 + // processed_at (Option<i64>)
        64 + // contribution_hash (string)
        256; // notes (string)
}

/// Contribution operation implementations
pub mod contribution_operations {
    use super::*;
    use crate::model_registry::ModelRegistry;
    use anchor_spl::token;
    
    /// Record a new contribution to a model
    pub fn record_contribution(
        ctx: Context<crate::RecordContribution>,
        description: String,
        contribution_type: String,
        accuracy_improvement: f64,
        performance_improvement: String,
        contribution_hash: String,
    ) -> Result<()> {
        let contribution = &mut ctx.accounts.contribution;
        let model = &mut ctx.accounts.model;
        let contributor = &ctx.accounts.contributor;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate input
        if accuracy_improvement < 0.0 || accuracy_improvement > 1.0 {
            return Err(ErrorCode::InvalidContributionImprovementValue.into());
        }
        
        if contribution_hash.len() < 16 {
            return Err(ErrorCode::InvalidDataHash.into());
        }
        
        // Initialize contribution
        contribution.model = model.key();
        contribution.contributor = contributor.key();
        contribution.description = description;
        contribution.contribution_type = contribution_type;
        contribution.accuracy_improvement = accuracy_improvement;
        contribution.performance_improvement = performance_improvement;
        contribution.status = ContributionStatus::Pending;
        contribution.reward_amount = 0;
        contribution.created_at = current_timestamp;
        contribution.updated_at = current_timestamp;
        contribution.processed_at = None;
        contribution.contribution_hash = contribution_hash;
        contribution.notes = String::new();
        
        // Update model contribution count
        model.contribution_count += 1;
        
        msg!("Contribution recorded for model {}", model.key());
        Ok(())
    }
    
    /// Review a contribution and update its status
    pub fn review_contribution(
        ctx: Context<ReviewContribution>,
        status: ContributionStatus,
        notes: String,
    ) -> Result<()> {
        let contribution = &mut ctx.accounts.contribution;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Update contribution
        contribution.status = ContributionStatus::InReview;
        contribution.notes = notes;
        contribution.updated_at = current_timestamp;
        
        msg!("Contribution now under review");
        Ok(())
    }
    
    /// Approve a contribution and distribute rewards
    pub fn approve_contribution(
        ctx: Context<crate::ApproveContribution>,
        reward_amount: u64,
    ) -> Result<()> {
        let contribution = &mut ctx.accounts.contribution;
        let model = &mut ctx.accounts.model;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Ensure contribution was not already processed
        if contribution.status == ContributionStatus::Approved || 
           contribution.status == ContributionStatus::Rejected {
            return Err(ErrorCode::ContributionAlreadyProcessed.into());
        }
        
        // Ensure model matches
        if contribution.model != model.key() {
            return Err(ErrorCode::ModelMismatch.into());
        }
        
        // Update model accuracy if contribution improves it
        if contribution.accuracy_improvement > 0.0 {
            let new_accuracy = model.accuracy + 
                (contribution.accuracy_improvement * (1.0 - model.accuracy));
            
            // Ensure accuracy doesn't exceed 1.0
            model.accuracy = new_accuracy.min(1.0);
        }
        
        // Update contribution status
        contribution.status = ContributionStatus::Approved;
        contribution.reward_amount = reward_amount;
        contribution.processed_at = Some(current_timestamp);
        contribution.updated_at = current_timestamp;
        
        // Transfer tokens if reward amount is greater than zero
        if reward_amount > 0 {
            // Transfer tokens from treasury to contributor
            let treasury = &ctx.accounts.treasury;
            let contributor_token_account = &ctx.accounts.contributor_token_account;
            let token_program = &ctx.accounts.token_program;
            let authority = &ctx.accounts.authority;
            
            // Create CPI context for token transfer
            let transfer_ctx = CpiContext::new(
                token_program.to_account_info(),
                token::Transfer {
                    from: treasury.to_account_info(),
                    to: contributor_token_account.to_account_info(),
                    authority: authority.to_account_info(),
                },
            );
            
            // Execute token transfer
            token::transfer(transfer_ctx, reward_amount)?;
            
            msg!("Transferred {} MDNX tokens to contributor", reward_amount);
        }
        
        msg!("Contribution approved for model {}", model.key());
        Ok(())
    }
    
    /// Reject a contribution
    pub fn reject_contribution(
        ctx: Context<RejectContribution>,
        rejection_reason: String,
    ) -> Result<()> {
        let contribution = &mut ctx.accounts.contribution;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Ensure contribution was not already processed
        if contribution.status == ContributionStatus::Approved || 
           contribution.status == ContributionStatus::Rejected {
            return Err(ErrorCode::ContributionAlreadyProcessed.into());
        }
        
        // Update contribution
        contribution.status = ContributionStatus::Rejected;
        contribution.notes = rejection_reason;
        contribution.processed_at = Some(current_timestamp);
        contribution.updated_at = current_timestamp;
        
        msg!("Contribution rejected");
        Ok(())
    }
}

/// Context for reviewing a contribution
#[derive(Accounts)]
pub struct ReviewContribution<'info> {
    /// Contribution to review
    #[account(mut)]
    pub contribution: Account<'info, Contribution>,
    
    /// Reviewer with permission to review
    pub reviewer: Signer<'info>,
}

/// Context for rejecting a contribution
#[derive(Accounts)]
pub struct RejectContribution<'info> {
    /// Contribution to reject
    #[account(mut)]
    pub contribution: Account<'info, Contribution>,
    
    /// Model account referenced by the contribution
    pub model: Account<'info, crate::model_registry::ModelRegistry>,
    
    /// Authority (must be model owner or MDNX authority)
    #[account(
        constraint = model.authority == authority.key() @ ErrorCode::Unauthorized
    )]
    pub authority: Signer<'info>,
} 