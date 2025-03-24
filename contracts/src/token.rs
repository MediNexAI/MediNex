use anchor_lang::prelude::*;
use anchor_spl::token::{self, Mint, Token, TokenAccount};
use crate::errors::ErrorCode;

/// MDNX Token data structure
#[account]
pub struct MdnxToken {
    /// Token name
    pub name: String,
    
    /// Token symbol
    pub symbol: String,
    
    /// URI to metadata
    pub uri: String,
    
    /// Total supply of tokens
    pub total_supply: u64,
    
    /// Token authority
    pub authority: Pubkey,
    
    /// Is the token initialized
    pub is_initialized: bool,
    
    /// Last update timestamp
    pub last_update_timestamp: i64,
    
    /// Mint address
    pub mint: Pubkey,
    
    /// Proposed new authority (for two-step authority transfer)
    pub proposed_authority: Option<Pubkey>,
    
    /// Timestamp of authority transfer proposal
    pub authority_proposal_timestamp: i64,
    
    /// Last mint timestamp (for rate limiting)
    pub last_mint_timestamp: i64,
    
    /// Treasury account
    pub treasury: Pubkey,
}

impl MdnxToken {
    pub const LEN: usize = 8 + // discriminator
        32 + // name (string)
        8 + // symbol (string)
        128 + // uri (string)
        8 + // total_supply
        32 + // authority
        1 + // is_initialized
        8 + // last_update_timestamp
        32 + // mint
        33 + // proposed_authority (Option<Pubkey>)
        8 + // authority_proposal_timestamp
        8 + // last_mint_timestamp
        32; // treasury
}

/// Authority transfer state - used for two-step authority transfer
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub struct AuthorityTransferState {
    pub proposed_authority: Option<Pubkey>,
    pub proposal_timestamp: i64,
}

/// Token operation implementations
pub mod token_operations {
    use super::*;
    
    /// Initialize a new MDNX token
    pub fn initialize_token(
        ctx: Context<crate::InitializeToken>,
        name: String,
        symbol: String,
        uri: String,
        total_supply: u64,
    ) -> Result<()> {
        let token = &mut ctx.accounts.mdnx_token;
        let authority = &ctx.accounts.authority;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Validate inputs
        if total_supply == 0 {
            return Err(ErrorCode::InvalidTokenSupply.into());
        }
        
        // Initialize token
        token.name = name;
        token.symbol = symbol;
        token.uri = uri;
        token.total_supply = total_supply;
        token.authority = authority.key();
        token.is_initialized = true;
        token.last_update_timestamp = current_timestamp;
        token.mint = ctx.accounts.mint.key();
        token.proposed_authority = None;
        token.authority_proposal_timestamp = 0;
        token.last_mint_timestamp = 0;
        token.treasury = authority.key(); // Initially set treasury to authority
        
        msg!("MDNX token initialized with supply: {}", total_supply);
        Ok(())
    }
    
    /// Propose a new authority for the token
    pub fn propose_authority_transfer(
        ctx: Context<ProposeAuthorityTransfer>,
        new_authority: Pubkey,
    ) -> Result<()> {
        let token = &mut ctx.accounts.mdnx_token;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Only current authority can propose a transfer
        require_keys_eq!(
            token.authority,
            ctx.accounts.authority.key(),
            ErrorCode::UnauthorizedAccess
        );
        
        // Set proposed authority
        token.proposed_authority = Some(new_authority);
        token.authority_proposal_timestamp = current_timestamp;
        token.last_update_timestamp = current_timestamp;
        
        msg!("Authority transfer proposed to: {}", new_authority);
        Ok(())
    }
    
    /// Accept authority transfer (must be called by proposed authority)
    pub fn accept_authority_transfer(
        ctx: Context<AcceptAuthorityTransfer>,
    ) -> Result<()> {
        let token = &mut ctx.accounts.mdnx_token;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Verify proposed authority exists
        let proposed_authority = token.proposed_authority
            .ok_or(ErrorCode::InvalidAuthorityTransferState)?;
        
        // Verify caller is the proposed authority
        require_keys_eq!(
            proposed_authority,
            ctx.accounts.new_authority.key(),
            ErrorCode::UnauthorizedAccess
        );
        
        // Check if proposal hasn't expired (24 hour window)
        let proposal_age = current_timestamp - token.authority_proposal_timestamp;
        if proposal_age > 86400 { // 24 hours in seconds
            return Err(ErrorCode::AuthorityTransferExpired.into());
        }
        
        // Transfer authority
        token.authority = proposed_authority;
        token.proposed_authority = None;
        token.authority_proposal_timestamp = 0;
        token.last_update_timestamp = current_timestamp;
        
        msg!("Authority transfer accepted, new authority: {}", proposed_authority);
        Ok(())
    }
    
    /// Cancel an authority transfer proposal
    pub fn cancel_authority_transfer(
        ctx: Context<CancelAuthorityTransfer>,
    ) -> Result<()> {
        let token = &mut ctx.accounts.mdnx_token;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Only current authority can cancel a transfer
        require_keys_eq!(
            token.authority,
            ctx.accounts.authority.key(),
            ErrorCode::UnauthorizedAccess
        );
        
        // Cancel proposed authority
        token.proposed_authority = None;
        token.authority_proposal_timestamp = 0;
        token.last_update_timestamp = current_timestamp;
        
        msg!("Authority transfer cancelled");
        Ok(())
    }
    
    /// Mint MDNX tokens
    pub fn mint_tokens(
        ctx: Context<MintTokens>,
        amount: u64,
    ) -> Result<()> {
        let token = &mut ctx.accounts.mdnx_token;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Only authority can mint
        require_keys_eq!(
            token.authority,
            ctx.accounts.authority.key(),
            ErrorCode::UnauthorizedAccess
        );
        
        // Check rate limiting (minimum 1 hour between mints)
        let time_since_last_mint = current_timestamp - token.last_mint_timestamp;
        if time_since_last_mint < 3600 && token.last_mint_timestamp > 0 { // 1 hour in seconds
            return Err(ErrorCode::RateLimited.into());
        }
        
        // Mint tokens
        let cpi_accounts = token::MintTo {
            mint: ctx.accounts.mint.to_account_info(),
            to: ctx.accounts.destination.to_account_info(),
            authority: ctx.accounts.authority.to_account_info(),
        };
        
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_context = CpiContext::new(cpi_program, cpi_accounts);
        
        token::mint_to(cpi_context, amount)?;
        
        // Update token state
        token.last_mint_timestamp = current_timestamp;
        token.last_update_timestamp = current_timestamp;
        
        msg!("Minted {} MDNX tokens to {}", amount, ctx.accounts.destination.key());
        Ok(())
    }
    
    /// Set treasury account
    pub fn set_treasury(
        ctx: Context<SetTreasury>,
        new_treasury: Pubkey,
    ) -> Result<()> {
        let token = &mut ctx.accounts.mdnx_token;
        let current_timestamp = Clock::get()?.unix_timestamp;
        
        // Only authority can set treasury
        require_keys_eq!(
            token.authority,
            ctx.accounts.authority.key(),
            ErrorCode::UnauthorizedAccess
        );
        
        // Update treasury
        token.treasury = new_treasury;
        token.last_update_timestamp = current_timestamp;
        
        msg!("Treasury updated to: {}", new_treasury);
        Ok(())
    }
}

/// Context for proposing authority transfer
#[derive(Accounts)]
pub struct ProposeAuthorityTransfer<'info> {
    /// The MDNX token
    #[account(mut)]
    pub mdnx_token: Account<'info, MdnxToken>,
    
    /// Current authority
    pub authority: Signer<'info>,
}

/// Context for accepting authority transfer
#[derive(Accounts)]
pub struct AcceptAuthorityTransfer<'info> {
    /// The MDNX token
    #[account(mut)]
    pub mdnx_token: Account<'info, MdnxToken>,
    
    /// New authority accepting the transfer
    pub new_authority: Signer<'info>,
}

/// Context for cancelling authority transfer
#[derive(Accounts)]
pub struct CancelAuthorityTransfer<'info> {
    /// The MDNX token
    #[account(mut)]
    pub mdnx_token: Account<'info, MdnxToken>,
    
    /// Current authority
    pub authority: Signer<'info>,
}

/// Context for minting tokens
#[derive(Accounts)]
pub struct MintTokens<'info> {
    /// The MDNX token
    #[account(mut)]
    pub mdnx_token: Account<'info, MdnxToken>,
    
    /// Token mint
    #[account(mut)]
    pub mint: Account<'info, Mint>,
    
    /// Destination account
    #[account(mut)]
    pub destination: Account<'info, TokenAccount>,
    
    /// Authority
    pub authority: Signer<'info>,
    
    /// Token program
    pub token_program: Program<'info, Token>,
}

/// Context for setting treasury
#[derive(Accounts)]
pub struct SetTreasury<'info> {
    /// The MDNX token
    #[account(mut)]
    pub mdnx_token: Account<'info, MdnxToken>,
    
    /// Authority
    pub authority: Signer<'info>,
} 