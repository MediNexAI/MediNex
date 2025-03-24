#![cfg(feature = "test-bpf")]

use {
    anchor_lang::{prelude::*, solana_program::system_program},
    anchor_spl::{
        associated_token::AssociatedToken,
        token::{Mint, Token, TokenAccount},
    },
    solana_program::instruction::Instruction,
    solana_program_test::*,
    solana_sdk::{
        account::Account,
        signature::{Keypair, Signer},
        transaction::Transaction,
    },
    std::str::FromStr,
    medinex_ai::{
        InitializeToken, RegisterModel, RecordContribution, VerifyData,
        ApproveContribution, VerifyAnalysis, 
    },
};

// Constants for testing
const MEDINEX_PROGRAM_ID: &str = "MdNxToKenxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

// Helper function to create an account with SOL
async fn create_and_fund_account(
    banks_client: &mut BanksClient,
    recent_blockhash: &solana_sdk::hash::Hash,
    account: &Keypair,
    lamports: u64,
) {
    let tx = Transaction::new_signed_with_payer(
        &[solana_program::system_instruction::transfer(
            &banks_client.get_rent().await.unwrap().0,
            &account.pubkey(),
            lamports,
        )],
        Some(&banks_client.get_rent().await.unwrap().0),
        &[&banks_client.get_rent().await.unwrap().1],
        *recent_blockhash,
    );
    banks_client.process_transaction(tx).await.unwrap();
}

// Helper function to create a program address
fn find_program_address(
    seeds: &[&[u8]],
    program_id: &Pubkey,
) -> (Pubkey, u8) {
    Pubkey::find_program_address(seeds, program_id)
}

#[tokio::test]
async fn test_initialize_token() {
    // Set up the test environment
    let program_id = Pubkey::from_str(MEDINEX_PROGRAM_ID).unwrap();
    let mut program_test = ProgramTest::new(
        "medinex_ai",
        program_id,
        processor!(medinex_ai::entry),
    );
    
    // Add accounts
    let authority = Keypair::new();
    
    // Fund the authority with SOL
    program_test.add_account(
        authority.pubkey(),
        Account {
            lamports: 1000000000,
            data: vec![],
            owner: system_program::ID,
            executable: false,
            rent_epoch: 0,
        },
    );
    
    // Start the test environment
    let (mut banks_client, payer, recent_blockhash) = program_test.start().await;
    
    // Create the token mint
    let mint = Keypair::new();
    
    // Create the MDNX token account
    let mdnx_token = Keypair::new();
    
    // Create the authority's token account
    let (authority_token_account, _) = Pubkey::find_program_address(
        &[
            authority.pubkey().as_ref(),
            anchor_spl::token::ID.as_ref(),
            mint.pubkey().as_ref(),
        ],
        &anchor_spl::associated_token::ID,
    );
    
    // Initialize token instruction
    let initialize_token_ix = Instruction {
        program_id,
        accounts: InitializeToken {
            mdnx_token: mdnx_token.pubkey(),
            mint: mint.pubkey(),
            authority: authority.pubkey(),
            authority_token_account,
            token_program: anchor_spl::token::ID,
            associated_token_program: anchor_spl::associated_token::ID,
            system_program: system_program::ID,
            rent: solana_program::sysvar::rent::ID,
        }
        .to_account_metas(None),
        data: medinex_ai::instruction::InitializeToken {
            name: "MediNex Token".to_string(),
            symbol: "MDNX".to_string(),
            uri: "https://medinex.life/token".to_string(),
            total_supply: 1000000000,
        }
        .data(),
    };
    
    // Create and sign the transaction
    let tx = Transaction::new_signed_with_payer(
        &[initialize_token_ix],
        Some(&authority.pubkey()),
        &[&authority, &mint, &mdnx_token],
        recent_blockhash,
    );
    
    // Process the transaction
    banks_client.process_transaction(tx).await.expect("Failed to initialize token");
    
    // TODO: Add verification that token was properly initialized
    // This would involve fetching the account and checking the data
}

#[tokio::test]
async fn test_model_registry() {
    // Set up the test environment
    let program_id = Pubkey::from_str(MEDINEX_PROGRAM_ID).unwrap();
    let mut program_test = ProgramTest::new(
        "medinex_ai",
        program_id,
        processor!(medinex_ai::entry),
    );
    
    // Add accounts
    let authority = Keypair::new();
    
    // Fund the authority with SOL
    program_test.add_account(
        authority.pubkey(),
        Account {
            lamports: 1000000000,
            data: vec![],
            owner: system_program::ID,
            executable: false,
            rent_epoch: 0,
        },
    );
    
    // Start the test environment
    let (mut banks_client, payer, recent_blockhash) = program_test.start().await;
    
    // Create the model registry account
    let model_registry = Keypair::new();
    
    // Register model instruction
    let register_model_ix = Instruction {
        program_id,
        accounts: RegisterModel {
            model_registry: model_registry.pubkey(),
            authority: authority.pubkey(),
            system_program: system_program::ID,
            rent: solana_program::sysvar::rent::ID,
        }
        .to_account_metas(None),
        data: medinex_ai::instruction::RegisterModel {
            name: "Medical Imaging Model".to_string(),
            description: "AI model for medical image analysis".to_string(),
            version: "1.0.0".to_string(),
            model_type: "medical_imaging".to_string(),
            model_hash: "abcdef1234567890abcdef1234567890".to_string(),
            accuracy: 0.95,
            performance_metrics: "{\"precision\": 0.94, \"recall\": 0.96, \"f1_score\": 0.95}".to_string(),
        }
        .data(),
    };
    
    // Create and sign the transaction
    let tx = Transaction::new_signed_with_payer(
        &[register_model_ix],
        Some(&authority.pubkey()),
        &[&authority, &model_registry],
        recent_blockhash,
    );
    
    // Process the transaction
    banks_client.process_transaction(tx).await.expect("Failed to register model");
    
    // TODO: Add verification that model was properly registered
}

#[tokio::test]
async fn test_contribution() {
    // TODO: Implement contribution testing
    // This would involve:
    // 1. Setting up test environment
    // 2. Initializing token and model
    // 3. Recording a contribution
    // 4. Approving the contribution and verifying token transfer
}

#[tokio::test]
async fn test_verification() {
    // TODO: Implement verification testing
    // This would involve:
    // 1. Setting up test environment
    // 2. Initializing token and model
    // 3. Verifying data or analysis
    // 4. Checking that verification record was created correctly
} 