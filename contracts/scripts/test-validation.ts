/**
 * MediNex AI Contract Validation Script
 * 
 * This script is used to validate the MediNex smart contracts on a local validator
 * or on the Solana devnet. It exercises the basic functionality of the contracts
 * to ensure they are working correctly.
 */

import * as anchor from '@project-serum/anchor';
import { Program } from '@project-serum/anchor';
import { PublicKey, Keypair, Connection, SystemProgram, SYSVAR_RENT_PUBKEY } from '@solana/web3.js';
import { TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID, Token } from '@solana/spl-token';
import fs from 'fs';
import path from 'path';
import * as chai from 'chai';
import { expect } from 'chai';

// Import the IDL from the build directory
const idl = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../target/idl/medinex_ai.json'), 'utf8'));

// Constants
const PROGRAM_ID = new PublicKey('MdNxToKenxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx');

// Helper function to create a connection to a Solana network
function getConnection(network: string = 'http://localhost:8899'): Connection {
  return new Connection(network, 'confirmed');
}

// Helper function to load a keypair from a file
function loadKeypair(filePath: string): Keypair {
  const keypairData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  return Keypair.fromSecretKey(new Uint8Array(keypairData));
}

// Main validation function
async function validateContracts() {
  console.log('Starting MediNex AI Contract Validation...');
  
  // Set up the connection and wallet
  const connection = getConnection();
  
  // Use a keypair for local testing
  const wallet = new anchor.Wallet(Keypair.generate());
  
  // Airdrop SOL to the wallet for transactions
  const airdropSignature = await connection.requestAirdrop(wallet.publicKey, 1000000000);
  await connection.confirmTransaction(airdropSignature);
  
  // Set up the provider and program
  const provider = new anchor.Provider(connection, wallet, { commitment: 'confirmed' });
  const program = new Program(idl, PROGRAM_ID, provider);
  
  // Create keypairs for our accounts
  const mdnxToken = Keypair.generate();
  const mint = Keypair.generate();
  
  // Token initialization parameters
  const tokenName = 'MediNex Token';
  const tokenSymbol = 'MDNX';
  const tokenUri = 'https://medinex.life/token';
  const totalSupply = new anchor.BN(1000000000);
  
  // Find the authority token account PDA
  const authorityTokenAccount = await Token.getAssociatedTokenAddress(
    ASSOCIATED_TOKEN_PROGRAM_ID,
    TOKEN_PROGRAM_ID,
    mint.publicKey,
    wallet.publicKey
  );
  
  console.log('Initializing token...');
  try {
    // Initialize token
    await program.rpc.initializeToken(
      tokenName,
      tokenSymbol,
      tokenUri,
      totalSupply,
      {
        accounts: {
          mdnxToken: mdnxToken.publicKey,
          mint: mint.publicKey,
          authority: wallet.publicKey,
          authorityTokenAccount,
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        },
        signers: [mdnxToken, mint],
      }
    );
    console.log('Token initialized successfully!');
    
    // Create a model registry
    const modelRegistry = Keypair.generate();
    const modelName = 'Medical Imaging AI';
    const modelDescription = 'AI model for analyzing medical images';
    const modelVersion = '1.0.0';
    const modelType = 'medical_imaging';
    const modelHash = 'abcdef1234567890abcdef1234567890';
    const accuracy = 0.95;
    const performanceMetrics = JSON.stringify({
      precision: 0.94,
      recall: 0.96,
      f1_score: 0.95,
    });
    
    console.log('Registering model...');
    await program.rpc.registerModel(
      modelName,
      modelDescription,
      modelVersion,
      modelType,
      modelHash,
      accuracy,
      performanceMetrics,
      {
        accounts: {
          modelRegistry: modelRegistry.publicKey,
          authority: wallet.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        },
        signers: [modelRegistry],
      }
    );
    console.log('Model registered successfully!');
    
    // Create a contribution
    const contribution = Keypair.generate();
    const contributionDescription = 'Improved model with more training data';
    const contributionType = 'data_addition';
    const accuracyImprovement = 0.03;
    const performanceImprovement = JSON.stringify({
      precision_improvement: 0.02,
      recall_improvement: 0.04,
    });
    const contributionHash = '0123456789abcdef0123456789abcdef';
    
    console.log('Recording contribution...');
    await program.rpc.recordContribution(
      contributionDescription,
      contributionType,
      accuracyImprovement,
      performanceImprovement,
      contributionHash,
      {
        accounts: {
          contribution: contribution.publicKey,
          model: modelRegistry.publicKey,
          contributor: wallet.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        },
        signers: [contribution],
      }
    );
    console.log('Contribution recorded successfully!');
    
    // Create a data verification
    const verification = Keypair.generate();
    const dataHash = 'fedcba9876543210fedcba9876543210';
    const verificationMethod = 'cryptographic_hash';
    const confidenceScore = 0.99;
    const metadata = JSON.stringify({
      source: 'hospital_data',
      timestamp: Date.now(),
    });
    const resultDetails = JSON.stringify({
      validation_steps: [
        'integrity_check',
        'source_verification',
        'format_validation',
      ],
      status: 'verified',
    });
    
    console.log('Verifying data...');
    await program.rpc.verifyData(
      dataHash,
      verificationMethod,
      confidenceScore,
      metadata,
      resultDetails,
      {
        accounts: {
          verification: verification.publicKey,
          model: null, // Optional model
          verifier: wallet.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        },
        signers: [verification],
      }
    );
    console.log('Data verified successfully!');
    
    console.log('All validations completed successfully!');
    
  } catch (error) {
    console.error('Validation failed with error:', error);
    throw error;
  }
}

// Run the validation if this script is executed directly
if (require.main === module) {
  validateContracts()
    .then(() => process.exit(0))
    .catch(error => {
      console.error(error);
      process.exit(1);
    });
}

export { validateContracts }; 