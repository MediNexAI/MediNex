# MediNex AI Smart Contracts

This directory contains the Solana smart contracts for the MediNex AI platform, a decentralized ecosystem for medical AI model management, contribution, and verification.

## Overview

The MediNex contracts provide functionality for:

1. **Token Management**: MDNX token operations including initialization, minting, and authority management
2. **Model Registry**: Registering and managing AI models with performance metrics
3. **Contributions**: Recording and rewarding contributions to AI models
4. **Verification**: Verifying medical data and analysis results

## Contract Structure

The codebase is organized into modules:

- `lib.rs`: Main entry point and instruction handlers
- `errors.rs`: Error codes and messages
- `token.rs`: MDNX token implementation
- `model_registry.rs`: AI model registry functionality
- `contribution.rs`: Contribution management
- `verification.rs`: Data and analysis verification

## Key Features

### MDNX Token

- Token initialization with name, symbol, URI, and total supply
- Authority management with secure transfer mechanisms
- Token minting with rate limiting
- Treasury account for token distribution

### Model Registry

- Registration of AI models with detailed metadata
- Performance tracking including accuracy and confidence metrics
- Model updates and versioning
- Derived model creation (from parent models)

### Contributions

- Recording contributions with detailed metrics
- Contribution approval workflow
- Automatic reward distribution
- Impact tracking on model improvement

### Verification

- Medical data verification
- Analysis result verification
- Model output verification
- Expert verification for high-quality validation

## Development

### Prerequisites

- Rust (1.58 or later)
- Solana CLI (1.9 or later)
- Anchor Framework (0.24 or later)
- Node.js (16 or later) for testing scripts

### Build Instructions

```bash
# Build the contracts
anchor build

# Run tests
anchor test

# Deploy to devnet
anchor deploy --provider.cluster devnet
```

### Validation

A TypeScript validation script is provided to test the contracts on a local validator or devnet:

```bash
# Run the validation script
yarn validate
```

## Testing

Tests are organized in the `tests` directory:

- `medinex_tests.rs`: Rust tests for contract functionality
- TypeScript tests for client-side interaction

## Security Considerations

The contracts implement several security measures:

- Rate limiting for token minting
- Authority checks for sensitive operations
- Input validation for all parameters
- Secure authority transfer with two-step process

## License

These contracts are released under the MIT License. See the LICENSE file for details. 