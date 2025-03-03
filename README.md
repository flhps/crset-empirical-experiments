# CRSet: Privacy-Preserving Credential Revocation Dataset Generator

This repository contains the dataset generation code for the research paper **[CRSet: Non-Interactive Verifiable Credential Revocation with Metadata Privacy for Issuers and Everyone Else](https://arxiv.org/abs/2501.17089)**.

## Background

This tool supports research on privacy-preserving revocation mechanisms for Self-Sovereign Identity (SSI) systems. CRSet allows anyone to verify the revocation status of individual Verifiable Credentials while preventing inference about issuer metrics like total issuance volume or revocation patterns. The generator creates test datasets for analyzing both privacy and performance characteristics.

## Features

- Generate Bloom filter cascades with configurable parameters
- Support for privacy-preserving padding
- Parallel processing capabilities
- Flexible output formats for machine learning analysis
- Performance benchmarking tools
- Support for both single cascade and paired cascade generation

## Setup

Install the required dependencies:

```sh
pip install -r requirements.txt
```

You'll also need to install the custom version of the rbloom library for Bloom filter operations ([rbloom](https://github.com/jfelixh/rbloom)).

## Dataset Generation Modes

The generator supports multiple modes configured via `jobs.yaml`:

### Single Cascade Mode

Generates individual Bloom filter cascades with configurable parameters:

- With/without padding for privacy protection
- Configurable false positive rates and cascade sizes
- Metadata tracking revocations and construction metrics

### Pair Generation Mode

Creates pairs of cascades for analyzing cascade comparability:

- Pairs can be from same or different parameter sets
- Optional padding for privacy analysis
- Includes classification labels for relationship between pairs

## Configuration

Key parameters in `jobs.yaml`:

```yaml
datasetGeneration:
  params:
    pairs_mode: bool        # Generate pairs or single cascades
    use_padding: bool       # Enable privacy-preserving padding
    rhat: int               # Target capacity for valid credentials
    p: float                # False positive rate (typically 0.53)
    k: int                  # Number of hash functions
    samples: int            # Number of samples to generate
    parallelize: bool       # Enable parallel generation
    outputDirectory: str    # Output directory path
    pad_output_format: bool # Pad output to consistent dimensions
```

## Output Format

Generated CSV files contain:

### Single Mode

```
concatenated_bitstrings,num_included,num_excluded,duration,tries
```

Where:

- concatenated_bitstrings: Base64-encoded Bloom filter cascade
- num_included: Number of valid credentials
- num_excluded: Number of revoked credentials
- num_excluded: Number of revoked credentials
- tries: Number of attempts needed for successful construction

### Pairs Mode

```
cascade1_bitstrings,cascade2_bitstrings,r1,s1,r2,s2,duration,total_tries,identical
```

Where:

- cascade1_bitstrings: First Bloom filter cascade
- cascade2_bitstrings: Second Bloom filter cascade
- r1, r2: Number of valid credentials for each cascade
- s1, s2: Number of revoked credentials for each cascade
- duration: Total construction time
- total_tries: Combined construction attempts
- identical: Boolean indicating if cascades were generated with same parameters

### Usage

1. Configure your experiment in `jobs.yaml`
2. Run the dataset generator:

```ssh
python jobRunner.py
```

The script will:

1. Read the job configuration
2. Generate datasets according to specified parameters
3. Save output files to the configured directory

## Advanced Usage

### Benchmarking

The tool includes benchmarking capabilities for:

- Construction time analysis
- Cascade size optimization

### Privacy Analysis

- Machine learning-based analysis
- Cascade comparison tests

## Acknowledgments

We thank the Ethereum Foundation for funding this work with an Ethereum Academic Grant under reference number FY24-1545.

## Links and References

- ![arXiv](https://img.shields.io/badge/arXiv-2501.17089-b31b1b.svg) **[CRSet: Non-Interactive Verifiable Credential Revocation with Metadata Privacy for Issuers and Everyone Else](https://arxiv.org/abs/2501.17089)**  
  _Hoops et al., 2025._
- ![GitHub](https://img.shields.io/badge/GitHub-crset--demo-blue?logo=github) **[crset-issuer-backend](https://github.com/jfelixh/crset-demo)**
- ![GitHub](https://img.shields.io/badge/GitHub-crset--issuer--backend-blue?logo=github) **[crset-issuer-backend](https://github.com/jfelixh/crset-issuer-backend)**
- ![GitHub](https://img.shields.io/badge/GitHub-crset--check-blue?logo=github) **[crset-check](https://github.com/jfelixh/crset-check)**
- ![GitHub](https://img.shields.io/badge/GitHub-crset--cascade-blue?logo=github) **[crset-cascade](https://github.com/jfelixh/crset-cascade/)**
