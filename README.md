# Dataset Generator for Verifiable Credential Revocation
This script generates datasets for the research paper "Metadata Privacy for Verifiable Credential Revocation". It creates raw Bloom filter cascade data structures along with information to test privacy guarantees for Verifiable Credentials (VCs) revocation mechanisms.

## Background
This tool supports research on revocation mechanisms for Self-Sovereign Identity (SSI) systems. It addresses the challenge of allowing anyone to query the status of a single VC while preventing inference about the wider set of issued or revoked VCs. The generated datasets represent raw data structures and associated metadata for privacy analysis.

### Setup
Install the required dependencies:

```
pip install -r requirements.txt
```

### Configuration
The script uses a YAML configuration file (config.yaml) to set various parameters. Create this file in the same directory as the script.
Configuration fields:
* `job_type`: Determines the mode of operation for the script.
    * "series": Generates individual Bloom filter cascade bitstrings along with the number of revocations.
    * "classification": Generates pairs of Bloom filter cascade bitstrings with a label indicating whether they were created using the same or different parameterizations.
* `output_directory`: Specifies where the generated CSV file will be saved.
* `maxinc`: Maximum number of inclusions (revocations) in a data structure.
* `maxexc`: Maximum number of exclusions (non-revoked VCs) in a data structure.
* `n_samples`: Number of samples to generate.
* `fprs`: False Positive Rate targets. Default is [0.006] if not specified.
* `max_workers`: Number of workers for concurrent processing.
* `padding`: Enable (true) or disable (false) padding for consistent data dimensions. When enabled, this brings each sample to the same format for use as input in deep learning models to empirically check privacy guarantees.
### Usage
Run the script with:

```
python data_generator.py
```

The script will read the configuration, generate the dataset, and save it to a CSV file in the specified output directory.
### Data Generation Process

1. Data Structure Creation:
* For "series": Generates individual Bloom filter cascade bitstrings with associated number of revocations and non-revoked VCs.
* For "classification": Generates pairs of Bloom filter cascade bitstrings with a label indicating if they were created using the same or different parameterizations.
2. Padding (Optional):
* If enabled, ensures all samples have consistent dimensions by padding to the maximum number of cascades and maximum length.
* Prepares data for input into deep learning models for empirical privacy guarantee checks.
3. CSV Output:
* For "series": Each row contains the Bloom filter cascade bitstrings, number of revocations, and number of non-revoked VCs.
* For "classification": Each row contains two Bloom filter cascade bitstrings and a classification label."



## CSV Output Format

### Series Job Type

Each row contains:
* Bloom filter cascade bitstrings
* Number of revocations
* Number of non-revoked VCs

Example:

```
00000111000000000000000000000000...,00000111000000000000000000000000...,82,59
```

### Classification Job Type

Each row contains:
* Two Bloom filter cascade bitstrings
* Classification label (1 or -1)

Example:

```
00000111000000000000000000000000...;00000111000000000000000000000000...;1
```