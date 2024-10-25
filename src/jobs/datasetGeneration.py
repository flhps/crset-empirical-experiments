import src.cascade.cascadeUtils as cu
import concurrent.futures
import csv
import time
import os
import numpy as np
from tqdm import tqdm
import random

def get_cascade_bitstrings(cascade):
    """Extract bitstrings from cascade filters."""
    bitstrings = []
    for bloom_filter in cascade.filters:
        bitstring = bloom_filter.save_bytes()
        bitstrings.append(bitstring)
    return bitstrings

def generate_single_datapoint(r, s, rhat, p, k, parallelize):
    """Generate a single cascade datapoint."""
    start_time = time.time()
    
    # Generate valid (included) and revoked (excluded) IDs
    valid_ids = cu.gen_ids(r)  
    revoked_ids = cu.gen_ids_wo_overlap(s, valid_ids)
    
    # Create cascade
    cascade, tries = cu.try_cascade(
        valid_ids,
        revoked_ids,
        rhat,
        p=p,
        k=k,
        multi_process=parallelize
    )
    
    duration = time.time() - start_time
    return get_cascade_bitstrings(cascade), [r, s, duration]

def generate_dataset_parallel(params, n_samples):
    """Generate multiple cascade datapoints in parallel."""
    r = params["r"]
    s = params["s"]
    rhat = params["rhat"]
    p_values = params["p"]
    k = params["k"]
    parallelize = params.get("parallelize", False)
    
    X = []  # Will store cascade bitstrings
    y = np.empty([n_samples, 3])  # Will store [r, s, duration]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_data = {
            executor.submit(
                generate_single_datapoint, 
                r, s, rhat, 
                p_values[0],  # Using first p value
                k, 
                parallelize
            ): i for i in range(n_samples)
        }
        
        with tqdm(total=n_samples, desc="Generating cascades") as pbar:
            for future in concurrent.futures.as_completed(future_to_data):
                i = future_to_data[future]
                try:
                    bitstrings, metadata = future.result()
                    X.append(bitstrings)
                    y[i, :] = metadata
                    pbar.update(1)
                except Exception as exc:
                    print(f"Sample {i} generated an exception: {exc}")
    
    return X, y

def save_to_csv(X, y, filename):
    """Save generated data to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(['concatenated_bitstrings', 'num_included', 'num_excluded', 'duration'])
        
        for bitstrings, metadata in zip(X, y):
            concatenated_bitstring = ','.join(
                ''.join(format(byte, '08b') for byte in cascade) 
                for cascade in bitstrings
            )
            writer.writerow([
                concatenated_bitstring,
                int(metadata[0]),  # num_included (r)
                int(metadata[1]),  # num_excluded (s)
                metadata[2]        # duration
            ])

def run(params):
    """
    Main function to generate dataset based on provided parameters.
    
    Args:
        params (dict): Dictionary containing:
            - r (int): Number of valid IDs
            - s (int): Number of revoked IDs
            - rhat (int): Target number of IDs
            - p (list): List of probabilities
            - k (int): Number of hash functions
            - samples (int): Number of samples to generate
            - classification (bool): Whether to generate classification data
            - collectRuntime (bool): Whether to collect runtime data
            - parallelize (bool): Whether to parallelize computation
            - formatPadding (bool): Whether to pad format
            - outputDirectory (str): Directory to save output
    
    Returns:
        dict: Message indicating success or failure
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = params.get("outputDirectory", "data")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate dataset
        print(f"Generating {params['samples']} samples...")
        start_time = time.time()
        
        X, y = generate_dataset_parallel(params, params['samples'])
        
        # Save to CSV
        output_file = os.path.join(
            output_dir, 
            f"training-data-series-{int(time.time_ns())}.csv"
        )
        save_to_csv(X, y, output_file)
        
        end_time = time.time()
        
        return {
            "message": (
                f"Successfully generated {params['samples']} samples and "
                f"saved to {output_file}. "
                f"Time taken: {end_time - start_time:.2f} seconds"
            )
        }
    
    except Exception as e:
        return {
            "message": f"Failed to generate dataset: {str(e)}"
        }