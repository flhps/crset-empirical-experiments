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

def generate_single_series_datapoint(r, s, rhat, p, k, parallelize):
    """Generate a single cascade datapoint for series mode."""
    start_time = time.time()
    
    valid_ids = cu.gen_ids(r)  
    revoked_ids = cu.gen_ids_wo_overlap(s, valid_ids)
    
    cascade, tries = cu.try_cascade(
        valid_ids,
        revoked_ids,
        rhat,
        p=p,
        k=k,
        multi_process=parallelize
    )
    
    duration = time.time() - start_time
    return get_cascade_bitstrings(cascade), [r, s, duration, tries]

def generate_single_classification_datapoint(params, same_class=True):
    """Generate a pair of cascades for classification mode."""
    start_time = time.time()
    
    # First cascade parameters
    r1 = params["r"]
    s1 = params["s"]
    rhat1 = params["rhat"]
    p1 = params["p"][0]
    k1 = params["k"]
    
    # Second cascade parameters - either identical or slightly different
    if same_class:
        r2, s2, rhat2, p2, k2 = r1, s1, rhat1, p1, k1
    else:
        # Randomly modify parameters within a reasonable range
        r2 = int(r1 * random.uniform(0.8, 1.2))
        s2 = int(s1 * random.uniform(0.8, 1.2))
        rhat2 = int(rhat1 * random.uniform(0.8, 1.2))
        p2 = p1 * random.uniform(0.8, 1.2)
        k2 = k1  # Keep k constant as it's typically a fixed parameter
    
    # Generate first cascade
    valid_ids1 = cu.gen_ids(r1)
    revoked_ids1 = cu.gen_ids_wo_overlap(s1, valid_ids1)
    cascade1, tries1 = cu.try_cascade(valid_ids1, revoked_ids1, rhat1, p=p1, k=k1, multi_process=params["parallelize"])
    
    # Generate second cascade
    valid_ids2 = cu.gen_ids(r2)
    revoked_ids2 = cu.gen_ids_wo_overlap(s2, valid_ids2)
    cascade2, tries2 = cu.try_cascade(valid_ids2, revoked_ids2, rhat2, p=p2, k=k2, multi_process=params["parallelize"])
    
    duration = time.time() - start_time
    
    return (
        [get_cascade_bitstrings(cascade1), get_cascade_bitstrings(cascade2)],
        [r1, s1, r2, s2, duration, tries1 + tries2, int(same_class)]
    )

def generate_dataset_parallel(params, n_samples):
    """Generate multiple cascade datapoints in parallel."""
    if params.get("classification", False):
        return generate_classification_dataset(params, n_samples)
    else:
        return generate_series_dataset(params, n_samples)

def generate_series_dataset(params, n_samples):
    """Generate dataset for series mode."""
    X = []
    y = np.empty([n_samples, 4])  # [r, s, duration, tries]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_data = {
            executor.submit(
                generate_single_series_datapoint, 
                params["r"], params["s"], params["rhat"], 
                params["p"][0], params["k"], 
                params.get("parallelize", False)
            ): i for i in range(n_samples)
        }
        
        with tqdm(total=n_samples, desc="Generating series data") as pbar:
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

def generate_classification_dataset(params, n_samples):
    """Generate dataset for classification mode."""
    X = []
    y = np.empty([n_samples, 7])  # [r1, s1, r2, s2, duration, total_tries, same_class]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Generate balanced dataset (50% same class, 50% different class)
        future_to_data = {}
        for i in range(n_samples):
            same_class = i < n_samples // 2
            future_to_data[executor.submit(
                generate_single_classification_datapoint,
                params,
                same_class
            )] = i
        
        with tqdm(total=n_samples, desc="Generating classification data") as pbar:
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

def process_cascade_bitstrings(X, remove_header_bits=False):
    """Process cascade bitstrings with optional header bit removal."""
    processed_X = []
    for sample in X:
        if isinstance(sample[0], list):  # Classification mode (pair of cascades)
            processed_sample = []
            for cascade_set in sample:
                processed_cascades = []
                for cascade in cascade_set:
                    bitstring = ''.join(format(byte, '08b') for byte in cascade)
                    if remove_header_bits:
                        bitstring = bitstring[64:]
                    processed_cascades.append(bitstring)
                processed_sample.append(processed_cascades)
            processed_X.append(processed_sample)
        else:  # Series mode (single cascade)
            processed_cascades = []
            for cascade in sample:
                bitstring = ''.join(format(byte, '08b') for byte in cascade)
                if remove_header_bits:
                    bitstring = bitstring[64:]
                processed_cascades.append(bitstring)
            processed_X.append(processed_cascades)
    return processed_X

def save_to_csv(X, y, filename, classification_mode=False):
    """Save generated data to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE, escapechar='\\')
        
        if classification_mode:
            writer.writerow([
                'cascade1_bitstrings', 'cascade2_bitstrings',
                'r1', 's1', 'r2', 's2', 'duration', 'total_tries', 'same_class'
            ])
            for (bitstrings1, bitstrings2), metadata in zip(X, y):
                writer.writerow([
                    ','.join(bitstrings1),
                    ','.join(bitstrings2),
                    int(metadata[0]),  # r1
                    int(metadata[1]),  # s1
                    int(metadata[2]),  # r2
                    int(metadata[3]),  # s2
                    metadata[4],       # duration
                    int(metadata[5]),  # total_tries
                    int(metadata[6])   # same_class
                ])
        else:
            writer.writerow(['concatenated_bitstrings', 'num_included', 'num_excluded', 'duration', 'tries'])
            for bitstrings, metadata in zip(X, y):
                writer.writerow([
                    ','.join(bitstrings),
                    int(metadata[0]),  # num_included (r)
                    int(metadata[1]),  # num_excluded (s)
                    metadata[2],       # duration
                    int(metadata[3])   # tries
                ])

def run(params):
    """Main function to generate dataset based on provided parameters."""
    try:
        output_dir = params.get("outputDirectory", "data")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {params['samples']} samples in {'classification' if params.get('classification') else 'series'} mode...")
        start_time = time.time()
        
        X, y = generate_dataset_parallel(params, params['samples'])
        X_processed = process_cascade_bitstrings(X, params.get("remove_header_bits", False))
        
        output_file = os.path.join(
            output_dir,
            f"{'classification' if params.get('classification') else 'series'}-data-{int(time.time_ns())}.csv"
        )
        
        save_to_csv(X_processed, y, output_file, params.get("classification", False))
        
        end_time = time.time()
        
        return {
            "message": (
                f"Successfully generated {params['samples']} samples in "
                f"{'classification' if params.get('classification') else 'series'} mode and "
                f"saved to {output_file}. "
                f"Time taken: {end_time - start_time:.2f} seconds"
            )
        }
    
    except Exception as e:
        return {
            "message": f"Failed to generate dataset: {str(e)}"
        }