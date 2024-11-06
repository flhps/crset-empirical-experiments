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


def generate_single_cascade_datapoint(max_r, max_s, rhat, p, k, parallelize, pad_to_target_size=False):
    """Generate a single cascade datapoint with random r and s values."""
    start_time = time.time()
    
    # Generate random r and s values within specified ranges
    actual_r = random.randint(1, max_r)
    actual_s = random.randint(1, max_s)
    
    # Generate valid IDs
    valid_ids = cu.gen_ids(actual_r)
    
    # If padding is enabled, generate additional IDs up to rhat
    if pad_to_target_size and rhat > actual_r:
        padding_ids = cu.gen_ids_wo_overlap(rhat - actual_r, valid_ids)
        valid_ids.update(padding_ids)
    
    # Generate revoked IDs
    revoked_ids = cu.gen_ids_wo_overlap(actual_s, valid_ids)
    
    # Use either padded rhat or actual r based on padding mode
    target_size = rhat if pad_to_target_size else actual_r
    
    cascade, tries = cu.try_cascade(
        valid_ids,
        revoked_ids,
        target_size,
        p=p,
        k=k,
        multi_process=parallelize
    )
    
    duration = time.time() - start_time
    return get_cascade_bitstrings(cascade), [actual_r, actual_s, duration, tries]


def generate_cascade_pair_datapoint(params, identical=True):
    """Generate a pair of cascades, either identical or different."""
    start_time = time.time()
    
    # Get random r value (1 to max_r)
    max_r = params["r"]
    actual_r = random.randint(1, max_r)
    
    # Set target size based on padding mode
    target_size = params["rhat"] if params.get("pad_to_target_size", False) else actual_r
    
    s = random.randint(1, params["s"])
    p = params["p"][0]
    k = params["k"]
    
    # Generate initial valid IDs
    valid_ids = cu.gen_ids(actual_r)
    
    # Add padding if enabled
    if params.get("pad_to_target_size", False) and target_size > actual_r:
        padding_ids = cu.gen_ids_wo_overlap(target_size - actual_r, valid_ids)
        valid_ids.update(padding_ids)
    
    revoked_ids = cu.gen_ids_wo_overlap(s, valid_ids)
    
    # Generate first cascade
    cascade1, tries1 = cu.try_cascade(
        valid_ids,
        revoked_ids,
        target_size,
        p=p,
        k=k,
        multi_process=params["parallelize"]
    )
    
    if identical:
        cascade2, tries2 = cu.try_cascade(
            valid_ids,
            revoked_ids,
            target_size,
            p=p,
            k=k,
            multi_process=params["parallelize"]
        )
    else:
        valid_ids_list = list(valid_ids)
        max_delta = min(target_size - actual_r, len(valid_ids_list)) if params.get("pad_to_target_size", False) else len(valid_ids_list)
        delta = random.sample(valid_ids_list, random.randint(1, max_delta))
        
        valid_ids2 = set(x for x in valid_ids if x not in delta)
        revoked_ids2 = set(list(revoked_ids) + delta)
        
        cascade2, tries2 = cu.try_cascade(
            valid_ids2,
            revoked_ids2,
            target_size,
            p=p,
            k=k,
            multi_process=params["parallelize"]
        )
    
    duration = time.time() - start_time
    
    return (
        [get_cascade_bitstrings(cascade1), get_cascade_bitstrings(cascade2)],
        [actual_r, s, actual_r, s, duration, tries1 + tries2, int(identical)]
    )

def generate_dataset_parallel(params, n_samples):
    """Generate multiple cascade datapoints in parallel."""
    if params.get("pairs_mode", False):
        return generate_pairs_dataset(params, n_samples)
    else:
        return generate_single_dataset(params, n_samples)

def generate_single_dataset(params, n_samples):
    """Generate dataset with single cascades using random r and s values."""
    X = []
    y = np.empty([n_samples, 4])  # [r, s, duration, tries]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_data = {
            executor.submit(
                generate_single_cascade_datapoint, 
                params["r"], 
                params["s"], 
                params["rhat"],
                params["p"][0], 
                params["k"],
                params.get("parallelize", False),
                params.get("pad_to_target_size", False)
            ): i for i in range(n_samples)
        }
        
        with tqdm(total=n_samples, desc="Generating single cascades") as pbar:
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

def generate_pairs_dataset(params, n_samples):
    """Generate dataset with pairs of cascades."""
    X = []
    y = np.empty([n_samples, 7])  # [r1, s1, r2, s2, duration, total_tries, identical]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_data = {}
        for i in range(n_samples):
            identical = i < n_samples // 2
            future_to_data[executor.submit(
                generate_cascade_pair_datapoint,
                params,
                identical
            )] = i
        
        with tqdm(total=n_samples, desc="Generating cascade pairs") as pbar:
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

def process_cascade_bitstrings(X, remove_header_bits=False, pad_output_format=False):
    """Process cascade bitstrings with optional header bit removal and format padding."""
    processed_X = []
    
    if not pad_output_format:
        # Original processing without padding
        for sample in X:
            if isinstance(sample[0], list):  # Pairs mode
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
            else:  # Single mode
                processed_cascades = []
                for cascade in sample:
                    bitstring = ''.join(format(byte, '08b') for byte in cascade)
                    if remove_header_bits:
                        bitstring = bitstring[64:]
                    processed_cascades.append(bitstring)
                processed_X.append(processed_cascades)
        return processed_X
    
    # If filter padding is enabled, proceed with padding logic
    max_lengths = {0: []}
    max_filters = 0
    
    # Find max number of filters and initialize max_lengths
    for sample in X:
        if isinstance(sample[0], list):
            max_lengths[1] = []
            for cascade_set in sample:
                max_filters = max(max_filters, len(cascade_set))
        else:
            max_filters = max(max_filters, len(sample))
    
    max_lengths[0] = [0] * max_filters
    if len(max_lengths) > 1:
        max_lengths[1] = [0] * max_filters
    
    # Find maximum lengths for each position
    for sample in X:
        if isinstance(sample[0], list):  # Pairs mode
            for cascade_idx, cascade_set in enumerate(sample):
                for filter_idx, cascade in enumerate(cascade_set):
                    bitstring = ''.join(format(byte, '08b') for byte in cascade)
                    if remove_header_bits:
                        bitstring = bitstring[64:]
                    max_lengths[cascade_idx][filter_idx] = max(
                        max_lengths[cascade_idx][filter_idx],
                        len(bitstring)
                    )
        else:  # Single mode
            for filter_idx, cascade in enumerate(sample):
                bitstring = ''.join(format(byte, '08b') for byte in cascade)
                if remove_header_bits:
                    bitstring = bitstring[64:]
                max_lengths[0][filter_idx] = max(
                    max_lengths[0][filter_idx],
                    len(bitstring)
                )
    
    # Process and pad bitstrings
    for sample in X:
        if isinstance(sample[0], list):  # Pairs mode
            processed_sample = []
            for cascade_idx, cascade_set in enumerate(sample):
                processed_cascades = []
                # Process existing filters
                for filter_idx, cascade in enumerate(cascade_set):
                    bitstring = ''.join(format(byte, '08b') for byte in cascade)
                    if remove_header_bits:
                        bitstring = bitstring[64:]
                    padded_bitstring = bitstring.ljust(max_lengths[cascade_idx][filter_idx], '0')
                    processed_cascades.append(padded_bitstring)
                
                # Add missing filters as all zeros
                while len(processed_cascades) < max_filters:
                    missing_idx = len(processed_cascades)
                    processed_cascades.append('0' * max_lengths[cascade_idx][missing_idx])
                
                processed_sample.append(processed_cascades)
            processed_X.append(processed_sample)
        
        else:  # Single mode
            processed_cascades = []
            # Process existing filters
            for filter_idx, cascade in enumerate(sample):
                bitstring = ''.join(format(byte, '08b') for byte in cascade)
                if remove_header_bits:
                    bitstring = bitstring[64:]
                padded_bitstring = bitstring.ljust(max_lengths[0][filter_idx], '0')
                processed_cascades.append(padded_bitstring)
            
            # Add missing filters as all zeros
            while len(processed_cascades) < max_filters:
                missing_idx = len(processed_cascades)
                processed_cascades.append('0' * max_lengths[0][missing_idx])
            
            processed_X.append(processed_cascades)
    
    return processed_X

def save_to_csv(X, y, filename, pairs_mode=False):
    """Save generated data to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE, escapechar='\\')
        
        if pairs_mode:
            writer.writerow([
                'cascade1_bitstrings', 'cascade2_bitstrings',
                'r1', 's1', 'r2', 's2', 'duration', 'total_tries', 'identical'
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
                    int(metadata[6])   # identical
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
        base_dir = params.get("outputDirectory", "data")
        os.makedirs(base_dir, exist_ok=True)
        
        mode_dir = "pairs" if params.get("pairs_mode", False) else "single"
        output_dir = os.path.join(base_dir, mode_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {params['samples']} samples in {mode_dir} mode...")
        print(f"Certificate padding is {'enabled' if params.get('pad_to_target_size', False) else 'disabled'}")
        
        start_time = time.time()
        
        X, y = generate_dataset_parallel(params, params['samples'])
        X_processed = process_cascade_bitstrings(
            X, 
            remove_header_bits=params.get("remove_header_bits", False),
            pad_output_format=params.get("pad_output_format", False)
        )
        
        output_file = os.path.join(
            output_dir,
            f"{mode_dir}-data-{int(time.time_ns())}.csv"
        )
        
        save_to_csv(X_processed, y, output_file, params.get("pairs_mode", False))
        
        end_time = time.time()
        
        return {
            "message": (
                f"Successfully generated {params['samples']} samples in "
                f"{mode_dir} mode and saved to {output_file}. "
                f"Time taken: {end_time - start_time:.2f} seconds"
            )
        }
    
    except Exception as e:
        return {
            "message": f"Failed to generate dataset: {str(e)}"
        }