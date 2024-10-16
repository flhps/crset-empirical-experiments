import random
import uuid
import numpy as np
import concurrent.futures
import csv
import time
import os
import yaml
from tqdm import tqdm
import cascadeUtils
from paddedCascade import PaddedCascade

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def get_cascade_bitstrings(cascade):
    bitstrings = []
    for bloom_filter in cascade.filters:
        bitstring = bloom_filter.save_bytes()
        bitstrings.append(bitstring)
    return bitstrings

def generate_data_series(
    num_included: int,
    num_excluded: int,
    maxinc,
    maxexc,
    fprs,
    margin
):
    test_cascade = None
    tries = 0
    start_time = time.time()
    while not test_cascade:
        try:
            revoked = [uuid.uuid4() for _ in range(num_included)]
            valid = [uuid.uuid4() for _ in range(num_excluded)]
            test_cascade = PaddedCascade(revoked, valid, maxinc, maxexc, fprs, margin)
        except Exception:
            tries += 1
            if tries > 5: # TODO: Change back to 3
                break
    
    if not test_cascade:
        raise Exception(
            f"Cascade construction failed repeatedly for {num_included} inclusions and {num_excluded} exclusions with {fprs} fpr targets"
        )
    
    duration = time.time() - start_time
    return get_cascade_bitstrings(test_cascade), duration, tries

def rnd_data_point(maxinc, maxexc, fprs, margin):
    n_included = random.randint(1, maxinc)
    n_excluded = random.randint(1, maxexc)
    bitstrings, duration, tries = generate_data_series(n_included, n_excluded, maxinc, maxexc, fprs, margin)
    return bitstrings, [n_included, n_excluded, duration, tries]

def generate_benchmark_data(n_samples, maxinc, maxexc, fprs, margin):
    X = []
    y = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        future_to_data = {
            executor.submit(rnd_data_point, maxinc, maxexc, fprs, margin): i
            for i in range(n_samples)
        }
        
        for future in concurrent.futures.as_completed(future_to_data):
            try:
                data = future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
            else:
                X.append(data[0])
                y.append(data[1])
    
    return X, y

def save_to_csv(all_data, filename, batch_size=1000):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(['maxinc', 'maxexc', 'margin', 'fpr', 'num_included', 'num_excluded', 'duration', 'tries', 'total_bitstring_length'])
        
        for data, maxinc, maxexc, margin, fpr in tqdm(all_data, desc="Saving to CSV"):
            X, y = data
            for sample_X, sample_y in zip(X, y):
                total_bitstring_length = sum(len(cascade) * 8 for cascade in sample_X)  # Each byte is 8 bits 
                writer.writerow([maxinc, maxexc, margin, fpr] + sample_y + [total_bitstring_length])

def main():
    start_time = time.time()
    
    # Parameter ranges
    maxinc_maxexc_pairs = [(100_000, 1000)] # [(1000, 100_000), (100_000, 100_000), (100_000, 1000)]
    margins = np.arange(1.0, 1.26, 0.05).round(2)  # From 1.0 to 1.25 with steps of 0.01 (25 steps)
    fprs = np.arange(0.10, 1.01, 0.01).round(2)  # From 0.1% to 10% with steps of 0.1% (100 steps)

    samples = CONFIG['n_samples']
    
    print("Generating benchmark data...")
    all_data = []

    total_iterations = len(maxinc_maxexc_pairs) * len(margins) * len(fprs)
    with tqdm(total=total_iterations, desc="Generating data") as pbar:
        for maxinc, maxexc in maxinc_maxexc_pairs:
            for margin in margins:
                for fpr in fprs:
                    d = generate_benchmark_data(
                        n_samples=samples,
                        maxinc=maxinc,
                        maxexc=maxexc,
                        fprs=[fpr],
                        margin=margin
                    )
                    all_data.append((d, maxinc, maxexc, margin, fpr))
                    pbar.update(1)

    end_time = time.time()
    print(f"Done generating data. Time taken: {end_time - start_time:.2f} seconds")
    
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    fname = os.path.join(CONFIG['output_directory'], f"benchmarking-data-{time.time_ns()}.csv")
    print(f"Saving data to {fname}...")
    
    save_to_csv(all_data, fname, batch_size=1000)
    print("Data saved successfully.")

if __name__ == "__main__":
    main()