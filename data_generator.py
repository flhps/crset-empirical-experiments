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
    fprs=None,
):
    start_time = time.time()
    
    revoked = cascadeUtils.generate_id_set(num_included)
    valid = cascadeUtils.generate_id_set(num_excluded)

    cascade = cascadeUtils.create_padded_cascade(
        revokedids=revoked,
        validids=valid,
        revokedmax=maxinc,
        validmax=maxexc,
        fprs=fprs,
        multi_process=True,
    )
    
    duration = time.time() - start_time
    return get_cascade_bitstrings(cascade), duration

def rnd_data_point(maxinc, maxexc, fprs=None, job_type="series"):
    start_time = time.time()
    if job_type == "series":
        n_included = random.randint(1, maxinc)
        n_excluded = random.randint(1, maxexc)
        bitstrings, duration = generate_data_series(n_included, n_excluded, maxinc, maxexc, fprs)
        return bitstrings, [n_included, n_excluded, duration]
    elif job_type == "classification":
        n_included = random.randint(0, maxinc - 1)
        n_excluded = random.randint(1, maxexc)
        revoked = cascadeUtils.generate_id_set(n_included)
        valid = cascadeUtils.generate_id_set(n_excluded)
        cascade = cascadeUtils.create_padded_cascade(
            revoked, valid, maxinc, maxexc, fprs
        )
        if bool(random.getrandbits(1)):
            same = cascadeUtils.create_padded_cascade(
                revoked, valid, maxinc, maxexc, fprs
            )
            duration = time.time() - start_time
            return (
                get_cascade_bitstrings(cascade),
                get_cascade_bitstrings(same),
                [1, duration],
            )
        delta = random.sample(valid, random.randint(1, min(maxinc - n_included, len(valid))))
        revoked2 = revoked + delta
        valid2 = [x for x in valid if x not in delta]
        different = cascadeUtils.create_padded_cascade(
            revoked2,
            valid2,
            maxinc,
            maxexc,
            fprs,
        )
        duration = time.time() - start_time
        return (
            get_cascade_bitstrings(cascade),
            get_cascade_bitstrings(different),
            [-1, duration],
        )

def generate_data(maxinc, maxexc, n_samples, fprs=None, job_type="series"):
    if job_type == "series":
        X = []
        y = np.empty([n_samples, 3])  # Added one more column for duration
    elif job_type == "classification":
        X1, X2 = [], []
        y = np.empty([n_samples, 2])  # Added one more column for duration
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        future_to_data = {
            executor.submit(rnd_data_point, maxinc, maxexc, fprs, job_type): i
            for i in range(n_samples)
        }
        
        with tqdm(total=n_samples, desc="Generating data points") as pbar:
            for future in concurrent.futures.as_completed(future_to_data):
                i = future_to_data[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (i, exc))
                else:
                    if job_type == "series":
                        X.append(data[0])
                        y[i, :] = data[1]
                    elif job_type == "classification":
                        X1.append(data[0])
                        X2.append(data[1])
                        y[i, :] = data[2]
                    pbar.update(1)
                    
                    if i % 100 == 0:
                        pbar.set_postfix({"Filters": len(data[0])})
    
    return [X, y] if job_type == "series" else [X1, X2, y]

def int_to_bits(integer):
    return format(integer, '08b')

def find_max_dimensions(data, job_type="series"):
    """
    Find the maximum number of cascades and the maximum length of cascades across all samples.
    """
    max_cascades = 0
    max_length = 0
    
    if job_type == "series":
        X, _ = data
        for sample in X:
            max_cascades = max(max_cascades, len(sample))
            for cascade in sample:
                max_length = max(max_length, len(cascade))
    elif job_type == "classification":
        X1, X2, _ = data
        for sample1, sample2 in zip(X1, X2):
            max_cascades = max(max_cascades, len(sample1), len(sample2))
            for cascade in sample1 + sample2:
                max_length = max(max_length, len(cascade))
    
    return max_cascades, max_length

def standardize_dataset(data, max_cascades, max_length, job_type="series", batch_size=1000):
    """
    Generator function to standardize the dataset in batches.
    """
    if job_type == "series":
        X, y = data
    elif job_type == "classification":
        X1, X2, y = data
        X = X1 + X2  # Combine X1 and X2 for processing

    total_samples = len(X)
    for i in range(0, total_samples, batch_size):
        batch = X[i:i+batch_size]
        standardized_batch = []
        for sample in batch:
            standardized_sample = []
            for j in range(max_cascades):
                if j < len(sample):
                    # Convert bytes to binary string, pad or truncate
                    binary_str = ''.join(format(byte, '08b') for byte in sample[j])
                    cascade = binary_str.ljust(max_length, '0')[:max_length]
                else:
                    cascade = '0' * max_length
                standardized_sample.append(cascade)
            standardized_batch.append(standardized_sample)
        yield np.array(standardized_batch)

def save_to_csv_with_padding(data, filename, job_type="series", padding=False, batch_size=1000):
    """
    Save the data to a CSV file, optionally applying padding.
    """
    if padding:
        max_cascades, max_length = find_max_dimensions(data, job_type)
        standardized_generator = standardize_dataset(data, max_cascades, max_length, job_type, batch_size)
    else:
        if job_type == "series":
            standardized_generator = (data[0][i:i+batch_size] for i in range(0, len(data[0]), batch_size))
        elif job_type == "classification":
            standardized_generator = ((data[0][i:i+batch_size], data[1][i:i+batch_size]) for i in range(0, len(data[0]), batch_size))
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE, escapechar='\\')
        
        if job_type == "series":
            writer.writerow(['concatenated_bitstrings', 'num_included', 'num_excluded', 'duration'])
            X, y = data
            sample_index = 0
            for batch in tqdm(standardized_generator, desc="Processing batches"):
                for sample in batch:
                    if sample_index >= len(X):
                        break
                    if padding:
                        concatenated_bitstring = ','.join(sample)
                    else:
                        concatenated_bitstring = ','.join(''.join(format(byte, '08b') for byte in cascade) for cascade in X[sample_index])
                    num_included, num_excluded, duration = y[sample_index]
                    writer.writerow([concatenated_bitstring, num_included, num_excluded, duration])
                    sample_index += 1
        
        elif job_type == "classification":
            writer.writerow(['concatenated_bitstrings_X1', 'concatenated_bitstrings_X2', 'label', 'duration'])
            X1, X2, y = data
            sample_index = 0
            for batch in tqdm(standardized_generator, desc="Processing batches"):
                if padding:
                    batch1, batch2 = batch[:len(batch)//2], batch[len(batch)//2:]
                else:
                    batch1, batch2 = batch
                for sample1, sample2 in zip(batch1, batch2):
                    if sample_index >= len(X1):
                        break
                    if padding:
                        concatenated_bitstring1 = ','.join(sample1)
                        concatenated_bitstring2 = ','.join(sample2)
                    else:
                        concatenated_bitstring1 = ','.join(''.join(format(byte, '08b') for byte in cascade) for cascade in X1[sample_index])
                        concatenated_bitstring2 = ','.join(''.join(format(byte, '08b') for byte in cascade) for cascade in X2[sample_index])
                    label, duration = y[sample_index]
                    writer.writerow([concatenated_bitstring1, concatenated_bitstring2, label, duration])
                    sample_index += 1

def main():    
    job_type = CONFIG['job_type']
    if job_type not in ["series", "classification"]:
        raise ValueError("Invalid job_type in config. Must be 'series' or 'classification'.")

    print(f"Generating {job_type} data...")
    start_time = time.time()
    d = generate_data(
        CONFIG['maxinc'], 
        CONFIG['maxexc'], 
        CONFIG['n_samples'], 
        fprs=CONFIG['fprs'], 
        job_type=job_type
    )
    end_time = time.time()

    print(f"Done generating data. Time taken: {end_time - start_time:.2f} seconds")
    
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    fname = os.path.join(CONFIG['output_directory'], f"training-data-{job_type}-{time.time_ns()}.csv")
    print(f"Saving data to {fname}...")
    
    save_to_csv_with_padding(d, fname, job_type=job_type, padding=CONFIG['padding'], batch_size=1000)
    print("Data saved successfully.")

if __name__ == "__main__":
    main()
