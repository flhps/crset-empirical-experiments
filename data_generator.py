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
    deltainc=100,
    deltaexc=1000,
    negdeltainc=False,
):
    test_cascade = None
    tries = 0
    while not test_cascade:
        try:
            revoked = [uuid.uuid4() for _ in range(num_included)]
            valid = [uuid.uuid4() for _ in range(num_excluded)]
            test_cascade = PaddedCascade(revoked, valid, maxinc, maxexc, fprs)
        except Exception:
            tries += 1
            if tries > 3:
                break
    
    if not test_cascade:
        raise Exception(
            f"Cascade construction failed repeatedly for {num_included} inclusions and {num_excluded} exclusions with {fprs} fpr targets"
        )
    
    return get_cascade_bitstrings(test_cascade)

def rnd_data_point(maxinc, maxexc, fprs=None, job_type="series"):
    if job_type == "series":
        n_included = random.randint(1, maxinc)
        n_excluded = random.randint(1, maxexc)
        return generate_data_series(n_included, n_excluded, maxinc, maxexc, fprs), [
            n_included,
            n_excluded,
        ]
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
            return (
                get_cascade_bitstrings(cascade),
                get_cascade_bitstrings(same),
                [1],
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
        return (
            get_cascade_bitstrings(cascade),
            get_cascade_bitstrings(different),
            [-1],
        )

def generate_data(maxinc, maxexc, n_samples, fprs=None, job_type="series"):
    if job_type == "series":
        X = []
        y = np.empty([n_samples, 2])
    elif job_type == "classification":
        X1, X2 = [], []
        y = np.empty([n_samples, 1])
    
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

def save_to_csv(data, filename, job_type="series"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE, escapechar='\\')
        if job_type == "series":
            X, y = data
            for bitstrings, labels in zip(X, y):
                x_row = ','.join(''.join(int_to_bits(b) for b in bitstring) for bitstring in bitstrings)
                y_row = f"{int(labels[0])},{int(labels[1])}"
                writer.writerow([x_row, y_row])
        elif job_type == "classification":
            X1, X2, y = data
            for bitstrings1, bitstrings2, label in zip(X1, X2, y):
                x1_row = ','.join(''.join(int_to_bits(b) for b in bitstring) for bitstring in bitstrings1)
                x2_row = ','.join(''.join(int_to_bits(b) for b in bitstring) for bitstring in bitstrings2)
                y_row = str(int(label[0]))
                writer.writerow([x1_row, x2_row, y_row])

def main():
    start_time = time.time()
    
    job_type = CONFIG['job_type']
    if job_type not in ["series", "classification"]:
        raise ValueError("Invalid job_type in config. Must be 'series' or 'classification'.")

    print(f"Generating {job_type} data...")
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
    save_to_csv(d, fname, job_type=job_type)
    print("Data saved successfully.")

if __name__ == "__main__":
    main()