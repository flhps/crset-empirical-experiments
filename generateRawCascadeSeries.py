import random
import uuid
from paddedCascade import PaddedCascade
import numpy as np
import concurrent.futures
import pickle
import time
from tqdm import tqdm

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
    
    # Return the raw bitstrings
    bitstrings = []
    for bloom_filter in test_cascade.filters:
        bitstring = bloom_filter.save_bytes()
        bitstrings.append(bitstring)
    
    return bitstrings

def rnd_data_point(maxinc, maxexc, fprs=None):
    n_included = random.randint(1, maxinc)
    n_excluded = random.randint(1, maxexc)
    return generate_data_series(n_included, n_excluded, maxinc, maxexc, fprs), [
        n_included,
        n_excluded,
    ]

def generate_data(maxinc, maxexc, n_samples=100_000, fprs=None):
    X = []  # This will be a list of lists of bitstrings
    y = np.empty([n_samples, 2])
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        future_to_data = {
            executor.submit(rnd_data_point, maxinc, maxexc, fprs): i
            for i in range(n_samples)
        }
        
        # Create a tqdm progress bar
        with tqdm(total=n_samples, desc="Generating data points") as pbar:
            for future in concurrent.futures.as_completed(future_to_data):
                i = future_to_data[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (i, exc))
                else:
                    X.append(data[0])  # Append the list of bitstrings
                    y[i, :] = data[1]
                    pbar.update(1)  # Update the progress bar
                
                if i % 100 == 0:
                    pbar.set_postfix({"Filters": len(data[0])})
    
    return [X, y]

if __name__ == "__main__":
    print("Generating data...")
    start_time = time.time()
    d = generate_data(1000, 10000, 10000)
    end_time = time.time()
    print(f"Done generating data. Time taken: {end_time - start_time:.2f} seconds")
    
    fname = f"data/training-data-bitstrings-{time.time_ns()}.pkl"
    print(f"Saving data to {fname}...")
    with open(fname, "wb") as outp:
        pickle.dump(d, outp, pickle.HIGHEST_PROTOCOL)
    print("Data saved successfully.")