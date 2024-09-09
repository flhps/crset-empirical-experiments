import random
import cascadeUtils
import numpy as np
import concurrent.futures
import pickle
import time
from tqdm import tqdm

def rnd_data_point(maxinc, maxexc, fprs=None):
    n_included = random.randint(1, maxinc)
    n_excluded = random.randint(0, maxexc)
    revoked = cascadeUtils.generate_id_set(n_included)
    valid = cascadeUtils.generate_id_set(n_excluded)
    cascade = cascadeUtils.create_padded_cascade(
        revoked, valid, maxinc, maxexc, fprs
    )
    return cascadeUtils.vectorize_cascade(cascade), [n_included, n_excluded]

def generate_data(maxinc, maxexc, n_samples=100_000, fprs=None):
    X = np.empty([n_samples, cascadeUtils.vectorized_cascade_size()])
    y = np.empty([n_samples, 2])
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(rnd_data_point, maxinc, maxexc, fprs) for _ in range(n_samples)]
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=n_samples, desc="Generating data")):
            try:
                data = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (i, exc))
            else:
                X[i, :] = data[0]
                y[i, :] = data[1]
    return [X, y]

if __name__ == "__main__":
    print("Generating data...")
    d = generate_data(1000, 10000, 10000)
    print("Done generating data.")
    fname = f"data/independentCascades-{time.time_ns()}.pkl"
    with open(fname, "wb") as outp:
        pickle.dump(d, outp, pickle.HIGHEST_PROTOCOL)
    print("done")