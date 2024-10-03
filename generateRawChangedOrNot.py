import random
import numpy as np
import concurrent.futures
import pickle
import time
import cascadeUtils
from tqdm import tqdm

def get_cascade_bitstrings(cascade):
    bitstrings = []
    for bloom_filter in cascade.filters:
        bitstring = bloom_filter.save_bytes()
        bitstrings.append(bitstring)
    return bitstrings

def rnd_data_point(maxrevoked, maxvalid, fprs=None):
    n_included = random.randint(0, maxrevoked - 1)
    n_excluded = random.randint(1, maxvalid)
    revoked = cascadeUtils.generate_id_set(n_included)
    valid = cascadeUtils.generate_id_set(n_excluded)
    cascade = cascadeUtils.create_padded_cascade(
        revoked, valid, maxrevoked, maxvalid, fprs
    )
    if bool(random.getrandbits(1)):
        same = cascadeUtils.create_padded_cascade(
            revoked, valid, maxrevoked, maxvalid, fprs
        )
        return (
            get_cascade_bitstrings(cascade),
            get_cascade_bitstrings(same),
            [1],
        )
    # remove some ids from the valid set and add them to the revoked set
    delta = random.sample(valid, random.randint(1, min(maxrevoked - n_included, len(valid))))
    revoked2 = revoked + delta
    valid2 = [x for x in valid if x not in delta]
    different = cascadeUtils.create_padded_cascade(
        revoked2,
        valid2,
        maxrevoked,
        maxvalid,
        fprs,
    )
    return (
        get_cascade_bitstrings(cascade),
        get_cascade_bitstrings(different),
        [-1],
    )

def generate_data(maxrevoked, maxvalid, n_samples=100_000, fprs=None):
    X1 = []  # first cascade
    X2 = []  # second cascade
    y = np.empty([n_samples, 1])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_data = {
            executor.submit(rnd_data_point, maxrevoked, maxvalid, fprs): i
            for i in range(n_samples)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_data), total=n_samples, desc="Generating data"):
            i = future_to_data[future]
            try:
                data = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (i, exc))
            else:
                if i % 100 == 0:
                    print(f"Data point {i}")
                    print(f"Number of bitstrings in first cascade: {len(data[0])}")
                    print(f"Number of bitstrings in second cascade: {len(data[1])}")
                X1.append(data[0])
                X2.append(data[1])
                y[i, :] = data[2]
    return [X1, X2, y]

if __name__ == "__main__":
    print("Generating data...")
    d = generate_data(1000, 10000, 20000)
    print("Done generating data.")
    fname = f"data/changedOrNot-bitstrings-{time.time_ns()}.pkl"
    with open(fname, "wb") as outp:
        pickle.dump(d, outp, pickle.HIGHEST_PROTOCOL)
    print("done")