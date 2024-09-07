import random
import uuid
from paddedCascade import PaddedCascade
import numpy as np
import concurrent.futures
import pickle
import time


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
            revoked = []
            for _ in range(num_included):
                revoked.append(uuid.uuid4())
            valid = []
            for _ in range(num_excluded):
                valid.append(uuid.uuid4())
            test_cascade = PaddedCascade(revoked, valid, maxinc, maxexc, fprs)
        except Exception:
            tries = tries + 1
            if tries > 3:
                break
    if not test_cascade:
        raise Exception(
            f"Cascade construction failed repeatedly for {num_included} inclusions and {num_excluded} exclusions with {fprs} fpr targets"
        )
    return [
        float(test_cascade.size_in_bits()),
        float(len(test_cascade.filters)),
        float(test_cascade.filters[0].size_in_bits),
        float(test_cascade.count_set_bits()),
        test_cascade.calculate_entropy(),
    ]


def rnd_data_point(macinc, maxexc, fprs=None):
    n_included = random.randint(1, macinc)
    n_excluded = random.randint(1, maxexc)
    return generate_data_point(n_included, n_excluded, macinc, maxexc, fprs), [
        n_included,
        n_excluded,
    ]


def generate_data(maxinc, macexc, n_samples=100_000, fprs=None):
    X = np.empty([n_samples, 5])
    y = np.empty([n_samples, 2])
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        future_to_data = {
            executor.submit(rnd_data_point, maxinc, macexc, fprs): i
            for i in range(n_samples)
        }
        for future in concurrent.futures.as_completed(future_to_data):
            i = future_to_data[future]
            try:
                data = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (i, exc))
            else:
                if i % 100 == 0:
                    print(f"Data point {i}")
                    print(data[0])
                X[i, :] = data[0]
                y[i, :] = data[1]
    return [X, y]


if __name__ == "__main__":
    print("Generating data...")
    d = generate_data(1000, 10000, 10000)
    print("Done generating data.")
    fname = f"data/training-data-{time.time_ns()}.pkl"
    with open(fname, "wb") as outp:
        pickle.dump(d, outp, pickle.HIGHEST_PROTOCOL)
    print("done")
