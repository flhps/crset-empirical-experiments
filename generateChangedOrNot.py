import random
import numpy as np
import concurrent.futures
import pickle
import time
import cascadeUtils


def rnd_data_point(maxrevoked, maxvalid, fprs=None):
    n_included = random.randint(0, maxrevoked - 1)
    n_excluded = random.randint(1, maxvalid)
    # print("rnd data point ", n_included, "inc and ", n_excluded)
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
            cascadeUtils.vectorize_cascade(cascade)
            + cascadeUtils.vectorize_cascade(same),
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
        cascadeUtils.vectorize_cascade(cascade)
        + cascadeUtils.vectorize_cascade(different),
        [-1],
    )


def generate_data(maxrevoked, maxvalid, n_samples=100_000, fprs=None):
    X = np.empty([n_samples, 2 * cascadeUtils.vectorized_cascade_size()])
    y = np.empty([n_samples, 1])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_data = {
            executor.submit(rnd_data_point, maxrevoked, maxvalid, fprs): i
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
    d = generate_data(1000, 10000, 20000)
    print("Done generating data.")
    fname = f"data/changedOrNot-{time.time_ns()}.pkl"
    with open(fname, "wb") as outp:
        pickle.dump(d, outp, pickle.HIGHEST_PROTOCOL)
    print("done")
