import random
import numpy as np
import cascadeUtils


def rnd_data_point(maxinc, maxexc, fprs=None):
    n_included = random.randint(1, maxinc)
    n_excluded = random.randint(1, maxexc)
    revoked = cascadeUtils.generate_id_set(n_included)
    valid = cascadeUtils.generate_id_set(n_excluded)
    test_cascade = cascadeUtils.create_padded_cascade(revoked,valid, maxinc, maxexc, fprs)
    return test_cascade


def generate_data(maxinc, maxexc, n_samples=100_000, fprs=None):
    X = np.empty([n_samples, 5])
    y = np.empty([n_samples, 2])
    for _ in range(n_samples):
        rnd_data_point(maxinc, maxexc, fprs)


if __name__ == "__main__":
    print("Generating data...")
    generate_data(1000, 10000, 100)
    print("Done generating data.")
