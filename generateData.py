import random
import numpy
import uuid
from amqStatusList import FilterCascade
import numpy as np
import concurrent.futures


def generate_data_point(num_included, num_excluded, fprs=None):
    test_cascade = None
    tries = 0
    while not test_cascade:
        try:
            revoked = []
            for i in range(num_included):
                revoked.append(uuid.uuid4())
            valid = []
            for i in range(num_excluded):
                valid.append(uuid.uuid4())
            test_cascade = FilterCascade(revoked, valid, fprs)
        except:
            tries = tries + 1
            if tries > 3:
                break
    if not test_cascade:
        raise Exception(
            f"Cascade construction failed repeatedly for {num_included} inclusions and {num_excluded} exclusions with {fprs} fpr targets")
    return np.array([test_cascade.size_in_bits() / 8.0, len(test_cascade.filters)])


def rnd_data_point():
    n_included = random.randint(1, 10_000)
    n_excluded = random.randint(1, 10_000)
    return [generate_data_point(n_included, n_excluded), [n_included, n_excluded]]


def generate_data(n_samples=100_000):
    X = numpy.empty([n_samples, 2])
    y = numpy.empty([n_samples, 2])
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        future_to_data = {executor.submit(rnd_data_point): i for i in range(n_samples)}
        for future in concurrent.futures.as_completed(future_to_data):
            i = future_to_data[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (i, exc))
            else:
                if i % 10 == 0:
                    print(f"Data point {i}")
                X[i, :] = data[0]
                y[i, :] = data[1]
    return [X, y]
