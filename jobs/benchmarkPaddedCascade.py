import cascadeUtils
import concurrent.futures
import collections.abc
from tqdm import tqdm
import time
import statistics
import os
import csv


def prep_range(a):
    if isinstance(a, collections.abc.Sequence):
        return range(a[0], a[1], a[2])
    else:
        return [a]


def measure_one_filter_cascade(n, nmax, fprs, k):
    incl = cascadeUtils.generate_id_set(n)
    excl = cascadeUtils.generate_id_set(n)
    start = time.time()
    cascade = cascadeUtils.create_padded_cascade(incl, excl, nmax, nmax, fprs, k)
    dur = time.time() - start
    return (dur, cascade.size_in_bits())


def measurement(n, nmax, fprs, k, samples):
    res = []
    for _ in range(samples):
        res.append(measure_one_filter_cascade(n, nmax, fprs, k))
    dur = statistics.median(map(lambda a: a[0], res))
    size = statistics.median(map(lambda a: a[1], res))
    return (n, nmax, fprs[0], fprs[1], k, dur, size)


def run(params):
    n = prep_range(params["n"])
    nmax = prep_range(params["nmax"])
    fpra = prep_range(params["fpra"])
    fprb = prep_range(params["fprb"])
    k = prep_range(params["k"])
    samples = params["samples"]

    n_points = len(n) * len(nmax) * len(fpra) * len(fprb) * len(k)
    points = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_data = {}
        j = 0
        for ni in n:
            for nmaxi in nmax:
                for fprai in fpra:
                    for fprbi in fprb:
                        for ki in k:
                            future_to_data[
                                executor.submit(
                                    measurement, ni, nmaxi, [fprai, fprbi], ki, samples
                                )
                            ] = j
                            j = j + 1

        with tqdm(total=n_points, desc="Generating data points") as pbar:
            for future in concurrent.futures.as_completed(future_to_data):
                i = future_to_data[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (i, exc))
                else:
                    points.append(data)
                    pbar.update(1)

    fname = os.path.join("data", f"benchmarkPaddedCascade-{time.time()}.csv")
    with open(fname, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=";", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        writer.writerow(["n", "nmax", "fpra", "fprb", "k", "duration", "bitsize"])
        for row in points:
            writer.writerow(row)

    return {"message": "Measurement complete", "datafile": fname}
