from math import ceil
import src.cascade.cascadeUtils as cu
import concurrent.futures
import collections.abc
from tqdm import tqdm
import time
import statistics
import os
import csv
import random


def prep_range(a):
    if isinstance(a, collections.abc.Sequence):
        lst = []
        acc = a[0]
        while acc <= a[1]:
            lst.append(acc)
            if a[2] == -1:
                acc = acc * 2
            else:
                acc = acc + a[2]
        return lst
    else:
        return [a]


def measure_one_filter_cascade(r, s, rhat, p, k):
    validIds = cu.gen_ids(r)
    revokedIds = cu.gen_ids_wo_overlap(s, validIds)
    start = time.time()
    cascade = cu.try_cascade(validIds, revokedIds, rhat, p=p, k=k)
    dur = time.time() - start
    validTests = int(ceil(r * 1000.0 / s))
    validList = list(validIds)
    revokedList = list(revokedIds)
    start = time.time()
    for _ in range(validTests):
        assert random.choice(validList) in cascade[0]
    for _ in range(1000 - validTests):
        assert random.choice(revokedList) not in cascade[0]
    dur1k = time.time() - start
    return (dur, cascade[0].size_in_bits(), dur1k, cascade[1])


def measurement(r, s, rhat, p, k, samples):
    res = []
    for _ in range(samples):
        res.append(measure_one_filter_cascade(r, s, rhat, p, k))
    dur = statistics.median(map(lambda a: a[0], res))
    size = statistics.median(map(lambda a: a[1], res))
    dur1k = statistics.median(map(lambda a: a[2], res))
    tries = statistics.median(map(lambda a: a[3], res))
    return (r, s, rhat, p, k, dur, size, dur1k, tries)


def run(params):
    r = prep_range(params["r"])
    s = prep_range(params["s"])
    rhat = prep_range(params["rhat"])
    p = prep_range(params["p"])
    k = prep_range(params["k"])
    samples = params["samples"]

    n_points = len(r) * len(s) * len(rhat) * len(p) * len(k)
    points = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_to_data = {}
        j = 0
        for ri in r:
            for si in s:
                for rhati in rhat:
                    for pi in p:
                        for ki in k:
                            future_to_data[
                                executor.submit(
                                    measurement,
                                    ri,
                                    si,
                                    rhati,
                                    pi,
                                    ki,
                                    samples,
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

    name = params["name"]
    if name is None:
        name = ""
    else:
        name = name + "-"
    fname = os.path.join("out", f"benchStCas-{name+str(time.time())}.csv")
    with open(fname, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=";", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        writer.writerow(
            ["r", "s", "rhat", "p", "k", "duration", "bitsize", "lookup1k", "tries"]
        )
        for row in points:
            writer.writerow(row)

    return {"message": "Measurement complete", "datafile": fname}
