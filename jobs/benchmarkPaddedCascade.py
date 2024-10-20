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
        lst = []
        acc = a[0]
        while acc <= a[1]:
            lst.append(acc)
            acc = acc + a[2]
        return lst
    else:
        return [a]


def measure_one_filter_cascade(r, s, rmax, smax, fprs, k):
    incl = cascadeUtils.generate_id_set(r)
    excl = cascadeUtils.generate_id_set(s)
    start = time.time()
    cascade = cascadeUtils.create_padded_cascade(incl, excl, rmax, smax, fprs=fprs, k=k)
    dur = time.time() - start
    return (dur, cascade.size_in_bits())


def measurement(r, s, rmax, smax, fprs, k, samples):
    res = []
    for _ in range(samples):
        res.append(measure_one_filter_cascade(r, s, rmax, smax, fprs, k))
    dur = statistics.median(map(lambda a: a[0], res))
    size = statistics.median(map(lambda a: a[1], res))
    return (r, s, rmax, smax, fprs[0], fprs[1], k, dur, size)


def run(params):
    r = prep_range(params["r"])
    rmax = prep_range(params["rmax"])
    s = prep_range(params["s"])
    smax = prep_range(params["smax"])
    fpra = prep_range(params["fpra"])
    fprb = prep_range(params["fprb"])
    k = prep_range(params["k"])
    samples = params["samples"]

    n_points = len(r) * len(rmax) * len(s) * len(smax) * len(fpra) * len(fprb) * len(k)
    points = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_data = {}
        j = 0
        for ri in r:
            for si in s:
                for rmaxi in rmax:
                    for smaxi in smax:
                        for fprai in fpra:
                            for fprbi in fprb:
                                for ki in k:
                                    future_to_data[
                                        executor.submit(
                                            measurement,
                                            ri,
                                            si,
                                            rmaxi,
                                            smaxi,
                                            [fprai, fprbi],
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

    fname = os.path.join("out", f"benchmarkPaddedCascade-{time.time()}.csv")
    with open(fname, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=";", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        writer.writerow(
            ["r", "s", "rmax", "smax", "fpra", "fprb", "k", "duration", "bitsize"]
        )
        for row in points:
            writer.writerow(row)

    return {"message": "Measurement complete", "datafile": fname}
