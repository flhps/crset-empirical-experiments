from rbloom import Bloom
import secrets
from hashlib import sha256
from pickle import dumps
import math
from collections import Counter
import concurrent.futures


def hash_func(obj):
    h = sha256(dumps(obj)).digest()
    return int.from_bytes(h[:16], "big") - 2**127


def new_bloom(size, fpr, k):
    if k is None:
        return Bloom(size, fpr, hash_func)
    return Bloom(size, fpr, hash_func, k)


def build_cascade_part(positives, size, fpr, k, salt):
    bloom = new_bloom(size, fpr, k)
    for elem in positives:
        bloom.add(str(elem) + salt)
    return bloom.save_bytes()


# making sure code runs on python versions before 3.12
def batched(lst, n):
    if n < 1:
        raise ValueError("n must be at least one")
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class FilterCascade:
    def __init__(self, positives, negatives, fprs=None, k=None, multi_process=False):
        if len(positives) > len(negatives):
            raise ValueError("Cascade rquires less positives than negatives")
        if fprs is None:
            fprs = [len(positives) * math.sqrt(0.5) / len(negatives), 0.5]
        self.filters = []
        self.salt = str(secrets.randbits(256))
        self.__build_cascade(positives, negatives, fprs, k, multi_process)

    def __build_cascade(self, positives, negatives, fprs, k, multi_process):
        inclusions = positives
        exclusions = negatives
        cons_non_improvements = 0
        while len(inclusions) > 0:
            fpr = fprs[-1]
            if len(fprs) > len(self.filters):
                fpr = fprs[len(self.filters)]
            bloom = self.__help_build_filter(inclusions, fpr, k, multi_process)
            fps = set()
            ds = str(len(self.filters)) + self.salt
            for elem in exclusions:
                if str(elem) + ds in bloom:
                    fps.add(elem)
            self.filters.append(bloom)
            if (
                len(self.filters) > 1
                and self.filters[-1].size_in_bits >= self.filters[-2].size_in_bits
            ):
                cons_non_improvements += 1
                if cons_non_improvements > 5:
                    raise Exception("Cascade cannot solve")
            else:
                cons_non_improvements = 0
            exclusions = inclusions
            inclusions = fps

    def __help_build_filter(self, positives, fpr, k, multi_process):
        new_size = len(positives)
        ds = str(len(self.filters)) + self.salt
        processes = 8
        bloom = None
        # only worth using multiple processes if every chunk is big enough
        if multi_process and len(positives) > processes * 100_000:
            positive_chunks = list(
                batched(positives, math.ceil(len(positives) / processes))
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=processes
            ) as executor:
                future_to_data = {
                    executor.submit(
                        build_cascade_part, positive_chunks[i], new_size, fpr, k, ds
                    ): i
                    for i in range(processes)
                }

                for future in concurrent.futures.as_completed(future_to_data):
                    i = future_to_data[future]
                    try:
                        data = Bloom.load_bytes(future.result(), hash_func)
                    except Exception as exc:
                        print("%r generated an exception: %s" % (i, exc))
                    else:
                        if bloom is None:
                            bloom = data
                        else:
                            bloom.update(data)
            assert bloom is not None
            return bloom
        else:
            bloom = new_bloom(new_size, fpr, k)
            for elem in positives:
                bloom.add(str(elem) + ds)
            return bloom

    def __contains__(self, entry):
        for i in range(len(self.filters)):
            if str(entry) + str(i) + self.salt not in self.filters[i]:
                return i % 2 == 1
        return (len(self.filters) - 1) % 2 == 0

    def size_in_bits(self):
        size = 0
        for bf in self.filters:
            size = size + bf.size_in_bits
        return size

    def count_set_bits(self):
        return sum(format(i, "08b").count("1") for i in self.filters[0].save_bytes())

    def calculate_entropy(self):
        s = "".join(format(i, "08b") for i in self.filters[0].save_bytes())
        p, lns = Counter(s), float(len(s))
        return (
            math.log2(lns) - sum(count * math.log2(count) for count in p.values()) / lns
        )
