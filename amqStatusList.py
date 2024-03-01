from rbloom import Bloom
import uuid
from hashlib import sha256
from pickle import dumps
import math
from collections import Counter


def hash_func(obj):
    h = sha256(dumps(obj)).digest()
    return int.from_bytes(h[:16], "big") - 2 ** 127


def new_bloom(size, fpr):
    # if the filter is not slightly oversized, the cascade construction fails too often
    return Bloom(math.ceil(1.1 * size), fpr, hash_func)


class FilterCascade:
    def __init__(self, positives, negatives, fprs=None):
        if fprs is None:
            fprs = [0.006]
        self.filters = []
        self.salt = str(uuid.uuid4())
        self.__help_build_cascade(positives, negatives, fprs)

    def __help_build_cascade(self, positives, negatives, fprs):
        # print(f"Cascade build level: {len(self.filters)}")
        fpr = fprs[-1]
        if len(fprs) > len(self.filters):
            fpr = fprs[len(self.filters)]
        bloom = new_bloom(len(positives), fpr)
        for elem in positives:
            bloom.add(str(elem) + self.salt)
        fps = []
        for elem in negatives:
            if str(elem) + self.salt in bloom:
                fps.append(elem)
        self.filters.append(bloom)
        if len(fps) == 0:
            return
        if len(fps) == len(negatives):
            raise Exception("Cascade cannot solve")
        self.__help_build_cascade(fps, positives, fprs)

    def __contains__(self, entry):
        for i in range(len(self.filters)):
            if str(entry) + self.salt not in self.filters[i]:
                return i % 2 == 1
        return (len(self.filters) - 1) % 2 == 0

    def size_in_bits(self):
        size = 0
        for bf in self.filters:
            size = size + bf.size_in_bits
        return size

    def count_set_bits(self):
        return sum(format(i, '08b').count('1') for i in self.filters[0].save_bytes())

    def calculate_entropy(self):
        s = ''.join(format(i, '08b') for i in self.filters[0].save_bytes())
        p, lns = Counter(s), float(len(s))
        return math.log2(lns) - sum(count * math.log2(count) for count in p.values()) / lns
