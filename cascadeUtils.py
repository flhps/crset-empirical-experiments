import uuid
from paddedCascade import PaddedCascade
import random


def generate_id_set(size):
    return [uuid.uuid4() for _ in range(size)]
    # return [random.randint(0, 2**128) for _ in range(size)]


def create_padded_cascade(
    revokedids, validids, revokedmax, validmax, fprs=None, multi_threaded=False
):
    test_cascade = None
    tries = 0
    while not test_cascade:
        try:
            test_cascade = PaddedCascade(
                revokedids, validids, revokedmax, validmax, fprs, multi_threaded
            )
        except Exception as e:
            tries = tries + 1
            print("Trying cascade generation again after it failed: %s", e)
            if tries > 5:
                break
    if not test_cascade:
        raise Exception(
            f"Cascade construction failed repeatedly for {len(revokedids)} inclusions and {len(validids)} exclusions with {fprs} fpr targets"
        )
    return test_cascade


def vectorize_cascade(cascade):
    return [
        float(cascade.size_in_bits()),
        float(len(cascade.filters)),
        float(cascade.filters[0].size_in_bits),
        float(cascade.count_set_bits()),
        cascade.calculate_entropy(),
    ]


# function that returns the number of elements in a vectorized cascade
# wanted to keep this dynamic
def vectorized_cascade_size():
    cascade = create_padded_cascade([], [], 1, 1)
    vector = vectorize_cascade(cascade)
    return len(vector)
