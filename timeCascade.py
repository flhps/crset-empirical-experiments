import cascadeUtils
import time
import math


# create a padded cascade and time how long it takes to create it
def time_one_filter_cascade(incl, excl, multi_process):
    revoked = cascadeUtils.generate_id_set(incl)
    valid = cascadeUtils.generate_id_set(excl)
    start = time.time()
    cascade = cascadeUtils.create_padded_cascade(
        revoked,
        valid,
        incl,
        excl,
        fprs=[min(incl * math.sqrt(0.5) / excl, 0.5), 0.5],
        multi_process=multi_process,
    )
    print(
        "--- %s seconds --- (%s)"
        % (time.time() - start, "MT" if multi_process else "ST")
    )
    print("--- %s bytes ---" % (cascade.size_in_bits() / 8))


if __name__ == "__main__":
    time_one_filter_cascade(10000000, 10000000, False)
    time_one_filter_cascade(10000000, 10000000, True)
