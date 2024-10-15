import cascadeUtils
import time


# create a padded cascade and time how long it takes to create it
def time_one_filter_cascade(incl, excl, multi_process):
    revoked = cascadeUtils.generate_id_set(incl)
    valid = cascadeUtils.generate_id_set(excl)
    start = time.time()
    cascadeUtils.create_padded_cascade(
        revoked, valid, incl, excl, multi_process=multi_process
    )
    print(
        "--- %s seconds --- (%s)"
        % (time.time() - start, "MT" if multi_process else "ST")
    )


if __name__ == "__main__":
    time_one_filter_cascade(10000000, 10000000, False)
    time_one_filter_cascade(10000000, 10000000, True)
