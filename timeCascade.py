import cascadeUtils
import time


# create a padded cascade and time how long it takes to create it
# --- 0.5282530784606934 seconds --- before changes
def time_one_filter_cascade():
    revoked = cascadeUtils.generate_id_set(100000)
    valid = cascadeUtils.generate_id_set(100000)
    start = time.time()
    cascadeUtils.create_padded_cascade(revoked, valid, 100000, 100000)
    print("--- %s seconds ---" % (time.time() - start))


if __name__ == "__main__":
    time_one_filter_cascade()
