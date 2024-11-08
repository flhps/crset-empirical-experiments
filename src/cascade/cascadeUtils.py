from src.cascade.statusCascade import StatusCascade
from src.cascade.unpaddedStatusCascade import UnpaddedStatusCascade
import secrets


def gen_ids_wo_overlap(n, forbidden):
    if n == 0:
        return set()
    res = {secrets.randbits(256) for _ in range(n)}
    res = res - forbidden
    while len(res) < n:
        cand = secrets.randbits(256)
        if cand not in forbidden:
            res.add(cand)
    return res


def gen_ids(n):
    if n == 0:
        return set()
    res = {secrets.randbits(256) for _ in range(n)}
    # deal with possible random collisions
    while len(res) < n:
        res.add(secrets.randbits(256))
    return res


def try_some_cascade(
    r,
    s,
    rHat,
    p=0.5,
    k=1,
    multi_process=False,
    use_padding=True,
):
    validIds = gen_ids(r)
    revokedIds = gen_ids_wo_overlap(s, validIds)
    return try_cascade(validIds, revokedIds, rHat, p, k, multi_process, use_padding)


def try_cascade(validIds, revokedIds, rHat, p=0.5, k=1, multi_process=False, use_padding=True):
    cascade = None
    tries = 0
    while not cascade:
        try:
            if use_padding:
                cascade = StatusCascade(
                    validIds, revokedIds, rHat, p=p, k=k, multi_process=multi_process
                )
            else:
                cascade = UnpaddedStatusCascade(
                    validIds, revokedIds, p=p, k=k, multi_process=multi_process
                )
        except Exception as e:
            tries = tries + 1
            print("Trying cascade generation again after it failed: %s" % e)
            if tries > 50:
                break
    if not cascade:
        raise Exception(
            f"Cascade construction failed repeatedly for {len(validIds)} inclusions and {len(revokedIds)} exclusions with {p} as p"
        )
    return (cascade, tries + 1)


def vectorize_cascade(cascade):
    return [
        float(cascade.size_in_bits()),
        float(len(cascade.filters)),
        float(cascade.filters[0].size_in_bits),
        float(cascade.count_set_bits()),
        cascade.calculate_entropy(),
    ]