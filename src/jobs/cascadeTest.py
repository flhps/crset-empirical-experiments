import src.cascade.cascadeUtils as cu


def run(params):
    r = params["r"]
    s = params["s"]
    rhat = params["rhat"]
    parallelize = params["parallelize"]

    validIds = cu.gen_ids(r)
    revokedIds = cu.gen_ids_wo_overlap(s, validIds)

    cascade = cu.try_cascade(
        validIds,
        revokedIds,
        rhat,
        multi_process=parallelize,
    )[0]

    for elem in validIds:
        if elem not in cascade:
            return {"message": "Failed to confirm expected member"}
    for elem in revokedIds:
        if elem in cascade:
            return {"message": "Failed to confirm expected exclusion"}

    return {"message": "Cascade works"}
